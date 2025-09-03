import sys
import torch
import cv2
import numpy as np
import os
import pathlib
from collections import defaultdict
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import threading
import re

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- YOLOv5 and Path Setup ---
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_PATH = os.path.join(ROOT, 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

if os.name == 'nt':
    class PosixPathPatch(pathlib.PosixPath):
        def __new__(cls, *args, **kwargs):
            return pathlib.WindowsPath(*args, **kwargs)
    pathlib.PosixPath = PosixPathPatch

# --- Application State ---
class AppState:
    def __init__(self):
        self.is_processing = False
        self.start_time = None
        self.end_time = None
        self.loaded_models = {}
        self.current_model = None
        self.current_model_name = ""
        self.processing_thread = None
        self.websocket = None
        self.box_counts = defaultdict(int)
        self.main_loop = None

    def reset(self):
        self.is_processing = False
        self.start_time = None
        self.end_time = None
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        self.processing_thread = None
        self.box_counts.clear()

app_state = AppState()

# --- Model Loading ---
def load_model_dynamically(model_path):
    if model_path in app_state.loaded_models:
        logger.info(f"Using cached model: {model_path}")
        return app_state.loaded_models[model_path]
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model {model_path} onto device: {device}")
        model = torch.load(model_path, map_location=device)
        model.eval()
        app_state.loaded_models[model_path] = model
        logger.info(f"✅ Model {model_path} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model {model_path}: {e}")
        return None

# --- FastAPI App Initialization ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- Pydantic Models ---
class ProcessRequest(BaseModel):
    video_path: str
    vehicle_number: str
    supervisor_name: str
    model_name: str

class ProcessResponse(BaseModel):
    start_time: str
    end_time: str
    vehicle_number: str
    supervisor_name: str
    report_data: dict
    message: str


# ========================================================================
# UNIFIED DETECTION AND COUNTING LOGIC (without imshow for Render)
# ========================================================================
def run_detection_loop(video_path, vehicle_number, supervisor_name, model_config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Could not open video source: {video_path}")
        app_state.is_processing = False
        return

    model = app_state.current_model
    target_labels = model_config["labels"]
    target_class_ids = [
        i for i, name in model.names.items() if name.lower() in target_labels
    ]

    line_pos_ratio = model_config["line_pos"]
    line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * line_pos_ratio)
    direction = model_config["direction"]

    tracked_objects = {}
    next_object_id = 0

    while app_state.is_processing:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video source ended or was interrupted.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.xyxy[0].cpu().numpy()

        current_detections_info = []
        for *xyxy, conf, cls in detections:
            if int(cls) in target_class_ids:
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                current_detections_info.append({
                    'box': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'class_id': int(cls)
                })

        # --- Tracking Logic ---
        unmatched_trackers = list(tracked_objects.keys())
        for det_info in current_detections_info:
            matched_tracker_id = None
            min_dist = 100
            for tracker_id in unmatched_trackers:
                dist = np.sqrt(
                    (det_info['center'][0] - tracked_objects[tracker_id]['center'][0])**2 +
                    (det_info['center'][1] - tracked_objects[tracker_id]['center'][1])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    matched_tracker_id = tracker_id

            if matched_tracker_id is not None:
                px = tracked_objects[matched_tracker_id]['center'][0]
                cx = det_info['center'][0]

                if not tracked_objects[matched_tracker_id]['counted']:
                    counted = False
                    if direction == 'rtl' and px > line_x and cx <= line_x:
                        counted = True
                    elif direction == 'ltr' and px < line_x and cx >= line_x:
                        counted = True

                    if counted:
                        tracked_objects[matched_tracker_id]['counted'] = True
                        box_label = model.names[det_info['class_id']]
                        app_state.box_counts[box_label] += 1
                        logger.info(f"Counted ID {matched_tracker_id} as '{box_label}'. Counts: {dict(app_state.box_counts)}")
                        if app_state.websocket and app_state.main_loop:
                            asyncio.run_coroutine_threadsafe(
                                app_state.websocket.send_json({"counts": dict(app_state.box_counts)}),
                                app_state.main_loop
                            )

                tracked_objects[matched_tracker_id]['box'] = det_info['box']
                tracked_objects[matched_tracker_id]['center'] = det_info['center']
                unmatched_trackers.remove(matched_tracker_id)
            else:
                tracked_objects[next_object_id] = {
                    'box': det_info['box'],
                    'center': det_info['center'],
                    'counted': False,
                    'class_id': det_info['class_id']
                }
                next_object_id += 1

        for tracker_id in unmatched_trackers:
            del tracked_objects[tracker_id]

    cap.release()
    cleanup_and_email(vehicle_number, supervisor_name, {"box_counts": dict(app_state.box_counts)})


# --- Helper function for cleanup and email ---
def cleanup_and_email(vehicle_number, supervisor_name, report_data):
    logger.info("Cleaning up video processing thread.")
    app_state.is_processing = False
    app_state.end_time = datetime.now()

    response_details = ProcessResponse(
        start_time=app_state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        end_time=app_state.end_time.strftime('%Y-%m-%d %H:%M:%S'),
        vehicle_number=vehicle_number,
        supervisor_name=supervisor_name,
        report_data=report_data,
        message="Processing finished."
    )
    send_email(response_details)


# --- Email Sending Logic ---
def send_email(details: ProcessResponse):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("APP_PASSWORD")

    if not sender_email or not receiver_email or not password:
        logger.warning("⚠️ Email not sent. Missing environment variables.")
        return

    subject = f"YOLOv5 Report: {app_state.current_model_name}"

    report_body = ""
    box_counts = details.report_data.get("box_counts", {})
    if app_state.current_model_name == "multi_box_counter":
        counts_str = "\n".join([f"{box_type}: {count}" for box_type, count in box_counts.items()])
        weighted_total = 0
        for box_type, count in box_counts.items():
            num_match = re.search(r'\d+', box_type)
            if num_match:
                weighted_total += int(num_match.group(0)) * count
        report_body = f"Counts:\n{counts_str}\n--------------------------\nTotal Value: {weighted_total}\n"
    elif app_state.current_model_name in ["box_counter_rtl", "box_counter_ltr"]:
        total_count = sum(box_counts.values())
        report_body = f"Total Count: {total_count}\n"

    body = (f"Processing Report\n"
            f"Model Used: {app_state.current_model_name}\n--------------------------\n"
            f"Vehicle Number: {details.vehicle_number}\nSupervisor: {details.supervisor_name}\n--------------------------\n"
            f"Start Time: {details.start_time}\nEnd Time: {details.end_time}\n--------------------------\n"
            f"{report_body}")

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        logger.info("📧 Email sent successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")


# --- API Endpoints ---
@app.on_event("startup")
def startup_event():
    app_state.main_loop = asyncio.get_running_loop()
    logger.info("Application startup complete.")

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    await websocket.accept()
    app_state.websocket = websocket
    await websocket.receive_text()

@app.post("/stop_processing")
async def stop_processing():
    if not app_state.is_processing:
        return {"message": "No active processing to stop."}
    logger.info("🛑 Stop signal received.")
    app_state.is_processing = False
    if app_state.processing_thread:
        app_state.processing_thread.join(timeout=5)
    return {"message": "Processing stopped."}

@app.post("/start_processing")
async def start_processing(data: ProcessRequest):
    if app_state.is_processing:
        return {"message": "Processing is already in progress."}

    video_path = data.video_path
    is_url = video_path.lower().startswith(('rtsp://', 'http://', 'https://'))

    if not is_url and not os.path.exists(video_path):
        return {"error": f"Local video file not found at: {video_path}"}

    model_configs = {
        "multi_box_counter": {
            "path": "box_counter.pt", "conf": 0.5, "labels": ["4box", "5box", "6box"],
            "line_pos": 0.5, "direction": "rtl"
        },
        "box_counter_rtl": {
            "path": "best2.pt", "conf": 0.45, "labels": ["box"],
            "line_pos": 0.8, "direction": "rtl"
        },
        "box_counter_ltr": {
            "path": "best.pt", "conf": 0.5, "labels": ["box"],
            "line_pos": 0.2, "direction": "ltr"
        },
    }
    selected_model_key = data.model_name
    if selected_model_key not in model_configs:
        return {"error": f"Invalid model name: {selected_model_key}"}

    config = model_configs[selected_model_key]
    model = load_model_dynamically(config["path"])
    if not model:
        return {"error": f"Failed to load model: {config['path']}"}

    app_state.current_model = model
    app_state.current_model_name = selected_model_key
    app_state.reset()
    app_state.is_processing = True
    app_state.start_time = datetime.now()

    app_state.processing_thread = threading.Thread(
        target=run_detection_loop,
        args=(data.video_path, data.vehicle_number, data.supervisor_name, config),
        daemon=True
    )
    app_state.processing_thread.start()

    return {"message": f"Processing started with model: {selected_model_key}"}
