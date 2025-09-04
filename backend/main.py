import sys
import os
import re
import cv2
import torch
import asyncio
import logging
import threading
import pathlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("box-app")

# ─────────────────────────────────────────────────────────────────────────────
# Paths / YOLOv5 local repo setup
# ─────────────────────────────────────────────────────────────────────────────
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)                     # .../backend/my-app
YOLOV5_PATH = os.path.join(ROOT, "yolov5")       # .../backend/my-app/yolov5

if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Patch pathlib for Windows if ever needed (Render is Linux, but harmless)
if os.name == 'nt':
    class PosixPathPatch(pathlib.PosixPath):
        def __new__(cls, *args, **kwargs):
            return pathlib.WindowsPath(*args, **kwargs)
    pathlib.PosixPath = PosixPathPatch

# ─────────────────────────────────────────────────────────────────────────────
# Global App State
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.is_processing: bool = False
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        self.loaded_models: dict[str, any] = {}
        self.current_model: any = None
        self.current_model_name: str = ""

        self.processing_thread: threading.Thread | None = None
        self.websocket: WebSocket | None = None
        self.main_loop: asyncio.AbstractEventLoop | None = None

        self.box_counts: defaultdict[str, int] = defaultdict(int)

    def reset_runtime(self):
        """Reset runtime-only counters/flags (keeps loaded models cached)."""
        self.is_processing = False
        self.start_time = None
        self.end_time = None
        self.box_counts.clear()

    def stop_thread_if_running(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        self.processing_thread = None


app_state = AppState()

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Box Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    video_path: str
    vehicle_number: str
    supervisor_name: str
    model_name: str  # one of: multi_box_counter | box_counter_rtl | box_counter_ltr


class ProcessResponse(BaseModel):
    start_time: str
    end_time: str
    vehicle_number: str
    supervisor_name: str
    report_data: dict
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (YOLOv5 via local repo)
# ─────────────────────────────────────────────────────────────────────────────
def load_model_dynamically(model_path_rel: str):
    """
    Load a YOLOv5 model from local weights using the local yolov5 repo.
    Caches models by path.
    """
    # Normalize to absolute path (weights are placed next to this file)
    weight_path = os.path.join(ROOT, model_path_rel)

    if weight_path in app_state.loaded_models:
        logger.info(f"Using cached model: {weight_path}")
        return app_state.loaded_models[weight_path]

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model {weight_path} to device: {device}")

        # Use local YOLOv5 repo to load custom weights
        model = torch.hub.load(YOLOV5_PATH, 'custom', path=weight_path, source='local', verbose=False)
        model.to(device)

        # Optional: ensure FP32 on CPU (helps on tiny Render instances)
        if device == 'cpu':
            model.float()

        app_state.loaded_models[weight_path] = model
        logger.info(f"✅ Model loaded: {weight_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model {weight_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Detection / Counting Loop (NO cv2.imshow — suitable for headless servers)
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_loop(video_path: str, vehicle_number: str, supervisor_name: str, model_config: dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Could not open video source: {video_path}")
        app_state.is_processing = False
        return

    model = app_state.current_model
    if model is None:
        logger.error("❌ No model set in app_state.current_model")
        app_state.is_processing = False
        cap.release()
        return

    # Apply confidence threshold if present
    if "conf" in model_config:
        try:
            model.conf = float(model_config["conf"])
        except Exception:
            pass

    # Build label → class_id mapping from model.names (lowercased compare)
    target_labels = [s.lower() for s in model_config["labels"]]
    target_class_ids = [i for i, name in model.names.items() if str(name).lower() in target_labels]

    direction = model_config["direction"]
    line_pos_ratio = float(model_config["line_pos"])

    tracked_objects: dict[int, dict] = {}
    next_object_id = 0

    line_x = None  # resolve once we know frame width

    try:
        while app_state.is_processing:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video source ended or was interrupted.")
                break

            if line_x is None:
                w = frame.shape[1]
                line_x = int(w * line_pos_ratio)

            # YOLO expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            # (x1,y1,x2,y2,conf,cls)
            det = results.xyxy[0].cpu().numpy() if hasattr(results, "xyxy") else np.empty((0, 6))

            current = []
            for *xyxy, conf, cls in det:
                if int(cls) in target_class_ids:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    current.append({
                        "box": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "class_id": int(cls),
                    })

            # Greedy nearest-neighbor matching to existing trackers
            unmatched_trackers = list(tracked_objects.keys())
            for det_info in current:
                matched_id = None
                min_dist = 100.0  # pixel threshold
                for tid in unmatched_trackers:
                    px, py = tracked_objects[tid]['center']
                    cx, cy = det_info['center']
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = tid

                if matched_id is not None:
                    # Check crossing
                    prev_x = tracked_objects[matched_id]['center'][0]
                    curr_x = det_info['center'][0]

                    if not tracked_objects[matched_id]['counted']:
                        crossed = False
                        if direction == 'rtl' and prev_x > line_x and curr_x <= line_x:
                            crossed = True
                        elif direction == 'ltr' and prev_x < line_x and curr_x >= line_x:
                            crossed = True

                        if crossed:
                            tracked_objects[matched_id]['counted'] = True
                            label = model.names[det_info['class_id']]
                            app_state.box_counts[label] += 1
                            logger.info(f"Counted ID {matched_id} as '{label}' → {dict(app_state.box_counts)}")

                            # Push to WS, if connected
                            if app_state.websocket and app_state.main_loop:
                                try:
                                    asyncio.run_coroutine_threadsafe(
                                        app_state.websocket.send_json({"counts": dict(app_state.box_counts)}),
                                        app_state.main_loop
                                    )
                                except Exception:
                                    pass

                    # Update tracker state
                    tracked_objects[matched_id]['box'] = det_info['box']
                    tracked_objects[matched_id]['center'] = det_info['center']
                    unmatched_trackers.remove(matched_id)

                else:
                    tracked_objects[next_object_id] = {
                        "box": det_info['box'],
                        "center": det_info['center'],
                        "counted": False,
                        "class_id": det_info['class_id'],
                    }
                    next_object_id += 1

            # Remove trackers not matched this frame
            for tid in unmatched_trackers:
                del tracked_objects[tid]

        # Done
    finally:
        cap.release()

    # Send final report
    cleanup_and_email(
        vehicle_number=vehicle_number,
        supervisor_name=supervisor_name,
        report_data={"box_counts": dict(app_state.box_counts)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup + Email
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_and_email(vehicle_number: str, supervisor_name: str, report_data: dict):
    logger.info("Cleaning up processing.")
    app_state.is_processing = False
    app_state.end_time = datetime.now()

    details = ProcessResponse(
        start_time=app_state.start_time.strftime('%Y-%m-%d %H:%M:%S') if app_state.start_time else "",
        end_time=app_state.end_time.strftime('%Y-%m-%d %H:%M:%S'),
        vehicle_number=vehicle_number,
        supervisor_name=supervisor_name,
        report_data=report_data,
        message="Processing finished.",
    )
    send_email(details)


def send_email(details: ProcessResponse):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("APP_PASSWORD")

    if not sender_email or not receiver_email or not password:
        logger.warning("⚠️ Email not sent. Missing SENDER_EMAIL / RECEIVER_EMAIL / APP_PASSWORD env vars.")
        return

    subject = f"YOLOv5 Report: {app_state.current_model_name}"

    # Build report text
    box_counts = details.report_data.get("box_counts", {})
    if app_state.current_model_name == "multi_box_counter":
        counts_str = "\n".join([f"{k}: {v}" for k, v in box_counts.items()])
        weighted_total = 0
        for k, v in box_counts.items():
            m = re.search(r'\d+', str(k))
            if m:
                weighted_total += int(m.group(0)) * int(v)
        body_counts = f"Counts:\n{counts_str}\n--------------------------\nTotal Value: {weighted_total}\n"
    else:
        total = sum(int(v) for v in box_counts.values())
        body_counts = f"Total Count: {total}\n"

    body = (
        "Processing Report\n"
        f"Model Used: {app_state.current_model_name}\n"
        "--------------------------\n"
        f"Vehicle Number: {details.vehicle_number}\n"
        f"Supervisor: {details.supervisor_name}\n"
        "--------------------------\n"
        f"Start Time: {details.start_time}\n"
        f"End Time: {details.end_time}\n"
        "--------------------------\n"
        f"{body_counts}"
    )

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
        logger.info("📧 Email sent successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    app_state.main_loop = asyncio.get_running_loop()
    logger.info("Startup complete.")

@app.get("/")
def root():
    return {"status": "ok", "message": "🚀 Box Detection API is running!"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app_state.websocket = websocket
    try:
        # Keep connection open (client can send any ping text)
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        if app_state.websocket is websocket:
            app_state.websocket = None

@app.post("/stop_processing")
async def stop_processing():
    if not app_state.is_processing:
        return {"message": "No active processing to stop."}
    logger.info("🛑 Stop requested.")
    app_state.is_processing = False
    app_state.stop_thread_if_running()
    return {"message": "Processing stopped."}

@app.post("/start_processing")
async def start_processing(data: ProcessRequest):
    if app_state.is_processing:
        return {"message": "Processing is already in progress."}

    # Accept local path or URL
    video_path = data.video_path
    is_url = video_path.lower().startswith(("rtsp://", "http://", "https://"))
    if not is_url and not os.path.exists(video_path):
        return {"error": f"Local video file not found at: {video_path}"}

    # Model configs
    model_configs = {
        "multi_box_counter": {
            "path": "box_counter.pt",
            "conf": 0.5,
            "labels": ["4box", "5box", "6box"],
            "line_pos": 0.5,
            "direction": "rtl",
        },
        "box_counter_rtl": {
            "path": "best2.pt",
            "conf": 0.45,
            "labels": ["box"],
            "line_pos": 0.8,
            "direction": "rtl",
        },
        "box_counter_ltr": {
            "path": "best.pt",
            "conf": 0.5,
            "labels": ["box"],
            "line_pos": 0.2,
            "direction": "ltr",
        },
    }

    if data.model_name not in model_configs:
        return {"error": f"Invalid model name: {data.model_name}"}

    config = model_configs[data.model_name]
    model = load_model_dynamically(config["path"])
    if model is None:
        return {"error": f"Failed to load model: {config['path']}"}

    # Bind runtime state
    app_state.current_model = model
    app_state.current_model_name = data.model_name

    # Reset runtime counters & start thread
    app_state.reset_runtime()
    app_state.is_processing = True
    app_state.start_time = datetime.now()

    t = threading.Thread(
        target=run_detection_loop,
        args=(video_path, data.vehicle_number, data.supervisor_name, config),
        daemon=True,
    )
    app_state.processing_thread = t
    t.start()

    return {"message": f"Processing started with model: {data.model_name}"}


# ─────────────────────────────────────────────────────────────────────────────
# Local dev runner (Render will use your Start Command)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
