import os
import sys
import time
import math
import smtplib
import ssl
import cv2
import torch
import numpy as np
import pathlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Set
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# DeepSORT tracker
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    raise ImportError("Please install deep_sort_realtime: pip install deep_sort_realtime") from e

# Load environment variables from the .env file for security (like your email password)
load_dotenv()

# We're creating a simple web server with FastAPI.
app = FastAPI()

# This is important! It lets your website talk to this brain, even though they are separate programs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any website to connect
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (like POST)
    allow_headers=["*"],  # Allow all headers
)

# This is like a little container for the information we get from the website.
class StartRequest(BaseModel):
    video_url: str
    supervisor_name: str
    vehicle_number: str
    model_name: str # New field to hold the model selection

class StopRequest(BaseModel):
    supervisor_name: Optional[str] = None
    vehicle_number: Optional[str] = None

# This is a global container that holds the current state of our detection.
class DetectionState:
    is_running: bool = False
    vehicle_number: str = ""
    supervisor_name: str = ""
    start_time: float = 0
    frame_count: int = 0
    box_count: int = 0
    video_path: str = ""
    # Store per-label counts and a total value
    crossed_counts: Dict[str, int] = {}
    total_value: int = 0
    
# This is a global object that holds the current state of our detection.
state = DetectionState()

# This is the hardcoded, in-memory storage for reports.
# It's a list that will hold all the completed reports.
reports: List[Dict] = []

# DeepSORT tracker param (frames to keep tracks)
TRACKER_MAX_AGE = 30

# This is a function to send the email report.
def send_email(to_email: str, subject: str, body: str):
    """Sends an email with the detection report."""
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SENDER_SMTP_SERVER")
    smtp_port = int(os.getenv("SENDER_SMTP_PORT", 587))
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        print("Email configuration is missing. Cannot send email.")
        return
        
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    # Turn the body text into a plain-text email part.
    part = MIMEText(body, "plain")
    message.attach(part)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)  # Secure the connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, message.as_string())
        print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# -------------------------
# USER CONFIGURABLE PART (YOUR CODE)
# -------------------------
YOLOV5_PATH = "yolov5" # local yolov5 repo folder
CONF_THRESH = 0.6 # detection confidence threshold
MODEL_WEIGHTS_PATHS = {
    "4,5,6 box": "bestb.pt",
    "single box": "best3.pt",
    "multiple box": "2best.pt",
}
# A cache to store loaded models
models_cache = {}

# A dictionary to define the line position for each model
LINE_POSITIONS = {
    "4,5,6 box": 0.30, # 30% from the left
    "single box": 0.60, # 60% from the left
    "multiple box": 0.10, # 10% from the left
}

# Fix Path issues on Windows when using pathlib.PosixPath
class PosixPathPatch(pathlib.PosixPath):
    def __new__(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
pathlib.PosixPath = PosixPathPatch

# Add local yolov5 to path so torch.hub.load(..., source='local') works
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_FULLPATH = os.path.join(ROOT, YOLOV5_PATH)
if YOLOV5_FULLPATH not in sys.path:
    sys.path.append(YOLOV5_FULLPATH)

def get_model(model_name: str):
    """Loads a model from a file or from the cache."""
    if model_name in models_cache:
        print(f"Using cached model: {model_name}")
        return models_cache[model_name]
    
    weights_path = MODEL_WEIGHTS_PATHS.get(model_name)
    if not weights_path:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    print(f"Loading YOLOv5 model for: {model_name} from {weights_path}")
    try:
        model = torch.hub.load(YOLOV5_FULLPATH, 'custom', path=weights_path, source='local')
        model.conf = CONF_THRESH
        models_cache[model_name] = model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Could not load the YOLOv5 model. Check your YOLOV5_PATH and WEIGHTS_PATH.")
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_name}")

# ----------------------------------------------------

# This is the heart of the "brain." It runs in a background thread.
def run_detection_with_popup(model_name: str):
    """
    This function processes the video and displays a pop-up window using cv2.imshow.
    """
    global state
    
    try:
        model = get_model(model_name)
    except HTTPException as e:
        print(e.detail)
        state.is_running = False
        return

    cap = cv2.VideoCapture(state.video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {state.video_path}")
        state.is_running = False
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracker and counting sets for this session
    tracker = DeepSort(max_age=TRACKER_MAX_AGE)
    counted_ids: Set[int] = set()
    state.total_value = 0
    state.crossed_counts = {}
    
    # Get the line position based on the selected model
    line_position_ratio = LINE_POSITIONS.get(model_name, 0.50) # Defaults to center if not found
    line_x = int(width * line_position_ratio)
    
    # The main processing loop
    while state.is_running:
        ret, frame = cap.read()
        if not ret:
            state.is_running = False
            break

        state.frame_count += 1
        # Draw the vertical line on the frame before processing detections
        # Lowered the brightness of the line
        cv2.line(frame, (line_x, 0), (line_x, height), (150, 150, 150), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        dets = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, cls
        
        # Prepare detections for DeepSort: [ [x,y,w,h], conf, class_name ]
        deepsort_dets = []
        for *xyxy, conf, cls in dets:
            x1, y1, x2, y2 = map(float, xyxy)
            w = x2 - x1
            h = y2 - y1
            class_name = model.names[int(cls)]
            deepsort_dets.append(([x1, y1, w, h], float(conf), class_name))
        
        # Update the tracker
        tracks = tracker.update_tracks(deepsort_dets, frame=frame)
        
        state.box_count = len(tracks)

        # Draw detections and check for line crossing
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Calculate the center point of the bounding box
            cx = int((x1 + x2) / 2)
            label = track.get_det_class()
            
            # Draw the bounding box and tracker ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- The Counting Logic ---
            # Condition: bbox_left <= line_x <= bbox_right
            if x1 <= line_x <= x2:
                if track_id not in counted_ids:
                    # Update individual count and total value based on the label
                    if label == 'box':
                        state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                        state.total_value += 1
                        counted_ids.add(track_id)
                    elif label == '4box':
                        state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                        state.total_value += 4
                        counted_ids.add(track_id)
                    elif label == '5box':
                        state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                        state.total_value += 5
                        counted_ids.add(track_id)
                    elif label == '6box':
                        state.crossed_counts[label] = state.crossed_counts.get(label, 0) + 1
                        state.total_value += 6
                        counted_ids.add(track_id)
            
        # --- Display counts on the frame ---
        y_offset = 50
        cv2.putText(frame, f"Total Value: {state.total_value}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset += 40
        for label, count in state.crossed_counts.items():
            cv2.putText(frame, f"{label}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

        # Display the frame in a pop-up window
        cv2.imshow("Detection (press 'q' to stop)", frame)
        
        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            state.is_running = False
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Video stream ended.")
    
    # Prepare and send the email report
    report_body = f"""
    Detection Report
    ----------------
    Supervisor: {state.supervisor_name}
    Vehicle Number: {state.vehicle_number}
    
    Frames Processed: {state.frame_count}
    Total Boxes Counted: {state.box_count}
    Total Value: {state.total_value}
    
    Individual Counts:
    {', '.join([f'{label}: {count}' for label, count in state.crossed_counts.items()])}
    """
    send_email("your_email_address@example.com", "Detection Complete", report_body)

    # Save the report to our hardcoded list
    reports.append({
        "supervisor_name": state.supervisor_name,
        "vehicle_number": state.vehicle_number,
        "frames_processed": state.frame_count,
        "total_boxes_counted": state.box_count,
        "total_value": state.total_value,
        "individual_counts": state.crossed_counts,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })


# This is the "start" API endpoint. The website calls this.
@app.post("/start-detection")
async def start_detection(request: StartRequest, background_tasks: BackgroundTasks):
    """Starts the box detection process."""
    global state
    if state.is_running:
        return {"message": "Detection is already in progress."}
    
    state.is_running = True
    state.video_path = request.video_url
    state.vehicle_number = request.vehicle_number
    state.supervisor_name = request.supervisor_name
    state.frame_count = 0
    state.box_count = 0
    state.start_time = time.time()
    
    print(f"Starting detection on video: {state.video_path} with model: {request.model_name}")
    background_tasks.add_task(run_detection_with_popup, model_name=request.model_name)
    
    return {"message": "Detection started. A pop-up window should appear."}


# This is the "stop" API endpoint. The website calls this.
@app.post("/stop-detection")
async def stop_detection(request: StopRequest):
    """Stops the box detection process."""
    global state
    if not state.is_running:
        return {"message": "No active processing to stop."}
    
    state.is_running = False
    
    # Clean up the pop-up window.
    cv2.destroyAllWindows()

    report_body = f"""
    Detection Report (Manual Stop)
    -----------------------------
    Supervisor: {state.supervisor_name}
    Vehicle Number: {state.vehicle_number}
    
    Frames Processed: {state.frame_count}
    Total Boxes Counted: {state.box_count}
    Total Value: {state.total_value}
    
    Individual Counts:
    {', '.join([f'{label}: {count}' for label, count in state.crossed_counts.items()])}
    """
    send_email("your_email_address@example.com", "Detection Manually Stopped", report_body)

    # Save the report to our hardcoded list
    reports.append({
        "supervisor_name": state.supervisor_name,
        "vehicle_number": state.vehicle_number,
        "frames_processed": state.frame_count,
        "total_boxes_counted": state.box_count,
        "total_value": state.total_value,
        "individual_counts": state.crossed_counts,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })

    return {"message": "Processing stopped."}
    
@app.get("/reports")
async def get_reports():
    """Returns the list of all hardcoded reports."""
    return {"reports": reports}
