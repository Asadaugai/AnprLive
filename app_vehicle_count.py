
# Streamlit vehicle counting system with robust streaming and two counting lines
import cv2
import os
import subprocess
import numpy as np
import math
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import threading
import queue

# ----------------- Configuration -----------------

RTSP_URL = os.getenv("RTSP_URL")
FFMPEG_PATH = r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe'
WIDTH, HEIGHT = 1920, 1080
FRAME_SKIP = 0
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Line 1 configuration (e.g., for incoming traffic)
LINE1_PT1 = (100, 600)
LINE1_PT2 = (1100, 600)
LINE1_THRESHOLD = 25

# Line 2 configuration (e.g., for outgoing traffic)
LINE2_PT1 = (1250, 600)
LINE2_PT2 = (1850, 600)
LINE2_THRESHOLD = 25

# ----------------- Helper Functions -----------------

def point_line_distance(pt, line_pt1, line_pt2):
    x, y = pt
    x1, y1 = line_pt1
    x2, y2 = line_pt2

    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return math.sqrt(dx * dx + dy * dy)

def start_rtsp_stream(rtsp_url, width=1920, height=1080, queue_size=1000):
    """Start RTSP stream with robust buffering using threading and queue"""
    ffmpeg_cmd = [
        FFMPEG_PATH,
        '-rtsp_transport', 'tcp',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', rtsp_url,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    frame_queue = queue.Queue(maxsize=queue_size)

    def reader_thread():
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
        while True:
            raw_image = pipe.stdout.read(width * height * 3)
            if not raw_image:
                break
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                # Drop frames if queue is full to prevent buffering issues
                pass
        pipe.terminate()

    threading.Thread(target=reader_thread, daemon=True).start()
    return frame_queue

# ----------------- Main Function -----------------

def main():
    # Streamlit setup
    st.title("Vehicle Counting Live Stream")
    video_placeholder = st.empty()  # Placeholder for video feed
    
    st.sidebar.title("Vehicle Counts")
    count1_placeholder = st.sidebar.empty()  # Placeholder for Line 1 count
    count2_placeholder = st.sidebar.empty()  # Placeholder for Line 2 count
    status_placeholder = st.sidebar.empty()  # Status information

    # Initialize session state variables
    if "counted_ids_line1" not in st.session_state:
        st.session_state.counted_ids_line1 = set()
    if "counted_ids_line2" not in st.session_state:
        st.session_state.counted_ids_line2 = set()
    if "vehicle_count_line1" not in st.session_state:
        st.session_state.vehicle_count_line1 = 0
    if "vehicle_count_line2" not in st.session_state:
        st.session_state.vehicle_count_line2 = 0
    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0
    if "processed_frames" not in st.session_state:
        st.session_state.processed_frames = 0
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = []

    model = YOLO("yolov8s.pt")  # Load YOLO model
    
    # Start the robust RTSP stream
    frame_queue = start_rtsp_stream(RTSP_URL, WIDTH, HEIGHT)

    try:
        while True:
            try:
                # Get frame from queue with timeout
                frame = frame_queue.get(timeout=1)
                frame = frame.copy()  # Make writable copy
                st.session_state.frame_counter += 1

                # Process detection only every FRAME_SKIP frame
                #if st.session_state.frame_counter % FRAME_SKIP == 0:
                if FRAME_SKIP == 0 or st.session_state.frame_counter % FRAME_SKIP == 0:
                    st.session_state.processed_frames += 1
                    
                    # Run detection & tracking on frame
                    results = model.track(frame, persist=True)
                    
                    # Clear previous detections and store new ones
                    st.session_state.last_detections = []

                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                class_name = model.names[cls]

                                if class_name not in VEHICLE_CLASSES:
                                    continue

                                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
                                if track_id == -1:
                                    continue

                                # Centroid of bounding box
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2

                                # Check Line 1
                                if track_id not in st.session_state.counted_ids_line1:
                                    dist_line1 = point_line_distance((cx, cy), LINE1_PT1, LINE1_PT2)
                                    if dist_line1 <= LINE1_THRESHOLD:
                                        st.session_state.vehicle_count_line1 += 1
                                        st.session_state.counted_ids_line1.add(track_id)

                                # Check Line 2
                                if track_id not in st.session_state.counted_ids_line2:
                                    dist_line2 = point_line_distance((cx, cy), LINE2_PT1, LINE2_PT2)
                                    if dist_line2 <= LINE2_THRESHOLD:
                                        st.session_state.vehicle_count_line2 += 1
                                        st.session_state.counted_ids_line2.add(track_id)

                                # Store detection data for reuse in skipped frames
                                detection_data = {
                                    'bbox': (x1, y1, x2, y2),
                                    'track_id': track_id,
                                    'class_name': class_name,
                                    'conf': conf,
                                    'centroid': (cx, cy)
                                }
                                st.session_state.last_detections.append(detection_data)

                # Draw bounding boxes on every frame (using last detections for skipped frames)
                for detection in st.session_state.last_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    track_id = detection['track_id']
                    class_name = detection['class_name']
                    conf = detection['conf']
                    cx, cy = detection['centroid']
                    
                    label = f"ID:{track_id} {class_name} {conf:.2f}"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw centroid
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Draw counting lines on every frame for consistent visualization
                cv2.line(frame, LINE1_PT1, LINE1_PT2, (0, 0, 255), 3)  # Line 1 in red
                cv2.line(frame, LINE2_PT1, LINE2_PT2, (255, 255, 0), 3)  # Line 2 in yellow
                
                # Add line labels
                cv2.putText(frame, "ENTER", (LINE1_PT1[0], LINE1_PT1[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "EXIT", (LINE2_PT1[0], LINE2_PT1[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Add frame counter
                cv2.putText(frame, f"Frame: {st.session_state.frame_counter}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update video feed with use_container_width
                video_placeholder.image(frame_rgb, caption="Live Stream", use_container_width=True)

                # Update sidebar counts
                count1_placeholder.markdown(f"**Enter:** {st.session_state.vehicle_count_line1}")
                count2_placeholder.markdown(f"**Exit:** {st.session_state.vehicle_count_line2}")
                
             

            except queue.Empty:
                # Continue if no frame available
                continue
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                continue

    except KeyboardInterrupt:
        st.write("Stream stopped by user")
    except Exception as e:
        st.error(f"Stream error: {e}")

if __name__ == "__main__":
    main()
