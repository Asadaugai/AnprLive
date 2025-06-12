#Count Save into the txt file and combine the streaming for both models || Show the total vehicle count and total detected plate count for the selected date only
import cv2
import os
import subprocess
import numpy as np
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
from paddleocr import PaddleOCR
import threading
import queue
from datetime import datetime
import pandas as pd
import math

# ----------------- Configuration -----------------
load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")
FFMPEG_PATH = r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe'
WIDTH, HEIGHT = 1920, 1080
FRAME_SKIP = 2
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
LINE1_PT1 = (100, 600)
LINE1_PT2 = (1850, 600)
LINE1_THRESHOLD = 25
LINE2_PT1 = (1250, 600)
LINE2_PT2 = (1850, 600)
LINE2_THRESHOLD = 25
COUNT_FILE = "vehicle_counts.txt"

# ----------------- Helper Functions -----------------

def load_models():
    plate_model = YOLO("license_plate_detector.pt")
    vehicle_model = YOLO("yolov8s.pt")
    return plate_model, vehicle_model

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def align_plate(plate_img):
    return cv2.resize(plate_img, (240, 80))

def score_plate_view(plate_img, bbox):
    sharpness = calculate_sharpness(plate_img)
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return sharpness + area * 0.01

def read_plate_text(image, ocr):
    image = align_plate(image)
    result = ocr.ocr(image, cls=False)
    if result and result[0]:
        lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
        return " ".join(lines) if lines else "Unknown"
    return "Unknown"

def draw_plate_detections(frame, results, ocr, track_data, frame_idx):
    for box in results:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
        confidence = box.conf[0] if hasattr(box, 'conf') and box.conf is not None else 0.0
        if track_id == -1:
            continue

        plate_img = frame[y1:y2, x1:x2]
        score = score_plate_view(plate_img, (x1, y1, x2, y2))
        current_best = track_data.get(track_id, {})

        if not current_best or score > current_best.get("score", 0):
            number = read_plate_text(plate_img, ocr)
            if number and number != "Unknown":
                track_data[track_id] = {
                    "img": plate_img,
                    "score": score,
                    "number": number,
                    "last_seen": frame_idx
                }
        else:
            track_data[track_id]["last_seen"] = frame_idx

        number_to_show = track_data.get(track_id, {}).get("number", "Detecting...")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, number_to_show, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def cleanup_tracks(track_data, finalized_list, finalized_ids, frame_idx, max_age=50):
    to_remove = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for tid, data in track_data.items():
        if frame_idx - data.get("last_seen", 0) > max_age:
            if tid not in finalized_ids:
                number = data.get("number", "")
                if (number and number != "Unknown") and (number not in finalized_list):
                    finalized_list.append(number)
                    with open("final_plate_numbers.txt", "a") as f:
                        f.write(f"{current_time},{number}\n")
                    finalized_ids.add(tid)
            to_remove.append(tid)

    for tid in to_remove:
        del track_data[tid]

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

def draw_vehicle_detections(frame, results, counted_ids_line1, counted_ids_line2):
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                if class_name not in VEHICLE_CLASSES:
                    continue
                track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
                if track_id == -1:
                    continue
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if track_id not in counted_ids_line1:
                    dist_line1 = point_line_distance((cx, cy), LINE1_PT1, LINE1_PT2)
                    if dist_line1 <= LINE1_THRESHOLD:
                        counted_ids_line1.add(track_id)
                        save_vehicle_count("Enter", 1)
                        st.session_state.vehicle_count_line1 += 1
                if track_id not in counted_ids_line2:
                    dist_line2 = point_line_distance((cx, cy), LINE2_PT1, LINE2_PT2)
                    if dist_line2 <= LINE2_THRESHOLD:
                        counted_ids_line2.add(track_id)
                        save_vehicle_count("Exit", 1)
                        st.session_state.vehicle_count_line2 += 1
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'class_name': class_name,
                    'conf': conf,
                    'centroid': (cx, cy)
                })
    return detections

def start_rtsp_stream(rtsp_url, width=1920, height=1080, queue_size=1000):
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
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3)).copy()
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        pipe.terminate()
    threading.Thread(target=reader_thread, daemon=True).start()
    return frame_queue

def filter_plates_by_date(selected_date):
    try:
        with open("final_plate_numbers.txt", "r") as f:
            lines = f.readlines()
        filtered_plates = []
        for line in lines:
            timestamp, plate = line.strip().split(',')
            date = timestamp.split(' ')[0]
            if date == selected_date.strftime("%Y-%m-%d"):
                filtered_plates.append({"Timestamp": timestamp, "Plate Number": plate})
        return filtered_plates
    except FileNotFoundError:
        return []

def save_vehicle_count(direction, increment):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = current_time.split(' ')[0]
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        counts_by_date = {}
        for line in lines:
            if line.strip():
                timestamp, dir_, count = line.strip().split(',')
                date = timestamp.split(' ')[0]
                if date not in counts_by_date:
                    counts_by_date[date] = {"Enter": 0, "Exit": 0}
                counts_by_date[date][dir_] = int(count)
    except (FileNotFoundError, ValueError):
        counts_by_date = {}

    if current_date not in counts_by_date:
        counts_by_date[current_date] = {"Enter": 0, "Exit": 0}

    if direction == "Enter":
        counts_by_date[current_date]["Enter"] += increment
    elif direction == "Exit":
        counts_by_date[current_date]["Exit"] += increment

    with open(COUNT_FILE, "w") as f:
        for date, counts in counts_by_date.items():
            f.write(f"{date} 00:00:00,Enter,{counts['Enter']}\n")
            f.write(f"{date} 00:00:00,Exit,{counts['Exit']}\n")



def load_vehicle_counts():
    current_date = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                timestamp, dir_, count = line.strip().split(',')
                date = timestamp.split(' ')[0]
                if date == current_date:
                    if dir_ == "Enter":
                        enter_count = int(count)
                    elif dir_ == "Exit":
                        exit_count = int(count)
        return enter_count, exit_count
    except (FileNotFoundError, ValueError):
        return 0, 0
    

def count_total_plates(selected_date):
    try:
        with open("final_plate_numbers.txt", "r") as f:
            lines = f.readlines()
        return len([line for line in lines if line.strip() and line.strip().split(',')[0].split(' ')[0] == selected_date.strftime("%Y-%m-%d")])
    except FileNotFoundError:
        return 0
    


def get_vehicle_count_by_date(selected_date):
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                timestamp, dir_, count = line.strip().split(',')
                date = timestamp.split(' ')[0]
                if date == selected_date.strftime("%Y-%m-%d"):
                    if dir_ == "Enter":
                        enter_count = int(count)
                    elif dir_ == "Exit":
                        exit_count = int(count)
        return enter_count, exit_count
    except (FileNotFoundError, ValueError):
        return 0, 0
    

    


def main():
    st.image("images.jpeg", width=200)
    st.title("PetroI")

    # Initialize session state with counts from file
    enter_count, exit_count = load_vehicle_counts()
    if "counted_ids_line1" not in st.session_state:
        st.session_state.counted_ids_line1 = set()
    if "counted_ids_line2" not in st.session_state:
        st.session_state.counted_ids_line2 = set()
    if "vehicle_count_line1" not in st.session_state:
        st.session_state.vehicle_count_line1 = enter_count
    if "vehicle_count_line2" not in st.session_state:
        st.session_state.vehicle_count_line2 = exit_count
    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0
    if "processed_frames" not in st.session_state:
        st.session_state.processed_frames = 0
    if "last_vehicle_detections" not in st.session_state:
        st.session_state.last_vehicle_detections = []

    # Load models and OCR
    plate_model, vehicle_model = load_models()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Start RTSP stream
    frame_queue = start_rtsp_stream(RTSP_URL, WIDTH, HEIGHT)

    # UI placeholders
    stream_placeholder = st.empty()
    today_plates_container = st.empty()  # Container for plates detected today
   

   
    
    # Create sidebar containers
    with st.sidebar:
        st.subheader("Filter Plates by Date")
        selected_date = st.date_input("Select Date", value=datetime.today(), key="date_filter_input")
        filtered_plates = filter_plates_by_date(selected_date)
        if filtered_plates:
            st.dataframe(filtered_plates, use_container_width=True, hide_index=True)
        else:
            st.write("No plates found for selected date.")

        live_plates_container = st.empty()  # Single container for live plates
        count1_container = st.empty()
        #count2_container = st.empty()
        total_plates_container = st.empty()  # New container for total plates


    # Initialize tracking data
    track_data = {}
    final_plate_numbers = []
    finalized_ids = set()
    frame_idx = 0

    # Initialize tracking data
    track_data = {}
    final_plate_numbers = []
    finalized_ids = set()
    frame_idx = 0

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                frame_combined = frame.copy()
                st.session_state.frame_counter += 1

                if st.session_state.frame_counter % FRAME_SKIP == 0:
                    st.session_state.processed_frames += 1

                    # Vehicle detection and tracking
                    vehicle_results = vehicle_model.track(frame_combined, persist=True, conf=0.40)
                    st.session_state.last_vehicle_detections = draw_vehicle_detections(
                        frame_combined, vehicle_results, st.session_state.counted_ids_line1, st.session_state.counted_ids_line2
                    )

                    # Plate detection and tracking
                    plate_results = plate_model.track(frame_combined, persist=True, conf=0.40)
                    for res in plate_results:
                        draw_plate_detections(frame_combined, res.boxes, ocr, track_data, frame_idx)
                    cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)

                else:
                    # Draw last vehicle detections on skipped frames
                    for detection in st.session_state.last_vehicle_detections:
                        x1, y1, x2, y2 = detection['bbox']
                        track_id = detection['track_id']
                        class_name = detection['class_name']
                        conf = detection['conf']
                        cx, cy = detection['centroid']
                        label = f"ID:{track_id} {class_name} {conf:.2f}"
                        cv2.rectangle(frame_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_combined, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame_combined, (cx, cy), 5, (255, 0, 0), -1)

                # Draw counting lines
                cv2.line(frame_combined, LINE1_PT1, LINE1_PT2, (0, 0, 255), 3)
                #cv2.line(frame_combined, LINE2_PT1, LINE2_PT2, (255, 255, 0), 3)
                cv2.putText(frame_combined, "ENTER", (LINE1_PT1[0], LINE1_PT1[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                #cv2.putText(frame_combined, "EXIT", (LINE2_PT1[0], LINE2_PT1[1] - 20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Convert frame to RGB
                frame_combined_rgb = cv2.cvtColor(frame_combined, cv2.COLOR_BGR2RGB)

                # Update main screen
                stream_placeholder.image(frame_combined_rgb, caption="Vehicle and License Plate Detection Stream", use_container_width=True)

                # Update sidebar: Live plates (only show valid plates)
                with live_plates_container.container():
                    st.subheader("Live Plates")
                    valid_plates = [
                        {"ID": tid, "Plate Number": data["number"]}
                        for tid, data in track_data.items()
                        if data.get("number") not in ["Detecting...", "Unknown"]
                    ]
                    if valid_plates:
                        st.dataframe(valid_plates, use_container_width=True, hide_index=True)
                    else:
                        st.write("Detecting...")

               
                # Update sidebar: Vehicle counts for selected date
                enter_count, exit_count = get_vehicle_count_by_date(selected_date)
                count1_container.markdown(f"**Vehicle Count :** {enter_count}")
                #count2_container.markdown(f"**Exit Count :** {exit_count}")
                total_plates_container.markdown(f"**Total Detected Plates :** {count_total_plates(selected_date)}")

                # Update today's plates below video stream
                with today_plates_container.container():
                    st.subheader("License Plates Detected Today")
                    today_date = datetime.now()
                    today_plates = filter_plates_by_date(today_date)
                    if today_plates:
                        st.dataframe(pd.DataFrame(today_plates), use_container_width=True, hide_index=True)
                    else:
                        st.write("No license plates detected today.")

                frame_idx += 1

            except queue.Empty:
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