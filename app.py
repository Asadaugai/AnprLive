
#Combined3/Count store in the file and show one streaming for both models
import cv2
import os
import numpy as np
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import math

# ----------------- Configuration -----------------
WIDTH, HEIGHT = 1920, 1080
FRAME_SKIP = 1
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
LINE1_PT1 = (100, 400)
LINE1_PT2 = (1100, 400)
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
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                time, dir_, count = line.strip().split(',')
                if dir_ == "Enter":
                    enter_count = int(count)
                elif dir_ == "Exit":
                    exit_count = int(count)
    except (FileNotFoundError, ValueError):
        enter_count = 0
        exit_count = 0

    if direction == "Enter":
        enter_count += increment
    elif direction == "Exit":
        exit_count += increment

    with open(COUNT_FILE, "w") as f:
        f.write(f"{current_time},Enter,{enter_count}\n")
        f.write(f"{current_time},Exit,{exit_count}\n")

def load_vehicle_counts():
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                time, dir_, count = line.strip().split(',')
                if dir_ == "Enter":
                    enter_count = int(count)
                elif dir_ == "Exit":
                    exit_count = int(count)
        return enter_count, exit_count
    except (FileNotFoundError, ValueError):
        return 0, 0

# ----------------- MAIN APPLICATION -----------------

def main():
    st.title("Vehicle Counting and License Plate Detection from Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if not uploaded_file:
        st.warning("Please upload a video to start processing.")
        return

    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        return

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

    plate_model, vehicle_model = load_models()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    stream_placeholder = st.empty()
    with st.sidebar:
        st.subheader("Filter Plates by Date")
        selected_date = st.date_input("Select Date", value=datetime.today())
        filtered_plates = filter_plates_by_date(selected_date)
        if filtered_plates:
            st.dataframe(filtered_plates, use_container_width=True, hide_index=True)
        else:
            st.write("No plates found for selected date.")
        live_plates_container = st.empty()
        count1_container = st.empty()
        count2_container = st.empty()

    track_data = {}
    final_plate_numbers = []
    finalized_ids = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.success("Finished processing the uploaded video.")
            break

        st.session_state.frame_counter += 1
        frame_combined = frame.copy()

        if st.session_state.frame_counter % FRAME_SKIP == 0:
            st.session_state.processed_frames += 1

            vehicle_results = vehicle_model.track(frame_combined, persist=True, conf=0.40)
            st.session_state.last_vehicle_detections = draw_vehicle_detections(
                frame_combined, vehicle_results, st.session_state.counted_ids_line1, st.session_state.counted_ids_line2
            )

            plate_results = plate_model.track(frame_combined, persist=True, conf=0.40)
            for res in plate_results:
                draw_plate_detections(frame_combined, res.boxes, ocr, track_data, frame_idx)
            cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)

        else:
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

        cv2.line(frame_combined, LINE1_PT1, LINE1_PT2, (0, 0, 255), 3)
        #cv2.line(frame_combined, LINE2_PT1, LINE2_PT2, (255, 255, 0), 3)
        cv2.putText(frame_combined, "Vehicle Count", (LINE1_PT1[0], LINE1_PT1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_combined, "EXIT", (LINE2_PT1[0], LINE2_PT1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame_combined, cv2.COLOR_BGR2RGB)
        stream_placeholder.image(frame_rgb, caption="Vehicle and License Plate Detection Stream", use_container_width=True)

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

        count1_container.markdown(f"**Vehicle Count:** {st.session_state.vehicle_count_line1}")
        #count2_container.markdown(f"**Exit Count:** {st.session_state.vehicle_count_line2}")

        frame_idx += 1

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    main()




#Combined2/Count store in the file
# Vehicle Counting and Plate Detection from Uploaded Video
#Count Save into the txt file
'''import cv2
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
FRAME_SKIP = 4
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
LINE1_PT1 = (100, 400)
LINE1_PT2 = (1100, 400)
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
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                time, dir_, count = line.strip().split(',')
                if dir_ == "Enter":
                    enter_count = int(count)
                elif dir_ == "Exit":
                    exit_count = int(count)
    except (FileNotFoundError, ValueError):
        enter_count = 0
        exit_count = 0

    if direction == "Enter":
        enter_count += increment
    elif direction == "Exit":
        exit_count += increment

    with open(COUNT_FILE, "w") as f:
        f.write(f"{current_time},Enter,{enter_count}\n")
        f.write(f"{current_time},Exit,{exit_count}\n")

def load_vehicle_counts():
    try:
        with open(COUNT_FILE, "r") as f:
            lines = f.readlines()
        enter_count = 0
        exit_count = 0
        for line in lines:
            if line.strip():
                time, dir_, count = line.strip().split(',')
                if dir_ == "Enter":
                    enter_count = int(count)
                elif dir_ == "Exit":
                    exit_count = int(count)
        return enter_count, exit_count
    except (FileNotFoundError, ValueError):
        return 0, 0




def main():
    st.title("Vehicle Counting and License Plate Detection")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        st.warning("Please upload a video file to start processing.")
        return

    temp_video_path = "uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load models and OCR
    plate_model, vehicle_model = load_models()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Video capture setup
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

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
    if "last_vehicle_detections" not in st.session_state:
        st.session_state.last_vehicle_detections = []

    # UI placeholders
    vehicle_placeholder = st.empty()
    plate_placeholder = st.empty()

    with st.sidebar:
        st.subheader("Filter Plates by Date")
        selected_date = st.date_input("Select Date", value=datetime.today())
        filtered_plates = filter_plates_by_date(selected_date)
        if filtered_plates:
            st.dataframe(filtered_plates, use_container_width=True, hide_index=True)
        else:
            st.write("No plates found for selected date.")

        live_plates_container = st.empty()
        count1_container = st.empty()
        count2_container = st.empty()

    # Initialize tracking data
    track_data = {}
    final_plate_numbers = []
    finalized_ids = set()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_vehicle = frame.copy()
            frame_plate = frame.copy()

            if frame_idx % FRAME_SKIP == 0:
                # Vehicle detection and tracking
                vehicle_results = vehicle_model.track(frame_vehicle, persist=True, conf=0.40)
                st.session_state.last_vehicle_detections = draw_vehicle_detections(
                    frame_vehicle, vehicle_results, st.session_state.counted_ids_line1, st.session_state.counted_ids_line2
                )

                # Plate detection and tracking
                plate_results = plate_model.track(frame_plate, persist=True, conf=0.40)
                for res in plate_results:
                    draw_plate_detections(frame_plate, res.boxes, ocr, track_data, frame_idx)
                cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)
            else:
                for detection in st.session_state.last_vehicle_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    track_id = detection['track_id']
                    class_name = detection['class_name']
                    conf = detection['conf']
                    cx, cy = detection['centroid']
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.rectangle(frame_vehicle, (x1, y1), (x2, y2), (0,255, 0), 2)
                    cv2.putText(frame_vehicle, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame_vehicle, (cx, cy), 5, (255, 0, 0), -1)

            # Draw counting lines
            cv2.line(frame_vehicle, LINE1_PT1, LINE1_PT2, (0, 0, 255), 3)
            cv2.line(frame_vehicle, LINE2_PT1, LINE2_PT2, (255, 255, 0), 3)
            cv2.putText(frame_vehicle, "ENTER", (LINE1_PT1[0], LINE1_PT1[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_vehicle, "EXIT", (LINE2_PT1[0], LINE2_PT1[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Convert to RGB
            frame_vehicle_rgb = cv2.cvtColor(frame_vehicle, cv2.COLOR_BGR2RGB)
            frame_plate_rgb = cv2.cvtColor(frame_plate, cv2.COLOR_BGR2RGB)

            # Display on screen
            vehicle_placeholder.image(frame_vehicle_rgb, caption="Vehicle Counting Stream", use_container_width=True)
            plate_placeholder.image(frame_plate_rgb, caption="License Plate Detection Stream", use_container_width=True)

            # Update live plate sidebar
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

            count1_container.markdown(f"**Enter Count:** {st.session_state.vehicle_count_line1}")
            count2_container.markdown(f"**Exit Count:** {st.session_state.vehicle_count_line2}")

            frame_idx += 1

        cap.release()

    except Exception as e:
        st.error(f"Stream error: {e}")



if __name__ == "__main__":
    main()'''
















# Without aspect ratio panelty
'''import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR

def load_model(model_path="license_plate_detector.pt"):
    return YOLO(model_path)

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

def draw_detections(frame, results, ocr, track_data, frame_idx):
    for box in results:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
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
        cv2.putText(frame, number_to_show, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def cleanup_tracks(track_data, finalized_list, finalized_ids, frame_idx, max_age=50):
    to_remove = []

    for tid, data in track_data.items():
        # Check if this ID has been inactive (not seen in frame) for > max_age frames
        if frame_idx - data.get("last_seen", 0) > max_age:
            if tid not in finalized_ids:
                number = data.get("number", "")
                if (number and number != "Unknown") and (number not in finalized_list):
                    finalized_list.append(number)
                    with open("final_plate_numbers.txt", "a") as f:
                        f.write(number + "\n")
                finalized_ids.add(tid)
            to_remove.append(tid)

    for tid in to_remove:
        del track_data[tid]


def main():
    st.title("License Plate Detection from Uploaded Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if not uploaded_file:
        st.warning("Please upload a video to start processing.")
        return

    # Save the uploaded video to disk (optional, allows OpenCV reading)
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Failed to read video. Please upload a valid video file.")
        return

    model = load_model()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    finalized_ids = set()
    track_data = {}
    final_plate_numbers = []
    frame_idx = 0
    frame_skip = 1

    frame_placeholder = st.empty()
    live_placeholder = st.sidebar.empty()
    finalized_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model.track(frame, persist=True)
            for res in results:
                draw_detections(frame, res.boxes, ocr, track_data, frame_idx)

            cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            
   

            with live_placeholder.container():
                st.subheader("Live Plates")
                if track_data:
                    plates = [{"ID": tid, "Plate Number": data["number"]} for tid, data in track_data.items()]
                    st.dataframe(plates, use_container_width=True, hide_index=True)
                else:
                    st.text("detecting...")






            with finalized_placeholder.container():
                st.subheader("Finalized Plates")
                if final_plate_numbers:
                    for plate in final_plate_numbers:
                        st.write(plate)
                else:
                    st.write("No finalized plates yet.")

        frame_idx += 1

    cap.release()
    st.success("Finished processing video.")

if __name__ == "__main__":
    main()
'''





