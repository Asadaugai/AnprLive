# Without Aspect ratio and OCR runs only when better vew is found
# Updated
import cv2
import os
import subprocess
import numpy as np
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
from paddleocr import PaddleOCR
import threading
import queue

def load_model(model_path="license_plate_detector.pt"):
    return YOLO(model_path)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def align_plate(plate_img):
    return cv2.resize(plate_img, (240, 80))



def score_plate_view_preliminary(plate_img, bbox):
    sharpness = calculate_sharpness(plate_img)
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return sharpness + (area * 0.01)  # Preliminary score without OCR confidence

def score_plate_view(plate_img, bbox, ocr_confidence):
    sharpness = calculate_sharpness(plate_img)
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return sharpness + (area * 0.01) + (ocr_confidence * 100)  # Full score for storage

def read_plate_text_with_confidence(image, ocr):
    image = align_plate(image)
    result = ocr.ocr(image, cls=False)
    if result and result[0]:
        lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
        confidence = max([line[1][1] for line in result[0]]) if result[0] else 0
        return " ".join(lines) if lines else "Unknown", confidence
    return "Unknown", 0

def draw_detections(frame, results, ocr, track_data, frame_idx):
    for box in results:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
        if track_id == -1:
            continue

        plate_img = frame[y1:y2, x1:x2]
        # Calculate preliminary score to check if view is better
        preliminary_score = score_plate_view_preliminary(plate_img, (x1, y1, x2, y2))
        current_best = track_data.get(track_id, {})

        # Only run OCR if this is a new track or a better view
        if not current_best or preliminary_score > current_best.get("preliminary_score", 0):
            # Run OCR to get number and confidence
            number, ocr_confidence = read_plate_text_with_confidence(plate_img, ocr)
            # Only update if OCR confidence is higher (or no previous confidence)
            if number != "Unknown" and (not current_best or ocr_confidence > current_best.get("ocr_confidence", 0)):
                # Calculate full score for storage
                full_score = score_plate_view(plate_img, (x1, y1, x2, y2), ocr_confidence)
                track_data[track_id] = {
                    "img": plate_img,
                    "score": full_score,  # Store full score for reference
                    "preliminary_score": preliminary_score,  # Store preliminary score for future comparisons
                    "number": number,
                    "ocr_confidence": ocr_confidence,
                    "last_seen": frame_idx
                }

        # Draw annotations
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

def start_rtsp_stream(rtsp_url, width=1920, height=1080):
    cmd = [
        r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe',
        '-rtsp_transport', 'tcp',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', rtsp_url,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ]
    frame_q = queue.Queue(maxsize=1000)
    def reader():
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        while True:
            raw_image = pipe.stdout.read(width * height * 3)
            if not raw_image:
                break
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3)).copy()
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass
        pipe.terminate()
    threading.Thread(target=reader, daemon=True).start()
    return frame_q

def main():
    st.title("License Plate Detection")
    rtsp_url = os.getenv("RTSP_URL")
    width, height = 1920, 1080
    finalized_ids = set()  

    frame_q = start_rtsp_stream(rtsp_url, width, height)

    model = load_model()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    track_data = {}
    final_plate_numbers = []
    frame_idx = 0

    frame_skip = 0

    # Placeholders
    frame_placeholder = st.empty()
    live_placeholder = st.sidebar.empty()
    finalized_placeholder = st.empty()

    while True:
        try:
            frame = frame_q.get(timeout=1)

            if (frame_skip == 0) or (frame_idx % frame_skip == 0):
                results = model.track(frame, persist=True)
                for res in results:
                    draw_detections(frame, res.boxes, ocr, track_data, frame_idx)
                #cleanup_tracks(track_data, final_plate_numbers, frame_idx)
                cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)


                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Live plate display in sidebar
                with live_placeholder.container():
                    st.subheader("Live Plates")
                    if track_data:
                        plates = [{"ID": tid, "Plate Number": data["number"]} for tid, data in track_data.items()]
                        st.dataframe(plates, use_container_width=True, hide_index=True)
                    else:
                        st.write("Detecting...")

                # Finalized plate list below the video
                with finalized_placeholder.container():
                    st.subheader("Finalized Plates")
                    if final_plate_numbers:
                        for plate in final_plate_numbers:
                            st.write(plate)
                    else:
                        st.write("finalized plates...")

            frame_idx += 1

        except queue.Empty:
            continue


if __name__ == "__main__":
    main()








# Without Aspect ratio and OCR runs in every frame
# Updated
'''import cv2
import os
import subprocess
import numpy as np
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
from paddleocr import PaddleOCR
import threading
import queue

def load_model(model_path="license_plate_detector.pt"):
    return YOLO(model_path)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def align_plate(plate_img):
    return cv2.resize(plate_img, (240, 80))



def score_plate_view(plate_img, bbox, ocr_confidence):
    sharpness = calculate_sharpness(plate_img)
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return sharpness + (area * 0.01) + (ocr_confidence * 100)  

def read_plate_text_with_confidence(image, ocr):
    image = align_plate(image)
    result = ocr.ocr(image, cls=False)
    if result and result[0]:
        lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
        confidence = max([line[1][1] for line in result[0]]) if result[0] else 0
        return " ".join(lines) if lines else "Unknown", confidence
    return "Unknown", 0

def draw_detections(frame, results, ocr, track_data, frame_idx):
    for box in results:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
        if track_id == -1:
            continue

        plate_img = frame[y1:y2, x1:x2]
        # Run OCR once to get both number and confidence
        number, ocr_confidence = read_plate_text_with_confidence(plate_img, ocr)
        # Calculate score including OCR confidence
        score = score_plate_view(plate_img, (x1, y1, x2, y2), ocr_confidence)
        current_best = track_data.get(track_id, {})

        # Update if no previous data or better score
        if number != "Unknown" and (not current_best or score > current_best.get("score", 0)):
            track_data[track_id] = {
                "img": plate_img,
                "score": score,
                "number": number,
                "ocr_confidence": ocr_confidence,
                "last_seen": frame_idx
            }

        # Draw annotations
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

def start_rtsp_stream(rtsp_url, width=1920, height=1080):
    cmd = [
        r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe',
        '-rtsp_transport', 'tcp',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', rtsp_url,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ]
    frame_q = queue.Queue(maxsize=1000)
    def reader():
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        while True:
            raw_image = pipe.stdout.read(width * height * 3)
            if not raw_image:
                break
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3)).copy()
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass
        pipe.terminate()
    threading.Thread(target=reader, daemon=True).start()
    return frame_q

def main():
    st.title("License Plate Detection")
    rtsp_url = os.getenv("RTSP_URL")
    width, height = 1920, 1080
    finalized_ids = set()  

    frame_q = start_rtsp_stream(rtsp_url, width, height)

    model = load_model()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    track_data = {}
    final_plate_numbers = []
    frame_idx = 0

    frame_skip = 0

    # Placeholders
    frame_placeholder = st.empty()
    live_placeholder = st.sidebar.empty()
    finalized_placeholder = st.empty()

    while True:
        try:
            frame = frame_q.get(timeout=1)

            if (frame_skip == 0) or (frame_idx % frame_skip == 0):
                results = model.track(frame, persist=True)
                for res in results:
                    draw_detections(frame, res.boxes, ocr, track_data, frame_idx)
                #cleanup_tracks(track_data, final_plate_numbers, frame_idx)
                cleanup_tracks(track_data, final_plate_numbers, finalized_ids, frame_idx)


                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Live plate display in sidebar
                with live_placeholder.container():
                    st.subheader("Live Plates")
                    if track_data:
                        plates = [{"ID": tid, "Plate Number": data["number"]} for tid, data in track_data.items()]
                        st.dataframe(plates, use_container_width=True, hide_index=True)
                    else:
                        st.write("Detecting...")

                # Finalized plate list below the video
                with finalized_placeholder.container():
                    st.subheader("Finalized Plates")
                    if final_plate_numbers:
                        for plate in final_plate_numbers:
                            st.write(plate)
                    else:
                        st.write("finalized plates...")

            frame_idx += 1

        except queue.Empty:
            continue


if __name__ == "__main__":
    main()'''













