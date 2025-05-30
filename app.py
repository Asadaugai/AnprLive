
# With aspect ratio panelty
import cv2
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
    ar = (x2 - x1) / (y2 - y1 + 1e-6)
    aspect_score = -abs(ar - 3.0) * 50
    return sharpness + area * 0.01 + aspect_score

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

def cleanup_tracks(track_data, finalized_list, finalized_ids, frame_idx, max_age=10):
    to_remove = []
    for tid, data in track_data.items():
        if frame_idx - data.get("last_seen", 0) > max_age:
            if tid not in finalized_ids:
                number = data.get("number", "")
                if (number and number != "Unknown") and (number not in finalized_list):
                    finalized_list.append(number)
                    with open("final_plate_numbers.txt", "a") as f:
                        f.write(number + "\n")
                finalized_ids.add(tid)
                if len(finalized_ids) > 25:
                    finalized_ids.clear()
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

            
            '''with live_placeholder.container():
                st.subheader("Live Plates")
                if track_data:
                    for tid, data in track_data.items():
                        st.write(f"ID {tid}: {data['number']}")
                else:
                    st.write("Detecting...")'''

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






