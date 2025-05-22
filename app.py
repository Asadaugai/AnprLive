#live_app.py

import cv2
import subprocess
import numpy as np
import streamlit as st
from ultralytics import YOLO
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

def read_plate_text(image, ocr):
    image = align_plate(image)
    result = ocr.ocr(image, cls=False)
    if result and result[0]:
        lines = [line[1][0] for line in result[0]]
        return " ".join(lines)
    return "Unknown"

def score_plate_view(plate_img, bbox):
    sharpness = calculate_sharpness(plate_img)
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    ar = (x2 - x1) / (y2 - y1 + 1e-6)
    aspect_score = -abs(ar - 3.0) * 100
    return sharpness + area * 0.01 + aspect_score

def draw_annotations(frame, boxes, ocr, plate_cache, plate_views, track_last_seen, frame_idx, final_plate_numbers_by_id):
    current_ids = set()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
        if track_id == -1:
            continue
        current_ids.add(track_id)
        track_last_seen[track_id] = frame_idx
        plate_img = frame[y1:y2, x1:x2]
        score = score_plate_view(plate_img, (x1, y1, x2, y2))

        is_best_view = (track_id not in plate_views) or (score > plate_views[track_id]["score"])
        if is_best_view:
            plate_views[track_id] = {"img": plate_img, "score": score}
            plate_number = read_plate_text(plate_img, ocr)
            if plate_number and plate_number != "Unknown":
                plate_cache[track_id] = plate_number
                final_plate_numbers_by_id[track_id] = plate_number

        plate_number = plate_cache.get(track_id, "Detecting...")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return current_ids

def cleanup_stale_tracks(plate_views, track_last_seen, frame_idx, final_plate_numbers_by_id, final_plate_numbers_list, max_missing_frames=50):
    inactive_ids = [tid for tid, last_seen in track_last_seen.items() if frame_idx - last_seen > max_missing_frames]
    for tid in inactive_ids:
        plate_views.pop(tid, None)
        track_last_seen.pop(tid, None)
        final_plate = final_plate_numbers_by_id.pop(tid, None)
        if final_plate and final_plate not in final_plate_numbers_list:
            final_plate_numbers_list.append(final_plate)

def start_rtsp_stream(rtsp_url, width=1920, height=1080, queue_size=1000):
    ffmpeg_cmd = [
        r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe',
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
                pass
        pipe.terminate()

    threading.Thread(target=reader_thread, daemon=True).start()
    return frame_queue

def main():
    st.title("License Plate Detection")
    
    # Sidebar for live plates
    with st.sidebar:
        st.header("Live Detected License Plates")
        live_plate_dict_placeholder = st.empty()

    # RTSP stream configuration
    rtsp_url = "rtsp://swatitech:Pakistan%401122@182.176.86.172:554/Streaming/Channels/601"
    width, height = 1920, 1080

    # Initialize session state variables
    if "plate_cache" not in st.session_state:
        st.session_state.plate_cache = {}
    if "plate_views" not in st.session_state:
        st.session_state.plate_views = {}
    if "track_last_seen" not in st.session_state:
        st.session_state.track_last_seen = {}
    if "final_plate_numbers_by_id" not in st.session_state:
        st.session_state.final_plate_numbers_by_id = {}
    if "final_plate_numbers_list" not in st.session_state:
        st.session_state.final_plate_numbers_list = []
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    model = load_model()
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    frame_queue = start_rtsp_stream(rtsp_url, width, height)

    frame_placeholder = st.empty()

    while True:
        try:
            frame = frame_queue.get(timeout=1)
            # Create a writable copy of the frame
            frame = frame.copy()
        except queue.Empty:
            continue

        results = model.track(frame, persist=True)
        for result in results:
            draw_annotations(frame, result.boxes, ocr,
                             st.session_state.plate_cache,
                             st.session_state.plate_views,
                             st.session_state.track_last_seen,
                             st.session_state.frame_idx,
                             st.session_state.final_plate_numbers_by_id)

        cleanup_stale_tracks(st.session_state.plate_views,
                             st.session_state.track_last_seen,
                             st.session_state.frame_idx,
                             st.session_state.final_plate_numbers_by_id,
                             st.session_state.final_plate_numbers_list)

        st.session_state.frame_idx += 1

        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update sidebar with live plates
        with live_plate_dict_placeholder.container():
            if st.session_state.plate_cache:
                for plate_number in st.session_state.plate_cache.values():
                    st.write(plate_number)
            else:
                st.write("No live plates detected.")


if __name__ == "__main__":
    main()