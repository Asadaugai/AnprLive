

#ffmpeg
import cv2
import subprocess
import numpy as np

# Your RTSP URL
rtsp_url = "rtsp://swatitech:Pakistan%401122@182.176.86.172:554/Streaming/Channels/601"

# Adjust resolution if needed (check your camera settings)
width, height = 1920, 1080

# FFmpeg command
ffmpeg_cmd = [
    r'C:\Asad Mehmood\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe',
    '-rtsp_transport', 'tcp',
    '-i', rtsp_url,
    '-loglevel', 'quiet',
    '-an',
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-'
]

pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

while True:
    raw_image = pipe.stdout.read(width * height * 3)
    if not raw_image:
        print("Failed to read frame.")
        break

    frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()






'''import cv2
import subprocess
import numpy as np
import threading
import queue

# === Configuration ===
rtsp_url = "rtsp://swatitech:Pakistan%401122@182.176.86.172:554/Streaming/Channels/601"
width, height = 1920, 1080
queue_size = 1000  # You can change this based on your system

# === FFmpeg command ===
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

# === Frame Queue ===
frame_queue = queue.Queue(maxsize=queue_size)

# === Frame Reader Thread ===
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
            # Drop frame if queue is full (to stay real-time)
            pass
    pipe.terminate()

# === Start Reader Thread ===
threading.Thread(target=reader_thread, daemon=True).start()

# === Display Loop ===
while True:
    try:
        frame = frame_queue.get(timeout=1)  # Wait for next frame
        cv2.imshow("RTSP Stream", frame)
    except queue.Empty:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()'''
