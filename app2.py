import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Predefined realistic skin tone palette (light → dark)
skin_tone_colors = [
    (242,252,255),
    (224, 239, 255),  # Very Fair
    (189, 224, 255),  # Fair
    (125, 194, 241),  # Medium Light
    (105, 172, 224),  # Medium
    (66, 134, 198),   # Tan
    (36, 85, 141),    # Dark
    (30, 54, 90),     # Very Dark
    (22, 34, 45),      # Deepest Dark
    (3, 18, 33)
]

def extract_skin(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower, upper)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin, mask

def get_average_skin_color(skin_masked):
    pixels = skin_masked.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    if len(pixels) < 2:
        return None
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(int(c) for c in centers[np.argmax(np.bincount(labels.flatten()))])

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def find_closest_skin_tone(color, palette):
    diffs = [np.linalg.norm(np.array(color) - np.array(p)) for p in palette]
    return np.argmin(diffs)

def draw_skin_tone_scale(frame, avg_color):
    h, w, _ = frame.shape
    scale_width = 40
    num_segments = len(skin_tone_colors) - 1

    # Draw gradient scale
    for i in range(h):
        seg = int(i / (h / num_segments))
        t = (i % (h / num_segments)) / (h / num_segments)
        c1, c2 = skin_tone_colors[seg], skin_tone_colors[min(seg + 1, num_segments)]
        color = interpolate_color(c1, c2, t)
        cv2.line(frame, (w - scale_width, i), (w, i), color, 1)

    # Highlight closest skin tone on the scale
    if avg_color:
        closest_index = find_closest_skin_tone(avg_color, skin_tone_colors)
        y_start = int((closest_index / len(skin_tone_colors)) * h)
        y_end = int(((closest_index + 1) / len(skin_tone_colors)) * h)
        cv2.rectangle(frame, (w - scale_width - 5, y_start), (w, y_end), (0, 255, 0), 2)

        # Draw detected color box
        cv2.rectangle(frame, (10, 10), (110, 60), avg_color, -1)
        cv2.putText(frame, f"RGB: {avg_color}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        avg_color = None
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            skin, _ = extract_skin(face_roi)
            avg_color = get_average_skin_color(skin)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        draw_skin_tone_scale(img, avg_color)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("💡 Real-Time Skin Tone Detector with Gradient Scale")
webrtc_streamer(key="skin-tone", video_processor_factory=VideoProcessor)
