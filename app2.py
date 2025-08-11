import cv2
import numpy as np
import streamlit as st

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_skin(image):
    """Extract skin pixels using HSV range."""
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower, upper)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin, mask

def get_average_skin_color(skin_masked):
    """Get dominant skin color using k-means, safe from empty pixels."""
    pixels = skin_masked.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # remove black pixels

    if len(pixels) < 2:
        return None  # Not enough pixels

    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    return tuple(int(c) for c in dominant_color)

def draw_skin_tone_scale(frame, avg_color):
    """Draw color box + skin tone scale on the frame."""
    h, w, _ = frame.shape
    scale_width = 30

    # Draw vertical grayscale skin tone scale
    for i in range(h):
        color_value = int(255 - (i / h) * 255)
        cv2.line(frame, (w - scale_width, i), (w, i), (color_value, color_value, color_value), 1)

    # Draw detected skin color box
    if avg_color:
        cv2.rectangle(frame, (10, 10), (110, 60), avg_color, -1)
        cv2.putText(frame, f"RGB: {avg_color}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Streamlit UI
st.title("💡 Real-Time Skin Tone Detector")
st.write("Uses your webcam to detect your skin tone and show a color box with RGB values.")

run = st.checkbox("Start Camera")

frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        avg_color = None
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            skin, mask = extract_skin(face_roi)
            avg_color = get_average_skin_color(skin)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        draw_skin_tone_scale(frame, avg_color)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
