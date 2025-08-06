import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def extract_skin(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

def get_average_skin_tone(skin_image):
    mask = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    pixels = skin_image[mask > 0]
    if len(pixels) == 0:
        return (0, 0, 0)
    avg_color = np.mean(pixels, axis=0)
    return tuple(map(int, avg_color))

def report_image_to_bytes(image):
    image = Image.fromarray(image)
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Function to convert RGB to HEX
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# Streamlit

st.set_page_config(page_title="Skin Tone Analysis", layout="centered")
st.title(" Skin Tone Analysis")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)

    faces = detect_face(image_bgr)

    if len(faces) == 0:
        st.warning("No face detected. Try another image.")
    else:
        # Process the first detected face
        x, y, w, h = faces[0]
        face_region = image_bgr[y:y+h, x:x+w]
        skin_region = extract_skin(face_region)
        avg_bgr = get_average_skin_tone(skin_region)
        avg_rgb = tuple(reversed(avg_bgr))  # Convert BGR to RGB
        hex_color = rgb_to_hex(avg_rgb)

        # Draw rectangle and show RGB on image
        image_with_box = image_bgr.copy()
        cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_with_box, f"RGB: {avg_rgb}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Convert BGR to RGB for Streamlit display
        image_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="Detected Skin Tone")

        st.markdown(f"### Detected Skin Tone (RGB): `{avg_rgb}`")
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <div style="width: 100px; height: 100px; background-color: {hex_color}; border-radius: 10px; border: 2px solid #000;"></div>
                <div style="margin-left: 20px; font-size: 18px;">
                    <b>RGB:</b> {avg_rgb}<br>
                    <b>HEX:</b> {hex_color}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write ("  ")

        # Generate download button
        st.download_button(
            label="Download",
            data=report_image_to_bytes(image_rgb),
            file_name="skin_tone_report.jpg",
            mime="image/jpeg"
        )

