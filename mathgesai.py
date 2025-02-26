import os
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from google import genai
from streamlit_webrtc import webrtc_streamer
import av

# Initialize canvas **once** (fixes the erase issue)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
# Initialize Streamlit layout
st.set_page_config(layout='wide')
col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with col2:
    q1 = st.checkbox("Answer the Math problem", value=True)
    q2 = st.checkbox("Estimate what i have made")
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

draw_colour = (255, 0, 255)

# Load images from "Header" folder
folder_path = "Header"
lis = os.listdir(folder_path)
myimg = []

for pt in lis:
    img = cv2.imread(f"{folder_path}/{pt}")
    if img is not None and img.size > 0:
        img = cv2.resize(img, (640, 80))  # Resize to fit header
        myimg.append(img)
    else:
        print(f"Image '{pt}' not found or invalid.")

# Ensure header is set
header = myimg[0] if myimg else None

# Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.5, minTrackCon=0.5)

# OpenCV Video Capture


prev_pos = None
text = ""
last_selection_time = 0  # To track the last selection time
drawing_mode = False  # Track if we're in drawing mode


def gethandinfo(img):
    """ Detect hand and return landmarks & finger status """
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]  # First detected hand
        lmList1 = hand1["lmList"]  # Landmark points
        fingers1 = detector.fingersUp(hand1)
        return lmList1, fingers1
    return None


def draw(info, prev_pos, canvas, header):
    """ Handle drawing & selection logic """
    global draw_colour, last_selection_time, drawing_mode
    lmList1, fingers = info
    current_pos = None

    # Selection Mode (Two fingers up)
    if fingers == [0, 1, 1, 0, 0]:
        drawing_mode = False  # Reset drawing mode
        prev_pos = None  # Reset prev_pos to prevent unwanted lines
        current_pos = lmList1[8][0:2]
        cr2 = lmList1[12][0:2]
        differences = [a + b for a, b in zip(current_pos, cr2)]

        if 220 < differences[0] < 280 and differences[1] < 250:
            print("Pink selected")
            draw_colour = (255, 0, 255)
            header = myimg[0]
            last_selection_time = time.time()

        elif 500 < differences[0] < 580 and differences[1] < 250:
            print("Blue selected")
            header = myimg[1]
            draw_colour = (255, 0, 0)
            last_selection_time = time.time()

        elif 760 < differences[0] < 810 and differences[1] < 250:
            print("Yellow selected")
            header = myimg[2]
            draw_colour = (255, 255, 0)
            last_selection_time = time.time()

        elif 1000 < differences[0] < 1110 and differences[1] < 250:
            print("Eraser selected")
            draw_colour = (0, 0, 0)
            header = myimg[3]
            last_selection_time = time.time()

    # Drawing Mode (One finger up)
    elif fingers == [0, 1, 0, 0, 0]:
        if not drawing_mode:
            prev_pos = None  # Reset prev_pos when entering drawing mode
            drawing_mode = True  # Activate drawing mode

        current_pos = tuple(map(int, lmList1[8][0:2]))

        if prev_pos is not None:
            prev_pos_int = tuple(map(int, prev_pos))
            if draw_colour == (0, 0, 0):  # Eraser mode
                cv2.circle(canvas, current_pos, 80, (0, 0, 0), -1)  # Filled circle for erasing
                cv2.line(canvas, prev_pos_int, current_pos, (0, 0, 0), 80)

            else:
                cv2.line(canvas, prev_pos_int, current_pos, draw_colour, 10)

        prev_pos = current_pos

        # Clear Canvas (All fingers up)
    elif fingers == [0, 1, 1, 1, 0]:
        print("Clearing Canvas...")
        canvas.fill(0)

    return prev_pos, canvas, header


client = genai.Client(api_key="AIzaSyBM0juGs4Z8UtQ-If_0uA780_DDyerR03M")


def send_to_AI(canvas, fingers):
    """ Send image to AI for solving if needed """
    if fingers == [1, 1, 1, 1, 0]:  # Thumb up means send to AI
        pil_img = Image.fromarray(canvas)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=["Solve the Math problem ? ", pil_img]
        )
        return response.text


def send_to_AI2(canvas, fingers):
    """ Send image to AI for solving if needed """
    if fingers == [1, 1, 1, 1, 0]:  # Thumb up means send to AI
        pil_img = Image.fromarray(canvas)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=["Estimate what have i made  ", pil_img]
        )
        return response.text


# Run the application
if run:
    def video_frame_callback(frame):
        global canvas
        global header
        global prev_pos
        img = frame.to_ndarray(format="bgr24")

        if header is not None:
            img[0:80, 0:640] = header

        info = gethandinfo(img)

        if info:
            prev_pos, canvas, header = draw(info, prev_pos, canvas, header)

        img_combined = cv2.addWeighted(img, 0.75, canvas, 0.25, 0)

        return av.VideoFrame.from_ndarray(img_combined, format="bgr24")




# Run the WebRTC Streamer
if run:
    webrtc_streamer(key="video", video_frame_callback=video_frame_callback)

# streamlit run mathgesai.py