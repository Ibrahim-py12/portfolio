import os
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from google import genai
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
TF_ENABLE_ONEDNN_OPTS=0
# Set up Streamlit layout
st.set_page_config(layout='wide')
col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.empty()  # Not directly used, kept for compatibility

with col2:
    q1 = st.checkbox("Answer the Math problem", value=True)
    q2 = st.checkbox("Estimate what I have made")
    output_text_area = st.subheader("")

# Global default drawing colour
draw_colour = (255, 0, 255)

# Load header images from "Header" folder
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
header = myimg[0] if myimg else np.zeros((80, 640, 3), dtype=np.uint8)

# Initialize Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.5, minTrackCon=0.5)

# Global canvas to hold the drawing (will be updated by the transformer)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.prev_pos = None
        self.drawing_mode = False
        self.draw_colour = (255, 0, 255)

    def transform(self, frame):
        global header, canvas, myimg  # Declare globals for updating header and canvas
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)  # Flip horizontally

        # Apply the current header at the top of the frame
        image[0:80, 0:640] = header

        # Detect hand and get landmarks
        hands, image = detector.findHands(image, draw=False, flipType=True)
        if hands:
            lmList1 = hands[0]["lmList"]
            fingers = detector.fingersUp(hands[0])

            # Selection Mode (Two fingers up)
            if fingers == [0, 1, 1, 0, 0]:
                self.drawing_mode = False
                self.prev_pos = None
                current_pos = lmList1[8][0:2]
                cr2 = lmList1[12][0:2]
                differences = [a + b for a, b in zip(current_pos, cr2)]
                if 220 < differences[0] < 280 and differences[1] < 250:
                    self.draw_colour = (255, 0, 255)  # Pink
                    header = myimg[0]
                elif 500 < differences[0] < 580 and differences[1] < 250:
                    self.draw_colour = (255, 0, 0)  # Blue
                    if len(myimg) > 1:
                        header = myimg[1]
                elif 760 < differences[0] < 810 and differences[1] < 250:
                    self.draw_colour = (255, 255, 0)  # Yellow
                    if len(myimg) > 2:
                        header = myimg[2]
                elif 1000 < differences[0] < 1110 and differences[1] < 250:
                    self.draw_colour = (0, 0, 0)    # Eraser
                    if len(myimg) > 3:
                        header = myimg[3]

            # Drawing Mode (One finger up)
            elif fingers == [0, 1, 0, 0, 0]:
                if not self.drawing_mode:
                    self.prev_pos = None
                    self.drawing_mode = True
                current_pos = tuple(map(int, lmList1[8][0:2]))
                if self.prev_pos is not None:
                    prev_pos_int = tuple(map(int, self.prev_pos))
                    if self.draw_colour == (0, 0, 0):  # Eraser mode: draw circle and line for erasing
                        cv2.circle(self.canvas, current_pos, 30, (0, 0, 0), -1)
                        cv2.line(self.canvas, prev_pos_int, current_pos, (0, 0, 0), 30)
                    else:
                        cv2.line(self.canvas, prev_pos_int, current_pos, self.draw_colour, 10)
                self.prev_pos = current_pos

            # Clear Canvas (All fingers up)
            elif fingers == [0, 1, 1, 1, 0]:
                self.canvas.fill(0)

        # Combine the video frame with the drawing canvas
        img_combined = cv2.addWeighted(image, 0.75, self.canvas, 0.25, 0)
        # Update global canvas for AI processing
        canvas = self.canvas.copy()
        return av.VideoFrame.from_ndarray(img_combined, format="bgr24")

# Start the video streamer using the transformer
webrtc_streamer(key="example",  video_processor_factory=VideoTransformer)

# Initialize the generative AI client
client = genai.Client(api_key="AIzaSyBM0juGs4Z8UtQ-If_0uA780_DDyerR03M")

def send_to_AI(canvas_img):
    """Send image to AI to solve the math problem."""
    pil_img = Image.fromarray(canvas_img)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["Solve the Math problem?", pil_img]
    )
    return response.text

def send_to_AI2(canvas_img):
    """Send image to AI to estimate the drawing."""
    pil_img = Image.fromarray(canvas_img)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["Estimate what I have made", pil_img]
    )
    return response.text

# Process AI request when the button is clicked.
if st.button("Process AI Request"):
    outputs = []
    if q1:
        outputs.append(send_to_AI(canvas))
    if q2:
        outputs.append(send_to_AI2(canvas))
    text = "\n".join(outputs)
    if not outputs:
        st.write("Please select a question.")
    else:
        output_text_area.text(text)

# streamlit run mathgesai.py


'''
git add .
git commit -m "Fixed image upload issue in math gesture AI"
git push origin main


'''