import streamlit as st
import cv2
import mediapipe as mp
import pyautogui as pa
import math

st.title("Cursor control project")
image = st.image([])

# Initialize Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set up MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pa.size()

# Initialize state variables
drawing = False  # Flag to track drawing state
last_position = None  # Store the last smoothed mouse position
smooth_factor = 0.2  # Smoothing factor for mouse movements
 # Threshold (in pixels) for pinch gesture detection
pinch_threshold =st.sidebar.slider("Select the distace between fingers for click ", min_value=10, max_value=100, step=10,value=40)

st.write(f"You selected: {pinch_threshold}")

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def map_coordinates(x, y, frame_width, frame_height):
    """Map coordinates from the camera frame to screen dimensions."""
    mapped_x = screen_width / frame_width * x
    mapped_y = screen_height / frame_height * y
    return mapped_x, mapped_y


while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB for processing by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw hand landmarks on the frame
            drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Get index finger tip (id=8) and thumb tip (id=4) positions
            index_finger = landmarks[8]
            thumb = landmarks[12]

            # Convert normalized coordinates to pixel values
            index_x, index_y = int(index_finger.x * frame_width), int(index_finger.y * frame_height)
            thumb_x, thumb_y = int(thumb.x * frame_width), int(thumb.y * frame_height)

            # Draw circles for visualization
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)

            # Map index finger position to screen coordinates
            screen_index_x, screen_index_y = map_coordinates(index_x, index_y, frame_width, frame_height)

            # Smooth the mouse movement
            if last_position is None:
                smoothed_x, smoothed_y = screen_index_x, screen_index_y
            else:
                smoothed_x = smooth_factor * screen_index_x + (1 - smooth_factor) * last_position[0]
                smoothed_y = smooth_factor * screen_index_y + (1 - smooth_factor) * last_position[1]
            last_position = (smoothed_x, smoothed_y)

            # Move the mouse cursor
            pa.moveTo(smoothed_x, smoothed_y)

            # Calculate distance between thumb and index finger
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
            cv2.putText(frame, f"Dist: {int(distance)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Check for pinch gesture to start/stop drawing
            if distance < pinch_threshold:
                cv2.putText(frame, "Drawing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not drawing:
                    pa.mouseDown()  # Begin drawing
                    drawing = True
            else:
                if drawing:
                    pa.mouseUp()  # End drawing
                    drawing = False
                    last_position = None  # Reset the drawing path

    else:
        # If no hand is detected, ensure the mouse button is released
        if drawing:
            pa.mouseUp()
            drawing = False
            last_position = None

    # Display the frame
    image.image(frame,channels = "BGR")

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

