import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import numpy as np

# Initialize camera and Mediapipe
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)  # Set width
camera_video.set(4, 960)   # Set height

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

keyboard = Controller()

# Define thresholds for movement
THRESHOLD_LEFT = 10   # Tilt left for 'A'
THRESHOLD_RIGHT = -10  # Tilt right for 'D'
VERTICAL_THRESHOLD_UP = -0.05  # Raise head for 'S'
VERTICAL_THRESHOLD_DOWN = 0.1  # Lower head for 'W'

# Variables to store initial (null) position
initial_tilt = None
initial_vertical = None

# Track currently pressed key
current_key = None

def get_head_position(landmarks, image_shape):
    """Calculate head tilt and normalized vertical movement."""
    height, width = image_shape[:2]
    left_eye = np.array([landmarks[33].x * width, landmarks[33].y * height])
    right_eye = np.array([landmarks[263].x * width, landmarks[263].y * height])
    nose = np.array([landmarks[1].x * width, landmarks[1].y * height])
    chin = np.array([landmarks[152].x * width, landmarks[152].y * height])

    # Calculate tilt angle between eyes
    eye_line = right_eye - left_eye
    tilt_angle = np.degrees(np.arctan2(eye_line[1], eye_line[0]))

    # Calculate vertical movement ratio
    face_height = np.linalg.norm(chin - ((left_eye + right_eye) / 2))
    vertical_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / face_height

    return tilt_angle, vertical_ratio

def press_and_hold_key(new_key):
    """Press and hold a key, releasing the previous one if necessary."""
    global current_key
    if current_key != new_key:
        if current_key:
            keyboard.release(current_key)  # Release the previous key
        keyboard.press(new_key)  # Hold the new key
        current_key = new_key  # Update the current key

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get head position
        tilt_angle, vertical_ratio = get_head_position(landmarks, frame.shape)

        # Set the initial null position if not yet set
        if initial_tilt is None and initial_vertical is None:
            initial_tilt = tilt_angle
            initial_vertical = vertical_ratio

        # Calculate adjusted movement relative to the initial position
        adjusted_tilt = tilt_angle - initial_tilt
        adjusted_vertical = vertical_ratio - initial_vertical

        # Determine which key to press based on head movement
        if adjusted_tilt > THRESHOLD_LEFT:
            press_and_hold_key('a')  # Tilt Left -> 'A'
        elif adjusted_tilt < THRESHOLD_RIGHT:
            press_and_hold_key('d')  # Tilt Right -> 'D'
        elif adjusted_vertical > VERTICAL_THRESHOLD_DOWN:
            press_and_hold_key('w')  # Lower Head -> 'W'
        elif adjusted_vertical < VERTICAL_THRESHOLD_UP:
            press_and_hold_key('s')  # Raise Head -> 'S'
        else:
            if current_key:
                keyboard.release(current_key)  # Release key if in null position
                current_key = None

    # Exit if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
camera_video.release()
cv2.destroyAllWindows()

