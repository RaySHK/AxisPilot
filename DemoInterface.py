import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import threading
import tkinter as tk
from PIL import Image, ImageTk
import time
import numpy as np

# Initialize camera and Mediapipe
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 640)  # Set width
camera_video.set(4, 480)  # Set height

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

keyboard = Controller()

# Shared flags
recalibrate_flag = False
paused = False

# Variables to store initial (null) position
initial_tilt = None
initial_vertical = None

# Thresholds
THRESHOLD_LEFT = 10
THRESHOLD_RIGHT = -10
VERTICAL_THRESHOLD_UP = -0.05
VERTICAL_THRESHOLD_DOWN = 0.1

current_keys = []  # List to track currently pressed keys
frame_to_display = None  # Shared variable for the frame


def recalibrate():
    global recalibrate_flag
    recalibrate_flag = True


def toggle_pause():
    global paused
    paused = not paused


def process_frame(frame):
    global initial_tilt, initial_vertical, recalibrate_flag, paused, current_keys

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        height, width = frame.shape[:2]

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        left_eye = [landmarks[33].x * width, landmarks[33].y * height]
        right_eye = [landmarks[263].x * width, landmarks[263].y * height]
        nose = [landmarks[1].x * width, landmarks[1].y * height]
        chin = [landmarks[152].x * width, landmarks[152].y * height]

        for point in [33, 263, 1, 152]:
            x = int(landmarks[point].x * width)
            y = int(landmarks[point].y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        eye_line = [right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]]
        tilt_angle = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
        face_height = np.linalg.norm([chin[0] - (left_eye[0] + right_eye[0]) / 2, chin[1] - (left_eye[1] + right_eye[1]) / 2])
        vertical_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / face_height

        if recalibrate_flag:
            initial_tilt = tilt_angle
            initial_vertical = vertical_ratio
            recalibrate_flag = False

        if initial_tilt is None or initial_vertical is None:
            initial_tilt = tilt_angle
            initial_vertical = vertical_ratio

        adjusted_tilt = tilt_angle - initial_tilt
        adjusted_vertical = vertical_ratio - initial_vertical

        if adjusted_tilt > THRESHOLD_LEFT and adjusted_vertical > VERTICAL_THRESHOLD_DOWN:
            press_and_hold_key(['w', 'a'])
        elif adjusted_tilt > THRESHOLD_LEFT and adjusted_vertical < VERTICAL_THRESHOLD_UP:
            press_and_hold_key(['s', 'a'])
        elif adjusted_tilt < THRESHOLD_RIGHT and adjusted_vertical > VERTICAL_THRESHOLD_DOWN:
            press_and_hold_key(['w', 'd'])
        elif adjusted_tilt < THRESHOLD_RIGHT and adjusted_vertical < VERTICAL_THRESHOLD_UP:
            press_and_hold_key(['s', 'd'])
        elif adjusted_tilt > THRESHOLD_LEFT:
            press_and_hold_key(['a'])
        elif adjusted_tilt < THRESHOLD_RIGHT:
            press_and_hold_key(['d'])
        elif adjusted_vertical > VERTICAL_THRESHOLD_DOWN:
            press_and_hold_key(['w'])
        elif adjusted_vertical < VERTICAL_THRESHOLD_UP:
            press_and_hold_key(['s'])
        else:
            release_all_keys()

    if current_keys:
        keys_display = " & ".join(current_keys)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Keys Pressed: {keys_display}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return frame


def press_and_hold_key(new_keys):
    global current_keys
    if set(current_keys) != set(new_keys):
        release_all_keys()
        for key in new_keys:
            keyboard.press(key)
        current_keys = new_keys


def release_all_keys():
    global current_keys
    for key in current_keys:
        keyboard.release(key)
    current_keys = []


def capture_frames():
    global frame_to_display
    while True:
        ok, frame = camera_video.read()
        if ok:
            frame_to_display = process_frame(frame)
        time.sleep(0.03)


def update_camera_feed(label):
    global frame_to_display
    while True:
        if paused or frame_to_display is None:
            time.sleep(0.1)
            continue

        frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        time.sleep(0.03)


def run_gui():
    root = tk.Tk()
    root.title("AxisPilot Interface")

    tk.Button(root, text="Recalibrate", command=recalibrate).pack(pady=10)
    tk.Button(root, text="Pause/Resume", command=toggle_pause).pack(pady=10)

    status_label = tk.Label(root, text="Running", font=("Helvetica", 14))
    status_label.pack(pady=10)

    def update_status():
        status_label.config(text="Paused" if paused else "Running")
        root.after(500, update_status)

    update_status()

    video_label = tk.Label(root)
    video_label.pack(padx=10, pady=10)

    camera_thread = threading.Thread(target=update_camera_feed, args=(video_label,), daemon=True)
    camera_thread.start()

    root.mainloop()


def show_landing_page():
    def start_application():
        landing_root.destroy()
        run_gui()

    landing_root = tk.Tk()
    landing_root.title("Welcome to AxisPilot")

    # Load background image
    bg_image = Image.open("background.png")  # Ensure background.png is in the same directory
    img_width, img_height = bg_image.size
    landing_root.geometry("480x640")  # Set the window width to 640 and adjust height for space

    # Resize the image to fit the landing page width (640px)
    resized_width = 640
    resized_height = int(img_height * (resized_width / img_width))  # Maintain aspect ratio
    bg_image = bg_image.resize((512, 512), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Set the background image
    background_label = tk.Label(landing_root, image=bg_photo)
    background_label.place(relwidth=1, relheight=1, y=0)  # Place image at the top

    # Create a frame to position the image and the button
    # frame = tk.Frame(landing_root, bg='white', bd=5)
    # frame.place(relx=0.5, rely=0.4, relwidth=0.8, relheight=0.6, anchor="center")

    # Add the Start button outside the image area (below the image)
    start_button = tk.Button(landing_root, text="Start", command=start_application, font=("Helvetica", 14))
    start_button.place(relx=0.5, rely=0.95, anchor="center")

    landing_root.mainloop()


frame_capture_thread = threading.Thread(target=capture_frames, daemon=True)
frame_capture_thread.start()

if __name__ == "__main__":
    show_landing_page()



#sswwdsss