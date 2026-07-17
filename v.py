import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import urllib.request
import os
import time
import math

# ---------------------------------------------------------
# 1. THE 1 EURO FILTER ALGORITHM (Professional VR Smoothing)
# ---------------------------------------------------------
class OneEuroFilter:
    def __init__(self, mincutoff=0.5, beta=0.005, dcutoff=1.0):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        
        # Tuning parameters
        self.mincutoff = mincutoff # Lower = smoother at low speeds
        self.beta = beta           # Higher = less lag at high speeds
        self.dcutoff = dcutoff     # Velocity filter

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        t_e = t - self.t_prev
        if t_e < 1e-5: t_e = 1e-5
        
        # 1. Compute and smooth velocity
        dx = (x - self.x_prev) / t_e
        alpha_d = self.smoothing_factor(t_e, self.dcutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        
        # 2. Compute adaptive cutoff frequency based on velocity
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        
        # 3. Filter the actual coordinates
        alpha = self.smoothing_factor(t_e, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        
        # 4. Save states for next frame
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

# Initialize our X and Y filters
filter_x = OneEuroFilter(mincutoff=0.8, beta=0.007)
filter_y = OneEuroFilter(mincutoff=0.8, beta=0.007)

# ---------------------------------------------------------
# 2. SETUP (AI Model & Camera)
# ---------------------------------------------------------
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading Model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE, 
    num_hands=1, min_hand_detection_confidence=0.7, min_tracking_confidence=0.7
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 60)
screen_width, screen_height = pyautogui.size()

FRAME_PAD = 60
is_dragging = False
right_clicked = False
prev_filtered_x, prev_filtered_y = 0, 0

print("Starting 1-Euro Virtual Mouse... Press 'q' to exit.")

# ---------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    # Current timestamp in seconds for the 1 Euro Filter
    current_time = time.time()

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]

        thumb_tip = np.array([int(landmarks[4].x * w), int(landmarks[4].y * h)])
        index_tip = np.array([int(landmarks[8].x * w), int(landmarks[8].y * h)])
        middle_tip = np.array([int(landmarks[12].x * w), int(landmarks[12].y * h)])

        # PALM CENTER TRACKING
        palm_x = (landmarks[0].x + landmarks[9].x) / 2 * w
        palm_y = (landmarks[0].y + landmarks[9].y) / 2 * h
        cv2.circle(frame, (int(palm_x), int(palm_y)), 8, (0, 255, 255), -1)

        dist_left = np.linalg.norm(index_tip - thumb_tip)
        dist_right = np.linalg.norm(middle_tip - thumb_tip)

        index_is_up = landmarks[8].y < landmarks[6].y
        middle_is_up = landmarks[12].y < landmarks[10].y

        # A. SCROLL MODE
        if index_is_up and middle_is_up:
            cv2.putText(frame, "Scroll", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            if palm_y < h // 2 - 20: pyautogui.scroll(20)  
            elif palm_y > h // 2 + 20: pyautogui.scroll(-20) 
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False

        # B. 1-EURO FILTERED MOUSE TRACKING
        elif not middle_is_up:
            # 1. Map raw palm coordinates to raw screen coordinates
            raw_screen_x = np.interp(palm_x, (FRAME_PAD, w - FRAME_PAD), (0, screen_width))
            raw_screen_y = np.interp(palm_y, (FRAME_PAD, h - FRAME_PAD), (0, screen_height))

            # 2. Pass raw screen coordinates through the 1 Euro Filter
            filtered_x = filter_x(raw_screen_x, current_time)
            filtered_y = filter_y(raw_screen_y, current_time)

            # 3. Apply Deadzone: Only move if the change is noticeable (> 1 pixel)
            dist_moved = np.hypot(filtered_x - prev_filtered_x, filtered_y - prev_filtered_y)
            if dist_moved > 1.0:
                pyautogui.moveTo(int(filtered_x), int(filtered_y))
                prev_filtered_x, prev_filtered_y = filtered_x, filtered_y

            # C. LEFT CLICK / DRAG
            if dist_left < 15:
                if not is_dragging:
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
                cv2.putText(frame, "Dragging", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, tuple(index_tip), 15, (0, 255, 0), -1)
            else:
                if is_dragging:
                    pyautogui.mouseUp(button='left')
                    is_dragging = False

            # D. RIGHT CLICK
            if dist_right < 15:
                if not right_clicked:
                    pyautogui.click(button='right')
                    right_clicked = True
                    cv2.circle(frame, tuple(middle_tip), 15, (0, 0, 255), -1)
            else:
                right_clicked = False

    cv2.rectangle(frame, (FRAME_PAD, FRAME_PAD), (w - FRAME_PAD, h - FRAME_PAD), (255, 0, 0), 2)
    cv2.imshow('1-Euro VR Smoothed Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

if is_dragging: pyautogui.mouseUp(button='left')
cap.release()
cv2.destroyAllWindows()