import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import os

# ❗ Temporarily REMOVE log suppression for debugging
# os.environ["GLOG_minloglevel"] = "3"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("🚀 Finger tracking started — ESC to quit")

# CONFIG
MIN_RECORD_DIST = 5   # lowered for testing
SMOOTH_WINDOW   = 9

# MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# CAMERA
cap = cv2.VideoCapture(0)

# 🔴 CHECK CAMERA
if not cap.isOpened():
    print("❌ ERROR: Camera not detected")
    exit()
else:
    print("✅ Camera opened successfully")

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

# STATE
raw_points = []
smooth_buf = deque(maxlen=SMOOTH_WINDOW)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def fingers_up(lms):
    return [
        lms.landmark[8].y  < lms.landmark[6].y,
        lms.landmark[12].y < lms.landmark[10].y,
        lms.landmark[16].y < lms.landmark[14].y,
        lms.landmark[20].y < lms.landmark[18].y,
        lms.landmark[4].x  > lms.landmark[3].x,
    ]

# LOOP
while True:
    ok, img = cap.read()

    if not ok:
        print("⚠️ Frame not received")
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # 🔍 DEBUG: hand detection
    if results.multi_hand_landmarks:
        print("🖐️ Hand detected")

        for lms in results.multi_hand_landmarks:
            h, w, _ = img.shape

            x = int(lms.landmark[8].x * w)
            y = int(lms.landmark[8].y * h)

            smooth_buf.append((x, y))
            sx = int(np.mean([p[0] for p in smooth_buf]))
            sy = int(np.mean([p[1] for p in smooth_buf]))
            pos = (sx, sy)

            fingers = fingers_up(lms)
            index_up = fingers[0]
            others = fingers[1:4]
            all_up = all(fingers)

            # 🔍 DEBUG gestures
            print(f"Fingers: {fingers}")

            # DRAW MODE
            if index_up and not any(others):
                print("✏️ DRAW MODE")

                if len(raw_points) == 0 or dist(pos, raw_points[-1]) > MIN_RECORD_DIST:
                    raw_points.append(pos)

            # CLEAR
            if all_up:
                print("🧹 CLEAR")
                raw_points.clear()

            cv2.circle(img, pos, 10, (0,255,0), -1)
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

    else:
        print("❌ No hand detected")

    # DRAW PATH
    for i in range(1, len(raw_points)):
        cv2.line(img, raw_points[i-1], raw_points[i], (0,0,255), 2)

    cv2.imshow("Test", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()