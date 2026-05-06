import cv2
import mediapipe as mp
import math
import time

print("Finger tracking started...")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Window
cv2.namedWindow("Finger Drawing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Finger Drawing", 1280, 720)

points = []
MIN_DISTANCE = 5

robot_buffer = []
last_buffer_time = time.time()
BUFFER_INTERVAL = 1.5

robot_data = {"angle": 0, "distance": 0}


last_vector = None


# ---------------- FUNCTIONS ---------------- #

def fingers_up(handLms):
    fingers = []
    fingers.append(handLms.landmark[8].y < handLms.landmark[6].y)
    fingers.append(handLms.landmark[12].y < handLms.landmark[10].y)
    fingers.append(handLms.landmark[16].y < handLms.landmark[14].y)
    fingers.append(handLms.landmark[20].y < handLms.landmark[18].y)
    fingers.append(handLms.landmark[4].x > handLms.landmark[3].x)
    return fingers


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1, min(1, cos_theta))

    return round(math.degrees(math.acos(cos_theta)), 2)


# ✅ FIXED FUNCTION
def calculate_path_stats(point_list):
    global last_vector

    if len(point_list) < 2:
        return {"angle": 0, "distance": 0}

    # ONLY last segment distance
    distance = euclidean_distance(point_list[-2], point_list[-1])

    current_vector = vector(point_list[-2], point_list[-1])

    if last_vector is None:
        last_vector = current_vector
        return {"angle": 0, "distance": round(distance, 2)}

    angle = angle_between(last_vector, current_vector)
    last_vector = current_vector

    return {
        "angle": angle,
        "distance": round(distance, 2)
    }


# ---------------- MAIN LOOP ---------------- #

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    draw_mode = False
    current_time = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape

            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            fingers = fingers_up(handLms)

            index_up = fingers[0]
            all_up = all(fingers)

            # ✏️ DRAW MODE (only index finger)
            if index_up and not any(fingers[1:4]):
                draw_mode = True

                if len(points) == 0 or euclidean_distance((x, y), points[-1]) >= MIN_DISTANCE:
                    points.append((x, y))

            # 🧹 CLEAR when hand open
            if all_up:
                points = []
                robot_data = {"angle": 0, "distance": 0}
                last_vector = None

            cv2.circle(img, (x, y), 20, (0, 255, 0), cv2.FILLED)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # 🤖 SEND DATA
    if draw_mode and current_time - last_buffer_time >= BUFFER_INTERVAL:
        if len(points) >= 2:
            robot_data = calculate_path_stats(points)

            robot_buffer.append({
                "timestamp": current_time,
                "data": robot_data.copy(),
                "points": len(points)
            })

            last_buffer_time = current_time

            print(f"ANGLE={robot_data['angle']}° | DIST={robot_data['distance']} px")

    # 🎨 DRAW PATH
    for i in range(1, len(points)):
        cv2.line(img, points[i-1], points[i], (255, 0, 0), 3)

    # UI
    mode_text = "DRAWING" if draw_mode else "NOT DRAWING"
    cv2.putText(img, mode_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if draw_mode else (0, 0, 255), 3)

    cv2.putText(img, f"Angle: {robot_data['angle']}°", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(img, f"Distance: {robot_data['distance']} px", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Finger Drawing", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()