import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import os
import time

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Finger tracking started — ESC to quit, open hand to clear")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MIN_RECORD_DIST   = 12
SMOOTH_WINDOW     = 9
VECTOR_LOOKBACK   = 25
TURN_THRESHOLD    = 12.0
MIN_SEG_LENGTH    = 30
SIMPLIFY_EPS      = 6

# ─────────────────────────────────────────────
#  CAMERA INIT — robust multi-index attempt
# ─────────────────────────────────────────────
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(3):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Warm up: wait until we actually get a frame
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        print(f"✅ Camera opened — index {idx}, backend {backend}")
                        return cap
                    time.sleep(0.1)
                cap.release()
    return None

cap = open_camera()
if cap is None:
    print("❌ No camera found. Make sure it is connected and not in use by another app.")
    exit(1)

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

cv2.namedWindow("Robot Path Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Robot Path Tracker", 1280, 720)

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
raw_points: list = []
smooth_buf: deque = deque(maxlen=SMOOTH_WINDOW)
segments:   list  = []

last_heading    = None
seg_start_idx   = 0
live_heading_deg = 0.0
live_turn_deg    = 0.0

# ─────────────────────────────────────────────
#  MATH UTILITIES
# ─────────────────────────────────────────────

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def normalize(v):
    mag = math.hypot(v[0], v[1])
    if mag < 1e-6:
        return None
    return (v[0]/mag, v[1]/mag)

def heading_deg(v):
    return math.degrees(math.atan2(-v[1], v[0]))

def signed_angle_between(v1, v2):
    dot   = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1]))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    angle = math.degrees(math.acos(dot))
    return angle if cross >= 0 else -angle

def stable_direction(pts, end_idx, lookback):
    start_idx = max(0, end_idx - lookback)
    if start_idx == end_idx:
        return None
    dx = pts[end_idx][0] - pts[start_idx][0]
    dy = pts[end_idx][1] - pts[start_idx][1]
    return normalize((dx, dy))

def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return pts

    def pt_line_dist(p, a, b):
        if a == b:
            return dist(p, a)
        n = abs((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0])
        d = dist(a, b)
        return n / d if d > 0 else 0

    max_d, max_i = 0.0, 0
    for i in range(1, len(pts)-1):
        d = pt_line_dist(pts[i], pts[0], pts[-1])
        if d > max_d:
            max_d, max_i = d, i

    if max_d > eps:
        left  = douglas_peucker(pts[:max_i+1], eps)
        right = douglas_peucker(pts[max_i:],   eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]

def fingers_up(lms):
    return [
        lms.landmark[8].y  < lms.landmark[6].y,
        lms.landmark[12].y < lms.landmark[10].y,
        lms.landmark[16].y < lms.landmark[14].y,
        lms.landmark[20].y < lms.landmark[18].y,
        lms.landmark[4].x  > lms.landmark[3].x,
    ]

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_arrow(img, origin, direction_unit, length=55, color=(0, 220, 255), thickness=2):
    end = (int(origin[0] + direction_unit[0]*length),
           int(origin[1] + direction_unit[1]*length))
    cv2.arrowedLine(img, origin, end, color, thickness, tipLength=0.35)

def draw_angle_arc(img, pt, v1, v2, angle_deg, color=(50, 230, 120)):
    r  = 38
    a1 = math.degrees(math.atan2(-v1[1], v1[0]))
    a2 = math.degrees(math.atan2(-v2[1], v2[0]))
    sa, ea = sorted([a1, a2])
    if ea - sa > 180:
        sa, ea = ea, sa + 360
    cv2.ellipse(img, pt, (r, r), 0, int(sa), int(ea), color, 2)
    mid_rad = math.radians((sa + ea) / 2)
    lx = int(pt[0] + (r + 18) * math.cos(mid_rad))
    ly = int(pt[1] - (r + 18) * math.sin(mid_rad))
    sign = "+" if angle_deg >= 0 else ""
    cv2.putText(img, f"{sign}{angle_deg:.1f}", (lx-10, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

def draw_ui_panel(img, draw_mode, live_heading, live_turn, segs):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (310, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    status_col = (0, 210, 80) if draw_mode else (0, 80, 220)
    cv2.rectangle(img, (10, 10), (300, 50), status_col, -1)
    label = "  DRAWING" if draw_mode else "  IDLE"
    cv2.putText(img, label, (14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)

    y = 80
    for title, val in [
        ("Heading",  f"{live_heading:+.1f}°"),
        ("Turn",     f"{live_turn:+.1f}°"),
        ("Segments", str(len(segs))),
    ]:
        cv2.putText(img, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,180), 1)
        cv2.putText(img, val,   (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,220,100), 2)
        y += 34

    cv2.line(img, (10, y), (300, y), (60, 60, 80), 1)
    y += 20
    cv2.putText(img, "Segment Log", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,180,255), 1)
    y += 26
    for seg in segs[-8:]:
        turn_sign = "L" if seg["turn_deg"] >= 0 else "R"
        line = (f"#{seg['id']:02d}  hdg={seg['heading_deg']:+.0f}  "
                f"turn={turn_sign}{abs(seg['turn_deg']):.0f}  d={seg['dist_px']:.0f}px")
        col  = (100, 220, 100) if seg["turn_deg"] >= 0 else (100, 140, 255)
        cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        y += 22
        if y > h - 30:
            break

    cv2.putText(img, "Open hand = clear  |  ESC = quit",
                (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90,90,110), 1)

# ─────────────────────────────────────────────
#  SEGMENT LOGIC
# ─────────────────────────────────────────────

def maybe_commit_segment(pts, seg_start, last_hdg, turn_threshold, min_len):
    end_idx  = len(pts) - 1
    cur_vec  = stable_direction(pts, end_idx, VECTOR_LOOKBACK)
    if cur_vec is None:
        return None, seg_start, last_hdg

    seg_dist = dist(pts[seg_start], pts[end_idx])
    if seg_dist < min_len:
        return None, seg_start, last_hdg

    if last_hdg is None:
        return None, seg_start, cur_vec

    turn = signed_angle_between(last_hdg, cur_vec)
    if abs(turn) < turn_threshold:
        return None, seg_start, last_hdg

    seg = {
        "id":          len(segments) + 1,
        "start":       pts[seg_start],
        "end":         pts[end_idx],
        "heading_deg": round(heading_deg(cur_vec), 1),
        "turn_deg":    round(turn, 1),
        "dist_px":     round(seg_dist, 1),
        "vec_in":      last_hdg,
        "vec_out":     cur_vec,
    }
    print(f"[SEG #{seg['id']}]  heading={seg['heading_deg']:+.1f}°  "
          f"turn={'L' if turn>=0 else 'R'}{abs(turn):.1f}°  dist={seg['dist_px']:.1f}px")
    return seg, end_idx, cur_vec

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

consecutive_failures = 0

while cap.isOpened():
    ok, img = cap.read()
    if not ok or img is None:
        consecutive_failures += 1
        if consecutive_failures > 30:
            print("❌ Too many consecutive frame failures — exiting.")
            break
        time.sleep(0.03)
        continue
    consecutive_failures = 0

    img = cv2.flip(img, 1)

    # Resize only if needed (avoids error if frame is already 1280x720)
    h_raw, w_raw = img.shape[:2]
    if (w_raw, h_raw) != (1280, 720):
        img = cv2.resize(img, (1280, 720))

    rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_mode = False

    if results.multi_hand_landmarks:
        for lms in results.multi_hand_landmarks:
            h, w, _ = img.shape

            rx = int(lms.landmark[8].x * w)
            ry = int(lms.landmark[8].y * h)
            smooth_buf.append((rx, ry))
            sx = int(np.mean([p[0] for p in smooth_buf]))
            sy = int(np.mean([p[1] for p in smooth_buf]))
            pos = (sx, sy)

            fingers  = fingers_up(lms)
            index_up = fingers[0]
            others   = fingers[1:4]
            all_up   = all(fingers)

            if index_up and not any(others):
                draw_mode = True

                if len(raw_points) == 0 or dist(pos, raw_points[-1]) >= MIN_RECORD_DIST:
                    raw_points.append(pos)

                    if len(raw_points) >= VECTOR_LOOKBACK:
                        cur_vec = stable_direction(raw_points, len(raw_points)-1, VECTOR_LOOKBACK)
                        if cur_vec:
                            live_heading_deg = heading_deg(cur_vec)
                            if last_heading:
                                live_turn_deg = signed_angle_between(last_heading, cur_vec)

                    new_seg, seg_start_idx, last_heading = maybe_commit_segment(
                        raw_points, seg_start_idx, last_heading,
                        TURN_THRESHOLD, MIN_SEG_LENGTH
                    )
                    if new_seg:
                        segments.append(new_seg)

            if all_up:
                raw_points.clear()
                smooth_buf.clear()
                segments.clear()
                last_heading      = None
                seg_start_idx     = 0
                live_heading_deg  = 0.0
                live_turn_deg     = 0.0
                print("─── PATH CLEARED ───")

            cv2.circle(img, pos, 16, (0, 255, 80), cv2.FILLED)
            cv2.circle(img, pos, 16, (0, 140, 40), 2)
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

    if len(raw_points) >= 2:
        simplified = douglas_peucker(raw_points, SIMPLIFY_EPS)
        for i in range(1, len(simplified)):
            cv2.line(img, simplified[i-1], simplified[i], (220, 80, 60), 3)

        # Arrow follows the LAST DRAWN SEGMENT direction
        if len(raw_points) >= 2:
            p1 = raw_points[-2]
            p2 = raw_points[-1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            seg_vec = normalize((dx, dy))

            if seg_vec:
                draw_arrow(img, p2, seg_vec)


    for seg in segments:
        if seg["vec_in"] and seg["vec_out"]:
            pt = seg["end"]
            draw_angle_arc(img, pt, seg["vec_in"], seg["vec_out"], seg["turn_deg"])
            cv2.circle(img, pt, 6, (50, 230, 120), -1)

    draw_ui_panel(img, draw_mode, live_heading_deg, live_turn_deg, segments)

    cv2.imshow("Robot Path Tracker", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")