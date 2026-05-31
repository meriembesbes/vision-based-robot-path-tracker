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
VECTOR_LOOKBACK   = 40
TURN_THRESHOLD    = 12.0   # degrees — minimum turn to commit a new segment
MIN_SEG_LENGTH    = 30     # pixels — minimum segment length before checking for turn
SIMPLIFY_EPS      = 6

# ─────────────────────────────────────────────
#  CAMERA INIT
# ─────────────────────────────────────────────
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(3):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
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
    print("❌ No camera found.")
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
raw_points: list  = []
smooth_buf: deque = deque(maxlen=SMOOTH_WINDOW)

# Each segment: { id, start, end, vec, heading_deg, relative_angle_deg, dist_px }
segments:   list  = []

# The direction vector of the most recently committed segment.
# New relative angles are measured FROM this vector.
last_committed_vec = None   # unit vector (dx, dy) of previous segment
seg_start_idx      = 0      # index into raw_points where current segment began

# ─────────────────────────────────────────────
#  MATH UTILITIES
# ─────────────────────────────────────────────

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def normalize(v):
    mag = math.hypot(v[0], v[1])
    return (v[0]/mag, v[1]/mag) if mag > 1e-6 else None

def heading_deg(v):
    """Absolute heading: 0=right, +CCW (screen coords)."""
    return math.degrees(math.atan2(-v[1], v[0]))

def signed_angle_between(v_ref, v_new):
    """
    Signed angle from v_ref to v_new.
    Positive = left turn (CCW on screen), Negative = right turn (CW on screen).
    Range: (-180, 180].
    """
    dot   = max(-1.0, min(1.0, v_ref[0]*v_new[0] + v_ref[1]*v_new[1]))
    cross = v_ref[0]*v_new[1] - v_ref[1]*v_new[0]
    angle = math.degrees(math.acos(dot))
    return angle if cross >= 0 else -angle

def stable_direction(pts, end_idx, lookback):
    """Compute a stable direction vector by looking back `lookback` points."""
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
#  SEGMENT LOGIC
#
#  Key idea (matching your schema):
#    • The FIRST segment has no reference → relative_angle = 0°  (it IS the reference)
#    • Every subsequent segment's angle is measured relative to the PREVIOUS segment's
#      direction vector, NOT relative to absolute North/East.
#    • Straight continuation → 0°
#    • Left turn → positive degrees
#    • Right turn → negative degrees
# ─────────────────────────────────────────────

def try_commit_segment(pts, seg_start, prev_vec):
    """
    Returns (segment_dict | None, new_seg_start, new_prev_vec)

    A segment is committed when the finger has travelled far enough AND
    the current direction differs from prev_vec by more than TURN_THRESHOLD.
    """
    end_idx  = len(pts) - 1
    cur_vec  = stable_direction(pts, end_idx, VECTOR_LOOKBACK)
    if cur_vec is None:
        return None, seg_start, prev_vec

    seg_dist = dist(pts[seg_start], pts[end_idx])
    if seg_dist < MIN_SEG_LENGTH:
        return None, seg_start, prev_vec

    # ── First segment: no previous vector, just record direction ──
    if prev_vec is None:
        seg_id = len(segments) + 1
        seg = {
            "id":                   seg_id,
            "start":                pts[seg_start],
            "end":                  pts[end_idx],
            "vec":                  cur_vec,
            "heading_deg":          round(heading_deg(cur_vec), 1),
            "relative_angle_deg":   0.0,   # First segment has no turn reference
            "dist_px":              round(seg_dist, 1),
        }
        print(f"[SEG #{seg_id:02d}]  "
              f"relative_angle=0.0°  "
              f"heading={seg['heading_deg']:+.1f}°  "
              f"length={seg['dist_px']:.1f}px")
        return seg, end_idx, cur_vec

    # ── Compute relative turn angle ──
    relative_angle = signed_angle_between(prev_vec, cur_vec)

    if abs(relative_angle) < TURN_THRESHOLD:
        # Still going straight — update direction but don't commit
        return None, seg_start, cur_vec

    # ── Commit the segment that just ended (from seg_start → end_idx) ──
    seg_id = len(segments) + 1

    # The segment that is being committed is the one FROM seg_start TO now.
    # Its own direction is cur_vec (what we measured).
    # Its relative angle is vs. prev_vec (the previous segment's direction).
    seg = {
        "id":                   seg_id,
        "start":                pts[seg_start],
        "end":                  pts[end_idx],
        "vec":                  cur_vec,
        "heading_deg":          round(heading_deg(cur_vec), 1),
        "relative_angle_deg":   round(relative_angle, 1),   # ← THIS is θ in your schema
        "dist_px":              round(seg_dist, 1),
    }

    direction_label = "L" if relative_angle >= 0 else "R"
    print(f"[SEG #{seg_id:02d}]  "
          f"relative_angle={direction_label}{abs(relative_angle):.1f}°  "
          f"heading={seg['heading_deg']:+.1f}°  "
          f"length={seg['dist_px']:.1f}px")

    # The committed segment's direction becomes the new reference
    return seg, end_idx, cur_vec

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_arrow(img, origin, direction_unit, length=55, color=(0, 220, 255), thickness=2):
    end = (int(origin[0] + direction_unit[0]*length),
           int(origin[1] + direction_unit[1]*length))
    cv2.arrowedLine(img, origin, end, color, thickness, tipLength=0.35)

def draw_turn_arc(img, pt, v_in, v_out, angle_deg):
    """Draw a small arc at a turn point showing the relative angle."""
    r   = 38
    a1  = math.degrees(math.atan2(-v_in[1],  v_in[0]))
    a2  = math.degrees(math.atan2(-v_out[1], v_out[0]))

    # Normalise to [0, 360)
    a1 %= 360
    a2 %= 360
    diff = (a2 - a1 + 360) % 360
    if diff > 180:
        sa, ea = int(a2), int(a1)
    else:
        sa, ea = int(a1), int(a2)

    color = (100, 220, 100) if angle_deg >= 0 else (100, 140, 255)
    cv2.ellipse(img, pt, (r, r), 0, sa, ea, color, 2)

    mid_rad = math.radians((sa + ea) / 2)
    lx = int(pt[0] + (r + 20) * math.cos(mid_rad))
    ly = int(pt[1] - (r + 20) * math.sin(mid_rad))
    sign = "+" if angle_deg >= 0 else ""
    cv2.putText(img, f"{sign}{angle_deg:.1f}", (lx-14, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

def draw_ui_panel(img, draw_mode, segs):
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (340, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Status badge
    status_col = (0, 210, 80) if draw_mode else (0, 80, 220)
    cv2.rectangle(img, (10, 10), (330, 50), status_col, -1)
    label = "  DRAWING" if draw_mode else "  IDLE"
    cv2.putText(img, label, (14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2)

    y = 68

    # ── Angle legend ──
    cv2.putText(img, "Relative Angle  (ref = prev seg)", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 200), 1)
    y += 18
    cv2.putText(img, "  0°  = straight  |  +L  -R",
                (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1)
    y += 22

    cv2.line(img, (10, y), (330, y), (60, 60, 80), 1)
    y += 14

    # ── Latest segment ──
    cv2.putText(img, "Last Segment", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 255), 1)
    y += 26

    if segs:
        last = segs[-1]
        ra   = last["relative_angle_deg"]
        dir_label = "L" if ra >= 0 else "R"
        angle_color = (100, 220, 100) if ra >= 0 else (100, 140, 255)

        cv2.putText(img, "Turn  :", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 180), 1)
        cv2.putText(img, f"{dir_label}  {abs(ra):.1f} deg", (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, angle_color, 2)
        y += 30

        cv2.putText(img, "Length:", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 180), 1)
        cv2.putText(img, f"{last['dist_px']:.1f} px", (110, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 220, 100), 2)
        y += 34
    else:
        cv2.putText(img, "No segment yet", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,120), 1)
        y += 34

    cv2.line(img, (10, y), (330, y), (60, 60, 80), 1)
    y += 14

    # ── History ──
    cv2.putText(img, "Segment History", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 255), 1)
    y += 26

    row_h    = 50
    max_rows = max(1, (h - y - 30) // row_h)
    visible  = segs[-max_rows:]

    for seg in visible:
        ra        = seg["relative_angle_deg"]
        dir_label = "L" if ra >= 0 else "R"
        angle_col = (100, 220, 100) if ra >= 0 else (100, 140, 255)

        cv2.putText(
            img,
            f"#{seg['id']:02d}  Turn: {dir_label} {abs(ra):.1f}°",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, angle_col, 1
        )
        y += 22
        cv2.putText(
            img,
            f"      Len: {seg['dist_px']:.1f}px   Hdg: {seg['heading_deg']:+.1f}°",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 160), 1
        )
        y += row_h - 22
        if y > h - 30:
            break

    cv2.putText(img, "Open hand = clear  |  ESC = quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 110), 1)

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

                    new_seg, seg_start_idx, last_committed_vec = try_commit_segment(
                        raw_points, seg_start_idx, last_committed_vec
                    )
                    if new_seg:
                        segments.append(new_seg)

            if all_up:
                raw_points.clear()
                smooth_buf.clear()
                segments.clear()
                last_committed_vec = None
                seg_start_idx      = 0
                print("─── PATH CLEARED ───")

            cv2.circle(img, pos, 16, (0, 255, 80), cv2.FILLED)
            cv2.circle(img, pos, 16, (0, 140, 40), 2)
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

    # ── Draw path ──
    if len(raw_points) >= 2:
        simplified = douglas_peucker(raw_points, SIMPLIFY_EPS)
        for i in range(1, len(simplified)):
            cv2.line(img, simplified[i-1], simplified[i], (220, 80, 60), 3)

        p1 = raw_points[-2]
        p2 = raw_points[-1]
        seg_vec = normalize((p2[0]-p1[0], p2[1]-p1[1]))
        if seg_vec:
            draw_arrow(img, p2, seg_vec)

    # ── Annotate turn points ──
    for i, seg in enumerate(segments):
        pt = seg["end"]
        cv2.circle(img, pt, 7, (50, 230, 120), -1)

        # Draw the arc showing θ (relative angle)
        if i > 0:
            prev_vec = segments[i-1]["vec"]
            draw_turn_arc(img, pt, prev_vec, seg["vec"], seg["relative_angle_deg"])
        elif i == 0:
            # First committed segment — show its absolute heading only
            pass

        # Label with θ symbol + value
        ra        = seg["relative_angle_deg"]
        dir_label = "L" if ra >= 0 else "R"
        angle_col = (100, 220, 100) if ra >= 0 else (100, 140, 255)
        cv2.putText(
            img,
            f"\u03b8{seg['id']}={dir_label}{abs(ra):.0f}",
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, angle_col, 2
        )

    draw_ui_panel(img, draw_mode, segments)

    cv2.imshow("Robot Path Tracker", img)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")