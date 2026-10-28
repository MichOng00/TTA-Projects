import time
import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.88.1"

SHOW_MASK = False   # whether to show line mask
SHOW_DETECT = True  # whether to show detected object

DIR_LEFT = 2
DIR_RIGHT = 3

# --- Object detection (unripe cube) config ---
YOLO_WEIGHTS_PATH = "P34_AIMS_final_null.pt" # null trained on data with no cube (just line)

UNRIPE_LABEL = "unripe"       # label name the model uses for an unripe cube
UNRIPE_CONF_THRESHOLD = 0.7  # min confidence to react to a detection (0-1)
DETECT_EVERY_N_FRAMES = 1     # run YOLO every Nth frame (raise if the loop feels slow)
UNRIPE_Y_THRESHOLD = 200     # how low in the frame (px) a cube must be before we react
UNRIPE_X_MIN_FRAC = 0.1       # left edge of the horizontal swipe zone
UNRIPE_X_MAX_FRAC = 0.9       # right edge of the horizontal swipe zone

# --- Driving tuning knobs ---
# Just a bag of numbers the driving functions below read from.
CFG = {
    # Speed
    "max_speed": 20,              # speed on straights
    "min_speed": 10,              # speed floor on sharp turns
    "steering_at_min_speed": 15,  # |steering| at which speed bottoms out

    # PD steering
    "kp": 0.5,
    "kd": 0.2,

    # Smoothing / anti-jerk
    "error_deadband": 10,      # px; smaller errors are treated as 0
    "smoothing_alpha": 0.3,    # EMA weight for new readings (lower = smoother)
    "max_steering_delta": 20,  # max steering change per frame
    "max_speed_delta": 4,      # max speed change per frame
    "max_steering": 70,        # hard ceiling on steering sent to hardware

    # Lost-line handling
    "lost_line_threshold": 5,  # frames with no line before searching

    # End-of-course stop box
    "stop_box_min_area_frac": 0.55,  # min fraction of frame area to count as the box
    "stop_box_min_extent": 0.5,      # min fill ratio (area / bounding box) to count as solid
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def clamp(value, low, high):
    """Keep value within [low, high]."""
    return max(low, min(high, value))


def init_state():
    """Create fresh line-following state (used at startup, and again
    whenever we want to start clean, e.g. right after swiping a cube)."""
    return {
        "lost_line_count": 0,
        "smoothed_steering": 0.0,
        "smoothed_speed": float(CFG["max_speed"]),
    }


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------


def swipe_cube(got):
    """Swipe away a large cube (unripe fruit) with the arm."""
    print("[swipe_cube] swiping unripe cube out of the way")
    got.mechanical_joint_control(-40, -40, -60, 500)
    time.sleep(0.6)
    got.mechanical_joint_control(30, -40, -60, 1000)
    time.sleep(1)
    got.mechanical_joint_control(0, 50, 50, 800)
    time.sleep(0.6)


# ---------------------------------------------------------------------------
# Object detection (unripe cube)
# ---------------------------------------------------------------------------


def load_detector(weights_path=YOLO_WEIGHTS_PATH):
    """Load the YOLO model from disk."""
    return YOLO(weights_path)


def draw_detections(frame, results):
    """Draw every detected box and label onto the frame."""
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
    return frame


def find_unripe_detection(results, label=UNRIPE_LABEL, min_conf=UNRIPE_CONF_THRESHOLD):
    """Return the highest-confidence box matching `label` above min_conf,
    as (x1, y1, x2, y2, conf), or None if there isn't one."""
    best = None
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if r.names[cls_id] != label:
                continue
            conf = float(box.conf[0])
            if conf < min_conf:
                continue
            if best is None or conf > best[4]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                best = (x1, y1, x2, y2, conf)
    return best


def check_for_unripe_cube(got, detector, frame):
    """Run YOLO on this frame. If a close, high-confidence unripe cube is
    found, stop and swipe it away. Returns True if that happened."""
    yolo_results = detector(frame, verbose=False)

    if SHOW_DETECT:
        cv2.imshow("YOLO Detection", draw_detections(frame.copy(), yolo_results))

    box = find_unripe_detection(yolo_results)
    if box is None:
        return False

    x1, y1, x2, y2, conf = box
    if y2 <= UNRIPE_Y_THRESHOLD:
        return False  # seen, but not close enough yet

    frame_width = frame.shape[1]
    cube_center_x = (x1 + x2) / 2
    if not frame_width * UNRIPE_X_MIN_FRAC <= cube_center_x <= frame_width * UNRIPE_X_MAX_FRAC:
        print(f"[unripe] detected outside swipe zone at x={cube_center_x:.0f} -- ignoring.")
        return False

    print(f"[unripe] detected at ({x1},{y1})-({x2},{y2}) conf={conf:.2f} -- stopping to swipe.")
    stop(got)
    swipe_cube(got)
    return True


# ---------------------------------------------------------------------------
# Vision (line following)
# ---------------------------------------------------------------------------


def preprocess_frame(frame):
    """Convert to grayscale and blur, ready for thresholding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_stop_box(frame, blurred, threshold=180, min_area_frac=0.35, min_extent=0.6):
    """Check whether a big, solid bright blob (the end-of-course box)
    fills enough of the frame to count as reached."""
    height, width = frame.shape[:2]
    frame_area = height * width

    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < frame_area * min_area_frac:
        return False, None, mask

    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0.0
    if extent < min_extent:
        return False, None, mask

    return True, (x, y, w, h), mask


def find_centroid_in_strip(mask_strip, min_area=60):
    """Find the center point of the largest white blob in a strip."""
    contours, _ = cv2.findContours(
        mask_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def get_line_position_multistrip(
    frame, blurred, threshold=230, num_strips=3, scan_height_frac=0.4, scan_width_frac=0.75
):
    """Look for the line in a few horizontal strips near the bottom of the
    frame. Returns the (x, y, weight) found in each strip -- closest to
    the robot first -- plus the threshold mask and a debug overlay."""
    height, width = frame.shape[:2]

    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # only look in a centered vertical band
    scan_w = int(width * scan_width_frac)
    scan_left = (width - scan_w) // 2
    scan_right = scan_left + scan_w

    # mask out everything outside that band
    width_mask = np.zeros_like(mask)
    width_mask[:, scan_left:scan_right] = mask[:, scan_left:scan_right]
    mask = width_mask

    scan_top = int(height * (1 - scan_height_frac))
    scan_zone_height = height - scan_top
    strip_h = scan_zone_height // num_strips

    overlay = frame.copy()
    cv2.line(overlay, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
    # draw the band edges for debugging
    cv2.line(overlay, (scan_left, 0), (scan_left, height), (255, 0, 255), 1)
    cv2.line(overlay, (scan_right, 0), (scan_right, height), (255, 0, 255), 1)

    results = []  # closest strip (bottom) first
    for i in range(num_strips):
        # i=0 is the strip nearest the robot
        strip_bottom = height - i * strip_h
        strip_top = height - (i + 1) * strip_h
        strip_top = max(strip_top, scan_top)

        strip_mask = mask[strip_top:strip_bottom, :]
        centroid = find_centroid_in_strip(strip_mask)

        cv2.rectangle(overlay, (0, strip_top), (width, strip_bottom), (255, 0, 0), 1)

        if centroid is not None:
            cx, cy_local = centroid
            cy_global = strip_top + cy_local
            weight = num_strips - i  # closer strips weighted more heavily
            results.append((cx, cy_global, weight))
            cv2.circle(overlay, (cx, cy_global), 6, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                str(i),
                (cx + 10, cy_global),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    return results, mask, overlay


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def turn(got, steering, forward_speed):
    """Drive forward while steering left or right by the given amount."""
    if steering < 0:
        got.transform_move_turn(0, int(forward_speed), DIR_LEFT, int(-steering))
    else:
        got.transform_move_turn(0, int(forward_speed), DIR_RIGHT, int(steering))

    print(f"[turn] steering={steering:.1f}, forward_speed={forward_speed}")


def stop(got):
    """Stop the robot."""
    got.transform_stop()
    print("[stop]")


def search_for_line(got):
    """Line is lost -- creep forward slowly while looking for it again."""
    # got.transform_move_speed(0, 5)
    got.transform_stop()
    print("[search_for_line] line lost, searching...")


# ---------------------------------------------------------------------------
# Control math
# ---------------------------------------------------------------------------


def compute_steering_error(results, frame_width):
    """Turn the strip results into a steering error (how far off-center)
    and a curvature signal (how the line bends further ahead)."""
    if not results:
        return None, None

    center_x = frame_width // 2

    # average error, weighted toward the closer strips
    total_weight = sum(w for _, _, w in results)
    weighted_error = sum((cx - center_x) * w for cx, _, w in results) / total_weight

    # compare nearest vs farthest strip to estimate curvature
    near_cx = results[0][0]
    far_cx = results[-1][0]
    curvature = far_cx - near_cx  # positive = line curving right ahead

    return weighted_error, curvature


def pd_steering(error, curvature, kp, kd):
    """Simple PD controller: steering = kp * error + kd * curvature."""
    return kp * error + kd * curvature


def speed_for_steering(steering, max_speed, min_speed, steering_at_min_speed):
    """Slow down proportionally to how hard we're turning."""
    turn_fraction = min(abs(steering) / steering_at_min_speed, 1.0)
    return max_speed - turn_fraction * (max_speed - min_speed)


def smooth_and_send(got, state, raw_steering, raw_speed):
    """Blend the new steering/speed with last frame's (EMA smoothing),
    limit how fast they can change in one frame, clamp to a safe max,
    then send the drive command."""
    alpha = CFG["smoothing_alpha"]
    target_steering = alpha * raw_steering + (1 - alpha) * state["smoothed_steering"]
    target_speed = alpha * raw_speed + (1 - alpha) * state["smoothed_speed"]

    steering_delta = clamp(
        target_steering - state["smoothed_steering"],
        -CFG["max_steering_delta"], CFG["max_steering_delta"],
    )
    speed_delta = clamp(
        target_speed - state["smoothed_speed"],
        -CFG["max_speed_delta"], CFG["max_speed_delta"],
    )

    state["smoothed_steering"] += steering_delta
    state["smoothed_speed"] += speed_delta

    # rate-limiting bounds the change, not the final value -- clamp it too
    state["smoothed_steering"] = clamp(
        state["smoothed_steering"], -CFG["max_steering"], CFG["max_steering"]
    )

    turn(got, state["smoothed_steering"], state["smoothed_speed"])


def steer_along_line(got, strips, frame_width, state):
    """Steer for one frame: drive smoothly along the line, or if it's
    been missing too long, stop and search for it."""
    error, curvature = compute_steering_error(strips, frame_width)

    if error is None:
        state["lost_line_count"] += 1
        print(f"Line not found in any strip ({state['lost_line_count']} frames)")
        if state["lost_line_count"] >= CFG["lost_line_threshold"]:
            stop(got)
            search_for_line(got)
            state["smoothed_steering"] = 0.0
            state["smoothed_speed"] = float(CFG["max_speed"])
        return

    state["lost_line_count"] = 0

    # ignore tiny errors to avoid wobbling
    if abs(error) < CFG["error_deadband"]:
        error = 0.0

    raw_steering = pd_steering(error, curvature, CFG["kp"], CFG["kd"])
    raw_speed = speed_for_steering(
        raw_steering, CFG["max_speed"], CFG["min_speed"], CFG["steering_at_min_speed"]
    )
    print(
        f"error={error:.1f}, curvature={curvature:.1f}, "
        f"raw_steering={raw_steering:.1f}, strips_found={len(strips)}"
    )

    smooth_and_send(got, state, raw_steering, raw_speed)


# ---------------------------------------------------------------------------
# Per-frame helpers used by main()
# ---------------------------------------------------------------------------


def connect_robot(ip=ROBOT_IP):
    """Connect to the robot, open its camera, and set it into a
    driving-ready pose."""
    got = ugot.UGOT()
    got.initialize(ip)
    got.open_camera()
    got.transform_adaption_control(False)
    got.transform_set_chassis_height(7)
    got.mechanical_clamp_release()
    got.mechanical_joint_control(0, 70, 70, 800)
    # got.load_models(["gesture"])
    time.sleep(1)
    return got


def get_frame(got):
    """Grab one frame from the robot's camera and decode it.
    Returns None if the grab or decode failed."""
    raw = got.read_camera_data()
    if not raw:
        return None
    nparr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def stop_at_box(got, overlay, box_bbox, box_mask):
    """The end-of-course box fills the frame -- stop, nudge forward onto
    it, then stop again."""
    x, y, w, h = box_bbox
    print(f"[stop_box] detected at x={x}, y={y}, w={w}, h={h} -- stopping.")
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
    stop(got)
    got.transform_move_speed_times(0, 20, 23, 1)
    stop(got)  # make sure the nudge above doesn't leave us drifting
    show_debug_windows(overlay, box_mask)


def follow_line(got, frame, state):
    """Handle one frame of line-following: find the line and the
    end-of-course box, then steer or stop. Returns True if the box was
    reached (course finished)."""
    blurred = preprocess_frame(frame)
    strips, mask, overlay = get_line_position_multistrip(frame, blurred)

    box_found, box_bbox, box_mask = detect_stop_box(
        frame, blurred,
        min_area_frac=CFG["stop_box_min_area_frac"],
        min_extent=CFG["stop_box_min_extent"],
    )
    if box_found:
        stop_at_box(got, overlay, box_bbox, box_mask)
        return "box found"

    steer_along_line(got, strips, frame.shape[1], state)
    show_debug_windows(overlay, mask)
    return None


def show_debug_windows(overlay=None, mask=None):
    """Show whichever debug windows are turned on at the top of the file."""
    if SHOW_DETECT and overlay is not None:
        cv2.imshow("Webcam Feed", overlay)
    if SHOW_MASK and mask is not None:
        cv2.imshow("Mask", mask)


def check_quit():
    """Return True if 'q' was pressed (only matters when a debug window
    is open)."""
    if not (SHOW_DETECT or SHOW_MASK):
        return False
    return cv2.waitKey(1) & 0xFF == ord("q")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    """Connect to the robot, then loop: check for a cube to swipe,
    otherwise follow the line, until the course ends or 'q' is pressed."""
    got = connect_robot()
    detector = load_detector()
    state = init_state()
    frame_count = 0

    # gesture_count = 0
    # while True:
    #     gesture = got.get_gesture_result()
    #     if gesture == "Stone":
    #         gesture_count += 1
    #         if gesture_count > 3:
    #             break
    #     else:
    #         gesture_count = 0

    while True:
        frame = get_frame(got)
        if frame is None:
            print("Failed to grab or decode frame")
            break

        frame_count += 1
        check_cube_this_frame = frame_count % DETECT_EVERY_N_FRAMES == 0

        if check_cube_this_frame and check_for_unripe_cube(got, detector, frame):
            state = init_state()  # cube handled -- start line-following clean
        elif follow_line(got, frame, state) == "box found":
            break  # end-of-course box reached

        if check_quit():
            stop(got)
            break

    if SHOW_MASK or SHOW_DETECT:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    ### test swipe only:
    # got = connect_robot()
    # swipe_cube(got)