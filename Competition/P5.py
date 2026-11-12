from dataclasses import dataclass
import time
import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.88.1"  # TUNE: your robot's IP

SHOW_MASK = False
SHOW_DETECT = True

DIR_LEFT = 2
DIR_RIGHT = 3

CHASSIS_HEIGHT = 7  # fixed — set once at startup, NEVER changed after

# --- Object detection ---
YOLO_WEIGHTS_PATH = "best.pt"

RIPE_LABEL = "ripe"      # small cubes -> pick up
# ("unripe" = large cubes -> ignored entirely, we just drive over them)

RIPE_CONF_THRESHOLD = 0.70  # TUNE: lower (e.g. 0.6) if it misses real cubes,
                            # raise if it stops for things that aren't cubes

DETECT_EVERY_N_FRAMES = 1   # TUNE: raise to 2-3 if the loop feels laggy

RIPE_Y_THRESHOLD = 335      # TUNE: cube's bbox bottom edge (px from top of frame)
                            # must be below this before the robot stops to grab.
                            # Bigger = cube must be closer.

PICKUP_COOLDOWN_S = 0.5     # ignore detections this long after a pickup

# Arm pose while driving (up, out of the camera's way)
DRIVE_POSE = (0, 70, 70)


@dataclass
class Config:
    # Speed (slower than v1)
    max_speed: float = 17  # TUNE: lower further (e.g. 15) if still too fast
    min_speed: float = 10
    steering_at_min_speed: float = 15

    # PD steering
    kp: float = 0.5
    kd: float = 0.2

    # Smoothing / anti-jerk
    error_deadband: float = 10
    smoothing_alpha: float = 0.3
    max_steering_delta: float = 20
    max_speed_delta: float = 4
    max_steering: float = 70  # hard ceiling on |steering| sent to hardware

    # Lost-line handling
    lost_line_threshold: int = 5

    # End-of-course stop box (big white box fills the camera)
    stop_box_min_area_frac: float = 0.55  # TUNE: fraction of frame the box must fill
    stop_box_min_extent: float = 0.5      # solidity — rejects scattered bright blobs


# ---------------------------------------------------------------------------
# Arm — pick and place (from pickup.ipynb, NO chassis height changes)
# ---------------------------------------------------------------------------


def drive_pose(got):
    """Arm up and out of the way for driving/camera."""
    got.mechanical_joint_control(*DRIVE_POSE, 800)
    time.sleep(1)


def pickup_cube(got):
    """Grab the small cube in front and drop it in the compartment above.
    Chassis height is never touched."""
    print("[pickup_cube] grabbing ripe cube")
    got.mechanical_clamp_release()
    got.mechanical_joint_control(0, -40, -50, 1000)
    time.sleep(1)
    got.mechanical_clamp_close()
    time.sleep(0.3)
    got.mechanical_joint_control(0, 0, -90, 500)
    time.sleep(0.5)
    got.mechanical_joint_control(0, 90, 55, 1000)
    time.sleep(1.1)
    got.mechanical_clamp_release()
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------


def load_detector(weights_path=YOLO_WEIGHTS_PATH):
    return YOLO(weights_path)


def draw_detections(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            label = r.names[int(box.cls[0])]
            color = (0, 255, 0) if label == RIPE_LABEL else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def find_ripe_detection(results, label=RIPE_LABEL, min_conf=RIPE_CONF_THRESHOLD):
    """Highest-confidence 'ripe' (small cube) box above min_conf,
    as (x1, y1, x2, y2, conf), else None. 'unripe' is never returned."""
    best = None
    for r in results:
        for box in r.boxes:
            if r.names[int(box.cls[0])] != label:
                continue  # ignores unripe/large cubes entirely
            conf = float(box.conf[0])
            if conf < min_conf:
                continue
            if best is None or conf > best[4]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                best = (x1, y1, x2, y2, conf)
    return best


DEBUG_DETECTIONS = True  # print what YOLO sees each frame. Diagnostic only --
                         # set False to silence. Changes no robot behaviour.


def log_detections(results, threshold_y=RIPE_Y_THRESHOLD, min_conf=RIPE_CONF_THRESHOLD):
    """Print every box YOLO produced, including ones that do NOT trigger a
    pickup, and name the gate that rejected each. Read-only -- decides nothing.

    Tracing a missed cube:
      skip:not-ripe  -> misclassified as unripe
      skip:conf<...  -> seen, but confidence too low (glare?)
      skip:y2<=...   -> seen, but never got close enough
      (no line at all for that cube) -> never detected in any frame
    """
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            y2 = int(box.xyxy[0].cpu().numpy()[3])

            gates = []
            if label != RIPE_LABEL:
                gates.append("not-ripe")
            if conf < min_conf:
                gates.append(f"conf<{min_conf}")
            if y2 <= threshold_y:
                gates.append(f"y2<={threshold_y}")

            verdict = "PICKUP" if not gates else "skip:" + ",".join(gates)
            print(f"    [see] {label:6s} conf={conf:.2f} y2={y2:3d} -> {verdict}")


# ---------------------------------------------------------------------------
# Vision (line following) — unchanged from the new line-follow code
# ---------------------------------------------------------------------------


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_stop_box(frame, blurred, threshold=180, min_area_frac=0.35, min_extent=0.6):
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
    contours, _ = cv2.findContours(mask_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def get_line_position_multistrip(frame, blurred, threshold=230, num_strips=3,
                                 scan_height_frac=0.4, scan_width_frac=0.75):
    height, width = frame.shape[:2]
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    scan_w = int(width * scan_width_frac)
    scan_left = (width - scan_w) // 2
    scan_right = scan_left + scan_w
    width_mask = np.zeros_like(mask)
    width_mask[:, scan_left:scan_right] = mask[:, scan_left:scan_right]
    mask = width_mask

    scan_top = int(height * (1 - scan_height_frac))
    strip_h = (height - scan_top) // num_strips

    overlay = frame.copy()
    cv2.line(overlay, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
    cv2.line(overlay, (scan_left, 0), (scan_left, height), (255, 0, 255), 1)
    cv2.line(overlay, (scan_right, 0), (scan_right, height), (255, 0, 255), 1)

    results = []
    for i in range(num_strips):
        strip_bottom = height - i * strip_h
        strip_top = max(height - (i + 1) * strip_h, scan_top)
        centroid = find_centroid_in_strip(mask[strip_top:strip_bottom, :])
        cv2.rectangle(overlay, (0, strip_top), (width, strip_bottom), (255, 0, 0), 1)
        if centroid is not None:
            cx, cy_local = centroid
            cy_global = strip_top + cy_local
            results.append((cx, cy_global, num_strips - i))
            cv2.circle(overlay, (cx, cy_global), 6, (0, 0, 255), -1)

    return results, mask, overlay


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def turn(got, steering, forward_speed):
    if steering < 0:
        got.transform_move_turn(0, int(forward_speed), DIR_LEFT, int(-steering))
    else:
        got.transform_move_turn(0, int(forward_speed), DIR_RIGHT, int(steering))


def stop(got):
    got.transform_stop()


def search_for_line(got):
    # got.transform_move_speed(0, 3)
    got.transform_stop()
    print("[search_for_line] line lost, creeping forward...")


# ---------------------------------------------------------------------------
# Control math
# ---------------------------------------------------------------------------


def compute_steering_error(results, frame_width):
    if not results:
        return None, None
    center_x = frame_width // 2
    total_weight = sum(w for _, _, w in results)
    weighted_error = sum((cx - center_x) * w for cx, _, w in results) / total_weight
    curvature = results[-1][0] - results[0][0]
    return weighted_error, curvature


def pd_steering(error, curvature, kp, kd):
    return kp * error + kd * curvature


def speed_for_steering(steering, max_speed, min_speed, steering_at_min_speed):
    turn_fraction = min(abs(steering) / steering_at_min_speed, 1.0)
    return max_speed - turn_fraction * (max_speed - min_speed)


def connect_robot(ip=ROBOT_IP):
    got = ugot.UGOT()
    got.initialize(ip)
    got.open_camera()
    got.transform_adaption_control(False)
    got.transform_set_chassis_height(CHASSIS_HEIGHT)  # once, never changed again
    got.mechanical_clamp_release()
    drive_pose(got)
    got.load_models(["gesture"])
    return got


print("Definitions loaded.")

# ---------------------------------------------------------------------------
# Main loop — run this cell to start the robot
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    got = connect_robot()
    detector = load_detector()

    # while True:
    #     gesture = got.get_gesture_result()
    #     if gesture == "Stone":
    #         break

    lost_line_count = 0
    smoothed_steering = 0.0
    smoothed_speed = float(cfg.max_speed)
    frame_count = 0
    cubes_collected = 0
    pickup_cooldown_until = 0.0

    try:
        while True:
            frame = got.read_camera_data()
            if not frame:
                print("Failed to grab frame")
                break
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if data is None:
                continue
            frame_count += 1

            # -----------------------------------------------------------
            # Detection: look for a RIPE (small) cube. Unripe (large)
            # cubes are never acted on — the robot just drives over them.
            # -----------------------------------------------------------
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                yolo_results = detector(data, verbose=False)
                ripe_box = find_ripe_detection(yolo_results)

                if DEBUG_DETECTIONS:
                    log_detections(yolo_results)

                if SHOW_DETECT:
                    cv2.imshow("YOLO Detection", draw_detections(data.copy(), yolo_results))

                if ripe_box is not None and time.time() >= pickup_cooldown_until:
                    x1, y1, x2, y2, conf = ripe_box
                    if y2 > RIPE_Y_THRESHOLD:
                        print(f"[ripe] cube at ({x1},{y1})-({x2},{y2}) conf={conf:.2f} -- stopping to pick up.")
                        stop(got)
                        pickup_cube(got)
                        cubes_collected += 1
                        print(f"[ripe] collected so far: {cubes_collected}")
                        pickup_cooldown_until = time.time() + PICKUP_COOLDOWN_S
                        # Reset smoothing so line-following resumes cleanly
                        smoothed_steering = 0.0
                        smoothed_speed = float(cfg.max_speed)
                        lost_line_count = 0
                        if (SHOW_MASK or SHOW_DETECT) and cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

            blurred = preprocess_frame(data)
            results, mask, overlay = get_line_position_multistrip(data, blurred)

            # ---- End of course: big solid white box fills the frame ----
            box_found, box_bbox, box_mask = detect_stop_box(
                data, blurred,
                min_area_frac=cfg.stop_box_min_area_frac,
                min_extent=cfg.stop_box_min_extent,
            )
            if box_found:
                x, y, w, h = box_bbox
                fill = (w * h) / (data.shape[0] * data.shape[1])
                print(f"[stop_box] end detected ({fill*100:.0f}% of frame) -- stopping. Collected {cubes_collected} cubes.")
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
                stop(got)
                got.transform_move_speed_times(0, 20, 30, 1) # final nudge into the box
                stop(got)
                if SHOW_DETECT:
                    cv2.imshow("Webcam Feed", overlay)
                if SHOW_MASK:
                    cv2.imshow("Mask", box_mask)
                cv2.waitKey(1)
                break

            # ---- PD line following ----
            error, curvature = compute_steering_error(results, data.shape[1])

            if error is not None:
                lost_line_count = 0
                if abs(error) < cfg.error_deadband:
                    error = 0.0
                raw_steering = pd_steering(error, curvature, cfg.kp, cfg.kd)
                raw_speed = speed_for_steering(raw_steering, cfg.max_speed,
                                               cfg.min_speed, cfg.steering_at_min_speed)
                target_steering = cfg.smoothing_alpha * raw_steering + (1 - cfg.smoothing_alpha) * smoothed_steering
                target_speed = cfg.smoothing_alpha * raw_speed + (1 - cfg.smoothing_alpha) * smoothed_speed
                smoothed_steering += max(-cfg.max_steering_delta,
                                         min(cfg.max_steering_delta, target_steering - smoothed_steering))
                smoothed_speed += max(-cfg.max_speed_delta,
                                      min(cfg.max_speed_delta, target_speed - smoothed_speed))
                # Hard clamp before sending to hardware
                smoothed_steering = max(-cfg.max_steering, min(cfg.max_steering, smoothed_steering))
                # print(f"error={error:.1f}, steering={smoothed_steering:.1f}, "
                #       f"speed={smoothed_speed:.1f}, strips={len(results)}")
                turn(got, smoothed_steering, smoothed_speed)
            else:
                lost_line_count += 1
                print(f"Line not found ({lost_line_count} frames)")
                if lost_line_count >= cfg.lost_line_threshold:
                    stop(got)
                    search_for_line(got)
                    smoothed_steering = 0.0
                    smoothed_speed = float(cfg.max_speed)

            if SHOW_MASK or SHOW_DETECT:
                if SHOW_DETECT:
                    cv2.imshow("Webcam Feed", overlay)
                if SHOW_MASK:
                    cv2.imshow("Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop(got)
                    break
    finally:
        stop(got)
        if SHOW_MASK or SHOW_DETECT:
            cv2.destroyAllWindows()
main()