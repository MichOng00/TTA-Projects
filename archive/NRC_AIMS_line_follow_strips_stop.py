import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.217")
got.open_camera()


def detect_red_dot(frame, scan_left, scan_right, strip_top, strip_bottom, min_area=80):
    """
    Looks for a red blob within the given region (intended to be the same
    width band + bottom strip used by the line-following scan).
    Returns (cx, cy_in_frame) if a red dot of sufficient area is found,
    otherwise None.
    """
    region = frame[strip_top:strip_bottom, scan_left:scan_right]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Red wraps around hue 0/180 in OpenCV's HSV, so we need two ranges.
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx_local = int(M["m10"] / M["m00"])
    cy_local = int(M["m01"] / M["m00"])

    # Convert back to full-frame coordinates
    cx = cx_local + scan_left
    cy = cy_local + strip_top
    return (cx, cy)


def find_centroid_in_strip(mask_strip, min_area=80):
    """Find centroid of largest white blob in a single strip mask."""
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
    frame, threshold=180, num_strips=3, scan_height_frac=0.4, scan_width_frac=0.75
):
    """
    Slices the bottom scan_height_frac of the frame into num_strips horizontal
    bands, and restricts the search to the center scan_width_frac of the
    frame's width (1.0 = full width, 0.5 = center half, etc).
    Returns a list of (cx, cy_in_frame, weight) for strips where the
    line was found, ordered closest-to-robot first, plus a debug overlay,
    plus a (scan_left, scan_right, strip_top, strip_bottom) tuple describing
    the bottom strip's region in full-frame coordinates (handy for other
    detectors, e.g. a red-dot stop signal, that should look in the same
    spot).
    """
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Restrict to a centered vertical band of the given width fraction
    scan_w = int(width * scan_width_frac)
    scan_left = (width - scan_w) // 2
    scan_right = scan_left + scan_w

    # Zero out everything outside the width band so it's excluded from
    # contour detection in every strip
    width_mask = np.zeros_like(mask)
    width_mask[:, scan_left:scan_right] = mask[:, scan_left:scan_right]
    mask = width_mask

    scan_top = int(height * (1 - scan_height_frac))
    scan_zone_height = height - scan_top
    strip_h = scan_zone_height // num_strips

    overlay = frame.copy()
    cv2.line(overlay, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
    # Draw width band boundaries for visualization
    cv2.line(overlay, (scan_left, 0), (scan_left, height), (255, 0, 255), 1)
    cv2.line(overlay, (scan_right, 0), (scan_right, height), (255, 0, 255), 1)

    results = []  # closest strip (bottom) first
    bottom_strip_bounds = None
    for i in range(num_strips):
        # i=0 is the strip closest to the bottom (closest to robot)
        strip_bottom = height - i * strip_h
        strip_top = height - (i + 1) * strip_h
        strip_top = max(strip_top, scan_top)

        if i == 0:
            # Bounds of the strip closest to the robot, for reuse by
            # other detectors (e.g. red-dot stop signal) that should
            # scan the same region.
            bottom_strip_bounds = (scan_left, scan_right, strip_top, strip_bottom)

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

    return results, mask, overlay, bottom_strip_bounds


# ---------------------------------------------------------------------------
# Movement functions
# ---------------------------------------------------------------------------


def move_forward(speed):
    """Drive straight ahead at the given speed."""
    got.mecanum_move_speed(0, int(speed))
    print(f"[move_forward] speed={speed:.1f}")


def turn(steering, forward_speed):
    """
    Steer left/right while moving.
    steering < 0 -> turn left, steering > 0 -> turn right.
    Magnitude indicates how sharply to turn.
    forward_speed should match whatever speed move_forward() is using,
    so the robot doesn't slow down/speed up just because it's turning.
    """
    got.mecanum_move_xyz(x_speed=0, y_speed=int(forward_speed), z_speed=int(-steering))
    print(f"[turn] steering={steering:.1f}, forward_speed={forward_speed}")


def stop():
    """Stop all motion."""
    got.mecanum_stop()
    print("[stop]")


def search_for_line():
    """
    Called when the line isn't visible in any strip.
    Placeholder for a recovery behavior (e.g. slow rotate, back up, etc).
    """
    # TODO: replace with your actual search/recovery behavior
    print("[search_for_line] line lost, searching...")


def compute_steering_error(results, frame_width):
    """
    Combines multi-strip centroids into a single weighted steering error
    and an estimate of curvature (difference between near and far strips).
    """
    if not results:
        return None, None

    center_x = frame_width // 2

    # Weighted average error across all strips (overall position)
    total_weight = sum(w for _, _, w in results)
    weighted_error = sum((cx - center_x) * w for cx, _, w in results) / total_weight

    # Curvature signal: compare nearest strip vs farthest strip found
    near_cx = results[0][0]
    far_cx = results[-1][0]
    curvature = far_cx - near_cx  # positive = line curving right ahead

    return weighted_error, curvature


def pd_steering(error, curvature, kp=0.5, kd=0.15):
    """
    Combines current error and curvature (a stand-in for "future" error)
    into a single steering value using a P + D-like term.

    kp: how aggressively to correct current offset from center
    kd: how aggressively to anticipate upcoming turns based on curvature
    Tune these once movement is wired in -- start small and increase.
    """
    steering = kp * error + kd * curvature
    return steering


def speed_for_steering(steering, max_speed, min_speed, max_steering_for_slowdown):
    """
    Scales speed down as steering magnitude increases, so the robot moves
    slower on sharp turns and faster on straights.

    max_speed: speed used when steering is ~0 (straight line)
    min_speed: speed floor used at or beyond max_steering_for_slowdown
    max_steering_for_slowdown: |steering| value at which speed bottoms out
        at min_speed. Anything beyond this is clamped to min_speed.
    """
    turn_fraction = min(abs(steering) / max_steering_for_slowdown, 1.0)
    speed = max_speed - turn_fraction * (max_speed - min_speed)
    return speed


def main():
    max_speed = 30  # speed on straights (steering near 0)
    min_speed = 12  # speed floor on sharp turns
    max_steering_for_slowdown = (
        40  # |steering| at which speed bottoms out -- tune to your track
    )
    lost_line_count = 0
    lost_line_threshold = 5  # frames with no line before triggering search

    while True:
        frame = got.read_camera_data()
        if not frame:
            print("Failed to grab frame")
            break

        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results, mask, overlay, bottom_strip_bounds = get_line_position_multistrip(data)

        if bottom_strip_bounds is not None:
            scan_left, scan_right, strip_top, strip_bottom = bottom_strip_bounds
            red_dot = detect_red_dot(data, scan_left, scan_right, strip_top, strip_bottom)
            if red_dot is not None:
                cx, cy = red_dot
                print(f"[red_dot] detected at ({cx}, {cy}) in bottom strip -- stopping.")
                cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), 2)
                stop()
                cv2.imshow("Webcam Feed", overlay)
                cv2.imshow("Mask", mask)
                cv2.waitKey(1)
                break

        error, curvature = compute_steering_error(results, data.shape[1])

        if error is not None:
            lost_line_count = 0
            steering = pd_steering(error, curvature)
            speed = speed_for_steering(
                steering, max_speed, min_speed, max_steering_for_slowdown
            )
            print(
                f"error={error:.1f}, curvature={curvature:.1f}, "
                f"steering={steering:.1f}, speed={speed:.1f}, strips_found={len(results)}"
            )

            move_forward(speed)
            turn(steering, speed)
        else:
            lost_line_count += 1
            print(f"Line not found in any strip ({lost_line_count} frames)")

            if lost_line_count >= lost_line_threshold:
                stop()
                search_for_line()

        cv2.imshow("Webcam Feed", overlay)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()