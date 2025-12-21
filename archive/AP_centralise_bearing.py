from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.164")
got.load_models(["apriltag_qrcode"])
got.open_camera()

import time
while True:
    AP_info = got.get_apriltag_total_info()
    if AP_info:
        centre_x = AP_info[0][1]
        bearing_h = AP_info[0][10]
        print(f"centre x:{centre_x}, bearing x:{bearing_h}")
        if centre_x < 300:
                got.mecanum_move_xyz(-5, 0, 0)
        elif centre_x > 340:
            got.mecanum_move_xyz(5, 0, 0)
        else:
            if bearing_h < 0:
                got.mecanum_move_xyz(5, 0, 5)
                time.sleep(0.1)
            elif bearing_h > 0:
                got.mecanum_move_xyz(-5, 0, -5)
                time.sleep(0.1)
            else:
                got.mecanum_stop()
                break
print("Apriltag is centralized squarely.")
