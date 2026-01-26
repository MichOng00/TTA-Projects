from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.131")

import cv2
import numpy as np

got.load_models(["apriltag_qrcode"])
got.open_camera()

while True:
    if got.get_apriltag_total_info():
        print(got.get_apriltag_total_info()[0][12])