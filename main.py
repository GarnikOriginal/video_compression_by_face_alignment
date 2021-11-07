import os
import cv2
import torch
import numpy as np
from modules.fps_counter import FPSCounter
from modules.frame_utils import add_fps


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(0)
    fps_counter = FPSCounter(10)

    cv2.namedWindow('main')

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    fps_counter.run()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            # frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            fps = fps_counter.get_fps()
            add_fps(frame, fps)
            cv2.imshow("main", frame)
            fps_counter.step()

            c = cv2.waitKey(1)
            if c == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
