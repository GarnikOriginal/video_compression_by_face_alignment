import cv2


def add_fps(frame, fps):
    cv2.putText(frame, f"{fps:0.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
