import cv2


def get_camera(height, width, num=0):
    cap = cv2.VideoCapture(num)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
