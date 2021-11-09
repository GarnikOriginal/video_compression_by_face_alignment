import cv2
from modules._3DDFA_V2.utils.render import render
from modules._3DDFA_V2.utils.functions import cv_draw_landmark


def add_fps(frame, fps):
    cv2.putText(frame, f"{fps:0.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)


def draw_mask(frame, vert, tddfa=None, flag="2d_sparse"):
    if flag == "2d_sparse":
        return cv_draw_landmark(frame, vert)
    elif flag == "2d_dense":
        return cv_draw_landmark(frame, vert, size=1)
    elif flag == "3d":
        return render(frame, [vert], tddfa.tri, alpha=0.7)
    else:
        raise NotImplementedError(f"Flag {flag} is not implemented")
