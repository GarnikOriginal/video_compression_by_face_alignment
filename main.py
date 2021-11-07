import os
import cv2
import yaml
import torch
import logging
from modules.utils.camera import get_camera
from modules.fps_counter import FPSCounter
from modules.frame_utils import add_fps
from modules._3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from modules._3DDFA_V2.FaceBoxes.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX
from modules._3DDFA_V2.TDDFA import TDDFA
from modules._3DDFA_V2.utils.render import render
from modules._3DDFA_V2.utils.functions import cv_draw_landmark


TDDFA_CONFIG_PATH = "configs/tddfa_onnx_config.yml"
DENSE_FLAG = "2d_sparse"


if __name__ == '__main__':
    tddfa_cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)

    cap = get_camera(1920, 1080)

    fps_counter = FPSCounter(10)
    cv2.namedWindow('main')

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    # face_boxes = FaceBoxes_ONNX()
    face_boxes = FaceBoxes()
    # tddfa = TDDFA_ONNX(**tddfa_cfg)
    tddfa = TDDFA(**tddfa_cfg, gpu_mode=False)
    fps_counter.run()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            # frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            # (1080, 1920, 3)

            boxes = face_boxes(frame)
            if len(boxes) != 0:
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=DENSE_FLAG)[0]
                param_lst, roi_box_lst = tddfa(frame, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=DENSE_FLAG)[0]
                img_draw = cv_draw_landmark(frame, ver)
                # img_draw = render(frame, [ver], tddfa.tri, alpha=0.7)
            else:
                img_draw = frame

            fps = fps_counter.get_fps()
            add_fps(img_draw, fps)
            cv2.imshow("main", img_draw)
            fps_counter.step()

            c = cv2.waitKey(1)
            if c == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
