import os
from time import sleep

import cv2
import yaml
import torch
import numpy as np
from os.path import join
from modules._3DDFA_V2.utils.io import _load
from modules._3DDFA_V2.utils.tddfa_util import _to_ctype
from modules._3DDFA_V2.utils.uv import process_uv, load_uv_coords, bilinear_interpolate
from modules._3DDFA_V2.Sim3DR import rasterize
from modules._3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from modules._3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX
from modules._3DDFA_V2.FaceBoxes.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA import TDDFA
from modules.utils.camera import get_camera
from modules.utils.fps_counter import FPSCounter
from modules.utils.frame_utils import add_fps

h, w = 480, 640
z_h, z_w = 48 * 2, 64 * 2
# h, w = 1080, 1920
# z_h, z_w = 108, 192
ONNX_MODE = True
# TDDFA_CONFIG_PATH = "configs/tddfa_onnx_config.yml"
TDDFA_CONFIG_PATH = "configs/mb05_120x120.yml"

if __name__ == '__main__':
    cv2.namedWindow('Reconstruction')
    cap = get_camera(h, w, num=0)
    try:
        fps_counter = FPSCounter(30)

        tddfa_cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
        if ONNX_MODE:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '12'
            tddfa = TDDFA_ONNX(**tddfa_cfg)
            face_boxes = FaceBoxes_ONNX()
        else:
            tddfa = TDDFA(**tddfa_cfg, gpu_mode=False)
            face_boxes = FaceBoxes()

        frame_ind = 0
        with torch.no_grad():
            fps_counter.run()
            while True:
                ret, frame = cap.read()
                boxes = face_boxes(frame)
                background = cv2.resize(frame, (z_w, z_h), interpolation=cv2.INTER_AREA)
                if len(boxes) != 0:
                    # Sender
                    param_lst, roi_box_lst = tddfa(frame, [boxes[0]], crop_policy="box")
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=True)[0]
                    ver = _to_ctype(ver.T)
                    colors = bilinear_interpolate(frame, ver[:, 0], ver[:, 1]) / 255.

                    # Receiver
                    frame = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
                    frame = cv2.blur(frame, (7, 7))
                    frame = rasterize(ver, tddfa.tri, colors, bg=frame, height=h, width=w, channel=3)
                else:
                    frame = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
                    frame = cv2.blur(frame, (7, 7))

                fps = fps_counter.get_fps()
                fps_counter.step()
                add_fps(frame, fps)
                cv2.imshow("Reconstruction", frame)

                c = cv2.waitKey(1)
                if c == 27:
                    break
    except Exception as error:
        raise error
    finally:
        cv2.destroyAllWindows()
        cap.release()
