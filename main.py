import os
from time import sleep

import cv2
import ffmpeg
import yaml
import torch
from datetime import datetime
import numpy as np
from subprocess import Popen, PIPE, DEVNULL
from os.path import join
from modules._3DDFA_V2.utils.io import _load
from modules._3DDFA_V2.utils.tddfa_util import _to_ctype
from modules._3DDFA_V2.utils.uv import process_uv, load_uv_coords, bilinear_interpolate
from modules._3DDFA_V2.Sim3DR import rasterize


from modules.utils.model import load_model
from modules.utils.camera import get_camera
from modules.utils.fps_counter import FPSCounter
from modules.utils.frame_utils import add_fps


h, w = 480, 640             # h, w = 1080, 1920
z_h, z_w = 48 * 2, 64 * 2   # z_h, z_w = 108, 192
yuv_height = int(h + h // 2)

ONNX_MODE = False
# TDDFA_CONFIG_PATH = "configs/tddfa_onnx_config.yml"
TDDFA_CONFIG_PATH = "configs/mb05_120x120.yml"


if __name__ == '__main__':
    tddfa, faceboxes = load_model(TDDFA_CONFIG_PATH, ONNX_MODE)
    cv2.namedWindow('Reconstruction')
    fps_counter = FPSCounter(30)

    
    """
    stream_url = f'ffmpeg -f v4l2 -s {w}x{h} -i /dev/video0 ' \
                 f'-f rawvideo -pix_fmt yuv420p -y pipe:'.split(' ')
    with torch.no_grad():
        with Popen(stream_url, stdout=PIPE) as p:
            fps_counter.run()
            while p.stdout.readable():
                raw_frame = p.stdout.read(yuv_height * w)
                if len(raw_frame) == yuv_height * w:
                    frame = np.frombuffer(raw_frame, np.uint8).reshape((yuv_height, w))
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV420P2RGB)
                    boxes = faceboxes(frame)
                    background = cv2.resize(frame, (z_w, z_h), interpolation=cv2.INTER_AREA)
                    if boxes:
                        param_lst, roi_box_lst = tddfa(frame, [boxes[0]], crop_policy="box")
                        ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=True)[0]
                        ver = _to_ctype(ver.T)
                        colors = bilinear_interpolate(frame, ver[:, 0], ver[:, 1]) / 255.

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
                else:
                    print("Read broken frame", f"Len - {len(raw_frame)}")
                    break

            cv2.destroyAllWindows()
    """
