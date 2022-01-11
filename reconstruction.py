import cv2
import yaml
import torch
import numpy as np
from os.path import join
from subprocess import Popen, PIPE
from modules._3DDFA_V2.utils.tddfa_util import _to_ctype
from modules._3DDFA_V2.utils.uv import bilinear_interpolate
from modules._3DDFA_V2.Sim3DR import rasterize
from modules._3DDFA_V2.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA import TDDFA


# 38367 = 3 * 3 * 3 * 7 * 7 * 29
w, h = 1280, 720
scale = 4
fps = 15
source = join("samples", "original", "ira_desktop.mp4")
bg = join("samples", "results", "ira_desktop", "ira_desktop_bg.mp4")
mask = join("samples", "results", "ira_desktop", "ira_desktop_cs.webm")

z_h, z_w = int(h / scale), int(w / scale)
ONNX_MODE = True
TDDFA_CONFIG_PATH = "configs/mb05_120x120.yml"


def main():
    cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    faceboxes = FaceBoxes()

    orig = cv2.VideoCapture(source)
    background = cv2.VideoCapture(bg)
    colors = cv2.VideoCapture(mask)

    out = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'rgb24',
           '-r', f'{fps}', '-i', '-', '-an', '-vcodec', "libx264", '-pix_fmt', 'yuv420p',
           join("reconst", "ira_desktop.mp4")]

    with torch.no_grad():
        out = Popen(out, stdin=PIPE, stderr=PIPE)
        i = 0
        while True:
            print(f"Frame {i}")
            ret, frame = orig.read()
            ret, frame_bg = background.read()
            ret, frame_cs = colors.read()
            if frame is None or frame_bg is None or colors is None or i > 30:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
            frame_cs = cv2.cvtColor(frame_cs, cv2.COLOR_BGR2RGB)

            boxes = faceboxes(frame)
            frame_bg = cv2.resize(frame_bg, (w, h), interpolation=cv2.INTER_LINEAR)
            res_colors = np.zeros((38365, 3))

            frame_cs = np.reshape(frame_cs, (-1, 3))[0:38365, :] / 255

            param_lst, roi_box_lst = tddfa(frame, boxes, crop_policy="box")
            ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=True)
            frame_bg = rasterize(_to_ctype(ver[0].T), tddfa.tri, frame_cs, bg=frame_bg, height=h, width=w, channel=3)

            out.stdin.write(frame_bg.tobytes())
            i += 1
    out.terminate()


if __name__ == '__main__':
    main()
