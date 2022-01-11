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
scale = 2
fps = 25
source = join("samples", "original", "misha_light.mp4")
out_file = join("samples", "results", "misha_light", "misha_light")

z_h, z_w = int(h / scale), int(w / scale)
ONNX_MODE = True
TDDFA_CONFIG_PATH = "configs/mb05_120x120.yml"


def process_video(file, out, zipper='libx264'):
    cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    faceboxes = FaceBoxes()
    ffmpeg_output = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'rgb24',
                     '-r', f'{fps}', '-i', '-', '-an', '-vcodec', zipper, '-pix_fmt', 'yuv420p', f'{out}.mp4']
    ffmpeg_zip_background = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{z_w}x{z_h}',
                             '-pix_fmt', 'rgb24', '-r', f'{fps}', '-i', '-', '-an', '-vcodec', zipper, '-pix_fmt', 'yuv420p', f'{out}_bg.mp4']
    ffmpeg_zip_colors = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'147x261',
                         '-pix_fmt', 'rgb24', '-r', f'{fps}', '-i', '-', '-an', '-vcodec', "libvpx", f'{out}_cs.webm']

    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise IOError("Cannot open video source")

    with torch.no_grad():
        out = Popen(ffmpeg_output, stdin=PIPE, stderr=PIPE)
        bg_out = Popen(ffmpeg_zip_background, stdin=PIPE, stderr=PIPE)
        cl_out = Popen(ffmpeg_zip_colors, stdin=PIPE, stderr=PIPE)
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = faceboxes(frame)
            background_zipped = cv2.resize(frame.copy(), (z_w, z_h), interpolation=cv2.INTER_AREA)
            background = cv2.resize(background_zipped, (w, h), interpolation=cv2.INTER_LINEAR)
            if len(boxes) != 0:
                param_lst, roi_box_lst = tddfa(frame, boxes, crop_policy="box")
                ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=True)

                colors = []
                for v in ver:
                    v = _to_ctype(v.T)
                    colors.append(bilinear_interpolate(frame, v[:, 0], v[:, 1]) / 255)

                for i, v in enumerate(ver):
                    background = rasterize(_to_ctype(v.T), tddfa.tri, colors[i],
                                           bg=background, height=h, width=w, channel=3)

                c = np.pad(colors[0], ((0, 2), (0, 0)), mode='constant', constant_values=0).reshape((147, 261, 3))
                c = (c * 255).astype(np.uint8)
                cl_out.stdin.write(c.tobytes())

            out.stdin.write(background.tobytes())
            bg_out.stdin.write(background_zipped.tobytes())

        out.terminate()
        bg_out.terminate()
        cl_out.terminate()


if __name__ == '__main__':
    process_video(source, out_file, zipper="libx264")

