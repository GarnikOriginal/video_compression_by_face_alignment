import cv2
import yaml
import torch
from subprocess import Popen, PIPE
from tqdm import tqdm
from modules._3DDFA_V2.utils.uv import uv_tex
from modules._3DDFA_V2.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA import TDDFA


w, h = 1920, 1080
scale = 2
fps = 30
out_file = "uv_visualization"

ONNX_MODE = False
TDDFA_CONFIG_PATH = "configs/mb05_120x120.yml"


def create_uv_visualization(out, zipper='libx264'):
    cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    faceboxes = FaceBoxes()
    ffmpeg_output = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{512}x{512}', '-pix_fmt', 'rgb24',
                     '-r', f'{fps}', '-i', '-', '-an', '-vcodec', zipper, '-pix_fmt', 'yuv420p', f'{out}.mp4']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video source")

    with torch.no_grad():
        uv_texture = None
        out = Popen(ffmpeg_output, stdin=PIPE, stderr=PIPE)
        for i in tqdm(range(300)):
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = faceboxes(frame)
            if len(boxes) != 0:
                param_lst, roi_box_lst = tddfa(frame, boxes, crop_policy="box")
                ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=True)
                uv_texture = uv_tex(frame, ver, tddfa.tri, uv_h=512, uv_w=512)
            if uv_texture is not None:
                out.stdin.write(uv_texture.tobytes())
        out.terminate()
    cap.release()


if __name__ == '__main__':
    create_uv_visualization(out_file, zipper="libx264")

