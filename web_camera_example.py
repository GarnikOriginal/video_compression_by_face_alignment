import cv2
import yaml
import torch
from subprocess import Popen, PIPE
from tqdm import tqdm
from modules._3DDFA_V2.FaceBoxes.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA import TDDFA
from modules.utils.camera import get_camera
from modules.utils.fps_counter import FPSCounter
from modules.utils.frame_utils import add_fps, draw_mask
from modules._3DDFA_V2.utils.tddfa_util import _to_ctype
from modules._3DDFA_V2.utils.uv import bilinear_interpolate
from modules._3DDFA_V2.Sim3DR import rasterize


TDDFA_CONFIG_PATH = "configs/tddfa_onnx_config.yml"

h, w = 480, 640
z_h, z_w = 240, 320
out = "web_camera_sacle2"

if __name__ == '__main__':
    cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    faceboxes = FaceBoxes()
    ffmpeg_output = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{640}x{480}', '-pix_fmt', 'rgb24',
                     '-r', f'{25}', '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', f'{out}.mp4']

    cap = get_camera(640, 480)
    fps_counter = FPSCounter(30)

    out = Popen(ffmpeg_output, stdin=PIPE, stderr=PIPE)
    with torch.no_grad():
        fps_counter.run()
        for i in tqdm(range(300)):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = faceboxes(frame)
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

            out.stdin.write(frame.tobytes())
            c = cv2.waitKey(1)
            if c == 27:
                break
    out.terminate()
    cap.release()
