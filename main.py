import os
import cv2
import yaml
import torch
from modules.utils.camera import get_camera
from modules.utils.fps_counter import FPSCounter
from modules.utils.frame_utils import add_fps, draw_mask
from modules._3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from modules._3DDFA_V2.TDDFA_ONNX import TDDFA_ONNX
from modules._3DDFA_V2.FaceBoxes.FaceBoxes import FaceBoxes
from modules._3DDFA_V2.TDDFA import TDDFA


ONNX_MODE = False
TDDFA_CONFIG_PATH = "configs/tddfa_onnx_config.yml"
DENSE_FLAG = "2d_sparse"  # ['2d_sparse', '2d_dense', '3d']


if __name__ == '__main__':
    cv2.namedWindow('main')
    cap = get_camera(640, 480, num=0)
    try:
        fps_counter = FPSCounter(30)

        tddfa_cfg = yaml.load(open(TDDFA_CONFIG_PATH), Loader=yaml.SafeLoader)
        if ONNX_MODE:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            tddfa = TDDFA_ONNX(**tddfa_cfg)
            face_boxes = FaceBoxes_ONNX()
        else:
            tddfa = TDDFA(**tddfa_cfg, gpu_mode=False)
            face_boxes = FaceBoxes()

        with torch.no_grad():
            fps_counter.run()
            while True:
                ret, frame = cap.read()

                boxes = face_boxes(frame)
                if len(boxes) != 0:
                    param_lst, roi_box_lst = tddfa(frame, [boxes[0]], crop_policy="box")
                    ver = tddfa.recon_vers(param_lst, roi_box_lst, DENSE_FLAG=DENSE_FLAG != "2d_sparse")[0]
                    frame = draw_mask(frame, ver, flag=DENSE_FLAG, tddfa=tddfa)

                fps = fps_counter.get_fps()
                fps_counter.step()
                add_fps(frame, fps)
                cv2.imshow("main", frame)

                c = cv2.waitKey(1)
                if c == 27:
                    break
    except Exception as error:
        raise error
    finally:
        cap.release()
        cv2.destroyAllWindows()
