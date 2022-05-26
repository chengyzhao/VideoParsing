import os
import os.path as osp
import time
import cv2
import torch
import yaml

from loguru import logger

from yolox.utils import fuse_model, get_model_info
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from exps.yolox_x_mix_det import Exp

from tools.demo_track import Predictor

CONFIG_PATH = "./config.yaml"


class Defaults(object):
    def __init__(self):
        self.fps = 30
        self.fp16 = True
        self.fuse = True
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False


def video_inference(predictor, output_folder, save_name, video_path, exp, args):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_folder = osp.join(output_folder, save_name)
    os.makedirs(save_folder, exist_ok=True)

    save_path = osp.join(save_folder, f"{save_name}.mp4")
    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'],
                                          online_tlwhs,
                                          online_ids,
                                          frame_id=frame_id + 1,
                                          fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']

            vid_writer.write(online_im)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    res_file = osp.join(save_folder, f"{save_name}.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


class Tracker(object):
    def __init__(self, device, pretrain_path, args):
        self.args = args
        self.exp = Exp()
        self.device = device
        self.model = self._get_model(pretrain_path, args)
        self.predictor = self._get_predictor(args)

    def _get_model(self, pretrain_path, args):
        model = self.exp.get_model().to(self.device)
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.eval()

        ckpt_file = pretrain_path
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()

        return model

    def _get_predictor(self, args, trt_file=None, decoder=None):
        predictor = Predictor(self.model, self.exp, trt_file, decoder, self.device, args.fp16)

        return predictor

    def inference(self, video_path, save_path, save_name):
        current_time = time.localtime()
        video_inference(self.predictor, save_path, save_name, video_path, self.exp, self.args)


if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as fd:
        cfg = yaml.safe_load(fd)
    cfg['device'] = torch.device("cuda" if cfg['device'] == "gpu" else "cpu")
    logger.info("Configs: {}".format(cfg))

    output_dir = osp.join(cfg['save_path'], cfg['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    args = Defaults()
    tracker = Tracker(cfg['device'], cfg['pretrain_path'], args)

    tracker.inference(cfg['video_path'], cfg['save_path'], cfg['exp_name'])
