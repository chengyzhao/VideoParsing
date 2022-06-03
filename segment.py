import os
import os.path as osp
import cv2
import torch
import yaml
import moviepy.editor as mp
from tqdm import tqdm
from loguru import logger

from tracking import Tracker
from utils import video2images, image2video, add_audio_to_video

CONFIG_PATH = "./config.yaml"


def get_frame_preds(output_txt_path):
    frame2inst = {}
    inst2bbox = {}  # bbox: (x, y, w, h, score)
    with open(output_txt_path, 'r') as fd:
        for line in fd:
            ls = line.split(',')
            frame_id = int(ls[0])
            inst_id = int(ls[1])
            xmin = float(ls[2])
            ymin = float(ls[3])
            width = float(ls[4])
            height = float(ls[5])
            score = float(ls[6])

            if frame_id not in frame2inst.keys():
                frame2inst[frame_id] = []
            if inst_id not in frame2inst[frame_id]:
                frame2inst[frame_id].append(inst_id)

            if inst_id not in inst2bbox.keys():
                inst2bbox[inst_id] = {}
            inst2bbox[inst_id][frame_id] = (xmin, ymin, width, height, score)
    return frame2inst, inst2bbox


def get_candidate_segments(frame2inst, min_frames=2000):
    inst2segment = {}
    for frame_id in sorted(frame2inst.keys()):
        for inst_id in frame2inst[frame_id]:
            if inst_id not in inst2segment.keys():
                inst2segment[inst_id] = []
                inst2segment[inst_id].append({'start_frame': frame_id, 'end_frame': frame_id})
            elif inst2segment[inst_id][-1]['end_frame'] == frame_id - 1:
                inst2segment[inst_id][-1]['end_frame'] = frame_id
            else:
                inst2segment[inst_id].append({'start_frame': frame_id, 'end_frame': frame_id})

    candidate_segments = []
    for inst_id in inst2segment:
        for segm in inst2segment[inst_id]:
            start = segm['start_frame']
            end = segm['end_frame']
            length = end - start + 1
            if length > min_frames:
                candidate_segments.append((inst_id, start, end, length))  # (inst_id, start_frame, end_frame, length)

    candidate_segments = sorted(candidate_segments, key=lambda x: x[-1], reverse=True)
    return candidate_segments


def get_output_box(src_box, img_height, img_width, output_width, output_height, add_cfg):
    assert output_height > output_width, f"Output height {output_height} should greater than output width {output_width}"

    xmin, ymin, width, height = src_box
    x_center = xmin + width / 2
    y_center = ymin + height / 2

    ratio = add_cfg['ratio']
    if (height / width) >= (output_height / output_width):
        new_height = height / ratio
        new_width = new_height / output_height * output_width

        new_height = min(min(y_center, img_height - y_center) * 2, new_height)
        new_width = min(min(x_center, img_width - x_center) * 2, new_width)
        if new_height / output_height * output_width > new_width:
            new_height = new_width / output_width * output_height
        else:
            new_width = new_height / output_height * output_width

        new_ymin = y_center - new_height / 2
        new_xmin = x_center - new_width / 2
        new_ymin = max(new_ymin, 0)
        new_xmin = max(new_xmin, 0)
    else:
        new_width = width / ratio
        new_height = new_width / output_width * output_height

        new_height = min(min(y_center, img_height - y_center) * 2, new_height)
        new_width = min(min(x_center, img_width - x_center) * 2, new_width)
        if new_height / output_height * output_width > new_width:
            new_height = new_width / output_width * output_height
        else:
            new_width = new_height / output_height * output_width

        new_xmin = x_center - new_width / 2
        new_ymin = y_center - new_height / 2
        new_ymin = max(new_ymin, 0)
        new_xmin = max(new_xmin, 0)

    return (new_xmin, new_ymin, new_width, new_height)

def compute_midx(xmin, width):
    return int(xmin+width/2)

def segment_video(src_audio_path, src_images_path, inst2bbox, segment, output_dir, exp_name, new_cfg, keep_original):
    inst_id, start_frame, end_frame, length = segment
    save_name = f"{exp_name}_{inst_id}_{start_frame}_{end_frame}"
    img_save_path = osp.join(output_dir, exp_name, save_name)
    os.makedirs(img_save_path, exist_ok=True)

    for frame_id in tqdm(range(start_frame, end_frame + 1)):
        frame_path = osp.join(src_images_path, "{}.jpg".format(str(frame_id).zfill(8)))
        frame = cv2.imread(frame_path)

        interp = []
        for i in range(10):
            if frame_id - i > start_frame: interp.append(frame_id-i)
            if frame_id + i < end_frame: interp.append(frame_id+i)
        
        # interp
        interp_candidate = [compute_midx(inst2bbox[inst_id][i][0], inst2bbox[inst_id][i][2]) for i in interp if i in inst2bbox[inst_id]]
        x = int(sum(interp_candidate)/len(interp_candidate))
        left = int(x-new_cfg['width']/2)
        if left < 0: left = 0

        new_img = cv2.resize(frame[0: new_cfg['height'],
                                    left:new_cfg['width']+left], (new_cfg['width'], new_cfg['height']),
                                interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(osp.join(img_save_path, "{}.jpg".format(str(frame_id).zfill(8))), new_img)

    new_video_path = image2video(img_save_path, osp.join(output_dir, exp_name), save_name, new_cfg['fps'])
    final_video_path = add_audio_to_video(src_audio_path, new_video_path, osp.join(output_dir, exp_name), save_name,
                                          segment, new_cfg['fps'])

    return final_video_path


if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as fd:
        cfg = yaml.safe_load(fd)
    cfg['device'] = torch.device("cuda" if cfg['device'] == "gpu" else "cpu")
    logger.info("Configs: {}".format(cfg))
    logger.info("aspect_ratio:", cfg['aspect_ratio'])
    output_dir = osp.join(cfg['save_path'], cfg['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    tracker = Tracker(cfg['device'], cfg['pretrain_path'])

    tracker.inference(cfg['video_path'], cfg['save_path'], cfg['exp_name'])

    pred_txt = osp.join(cfg['save_path'], cfg['exp_name'], f"{cfg['exp_name']}.txt")
    frame2inst, inst2bbox = get_frame_preds(pred_txt)

    logger.info('get candidate segments...')
    candidate_segments = get_candidate_segments(frame2inst, cfg['output']['min_frames'])
    logger.info('candidates done')
    src_images_path = osp.join(cfg['save_path'], cfg['exp_name'], "src_images")

    
    os.makedirs(src_images_path, exist_ok=True)

    original_cfg = video2images(cfg['video_path'], src_images_path)
    
    my_clip = mp.VideoFileClip(cfg['video_path'])
    cap = cv2.VideoCapture(cfg['video_path'])
    inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inputFrameSize = (inputWidth, inputHeight)
    outputFrameSize = (int(inputHeight * cfg['aspect_ratio']), inputHeight)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cfg['output']['width'], cfg['output']['height'] = outputFrameSize
    original_cfg = {'width': inputWidth, 'height': inputHeight, 'fps': fps}
    logger.info("width = {}, height = {}".format(cfg['output']['width'], cfg['output']['height']))

    src_audio_path = osp.join(cfg['save_path'], cfg['exp_name'], f"{cfg['exp_name']}.wav")
    my_clip.audio.write_audiofile(src_audio_path)

    output_cfg = cfg['output']
    keep_original = cfg['output']['keep_original']
    if keep_original:
        output_cfg['fps'] = original_cfg['fps']
    elif output_cfg['fps'] == -1:
        output_cfg['fps'] = original_cfg['fps']

    for idx, segment in enumerate(candidate_segments):
        final_video_path = segment_video(src_audio_path,
                                         src_images_path,
                                         inst2bbox,
                                         segment,
                                         cfg['save_path'],
                                         cfg['exp_name'],
                                         output_cfg,
                                         keep_original=keep_original)

        print(f"{segment} -> {final_video_path}")

    print('Over!')