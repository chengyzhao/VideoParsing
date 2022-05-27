import os
import os.path as osp
import cv2
import torch
import yaml
import moviepy.editor as mp

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


# TODO parse instance ids, do same person - different instance id - matching
# TODO parse bboxs, if one inst id is missing in a few frames
def parsing(frame2inst, inst2bbox):
    pass

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


# TODO from tight bbox to output bbox
def get_output_box(src_box, output_width, output_height, add_cfg):
    assert output_height > output_width, f"Output height {output_height} should greater than output width {output_width}"

    xmin, ymin, width, height = src_box
    x_center = xmin + width / 2
    y_center = ymin + height / 2

    ratio = add_cfg['ratio']
    if (height / width) >= (output_height / output_width):
        new_height = height / ratio
        new_width = new_height / output_height * output_width
        new_ymin = y_center - new_height / 2
        new_xmin = x_center - new_width / 2
        new_ymin = max(new_ymin, 0)
        new_xmin = max(new_xmin, 0)
    else:
        new_width = width / ratio
        new_height = new_width / output_width * output_height
        new_xmin = x_center - new_width / 2
        new_ymin = y_center - new_height / 2
        new_ymin = max(new_ymin, 0)
        new_xmin = max(new_xmin, 0)

    return (new_xmin, new_ymin, new_width, new_height)


def segment_video(src_audio_path, src_images_path, inst2bbox, segment, output_dir, exp_name, new_cfg, keep_original):
    inst_id, start_frame, end_frame, length = segment
    save_name = f"{exp_name}_{inst_id}_{start_frame}_{end_frame}"
    img_save_path = osp.join(output_dir, exp_name, save_name)
    os.makedirs(img_save_path, exist_ok=True)

    for frame_id in range(start_frame, end_frame + 1):
        xmin, ymin, width, height, _ = inst2bbox[inst_id][frame_id]
        frame_path = osp.join(src_images_path, "{}.jpg".format(str(frame_id).zfill(8)))
        frame = cv2.imread(frame_path)
        output_bbox = get_output_box((xmin, ymin, width, height), new_cfg['width'], new_cfg['height'], new_cfg)
        xmin, ymin, width, height = output_bbox

        if keep_original:
            new_img = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmin + width), int(ymin + height)), (0, 0, 255),
                                    thickness=2)
        else:
            new_img = cv2.resize(frame[int(ymin):int(ymin + height),
                                       int(xmin):int(xmin + width)], (new_cfg['width'], new_cfg['height']),
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

    output_dir = osp.join(cfg['save_path'], cfg['exp_name'])
    os.makedirs(output_dir, exist_ok=True)

    # tracker = Tracker(cfg['device'], cfg['pretrain_path'])

    # tracker.inference(cfg['video_path'], cfg['save_path'], cfg['exp_name'])

    pred_txt = osp.join(cfg['save_path'], cfg['exp_name'], f"{cfg['exp_name']}.txt")
    frame2inst, inst2bbox = get_frame_preds(pred_txt)

    frame2inst, inst2bbox = parsing(frame2inst, inst2bbox)

    candidate_segments = get_candidate_segments(frame2inst, cfg['output']['min_frames'])

    src_images_path = osp.join(cfg['save_path'], cfg['exp_name'], "src_images")

    os.makedirs(src_images_path, exist_ok=True)
    original_cfg = video2images(cfg['video_path'], src_images_path)

    my_clip = mp.VideoFileClip(cfg['video_path'])
    src_audio_path = osp.join(cfg['save_path'], cfg['exp_name'], f"{cfg['exp_name']}.wav")
    my_clip.audio.write_audiofile(src_audio_path)

    output_cfg = cfg['output']
    keep_original = cfg['output']['keep_original']
    if keep_original:
        output_cfg['fps'] = original_cfg['fps']
    elif output_cfg['fps'] == -1:
        output_cfg['fps'] = original_cfg['fps']

    for segment in candidate_segments:
        final_video_path = segment_video(src_audio_path,
                                         src_images_path,
                                         inst2bbox,
                                         segment,
                                         cfg['save_path'],
                                         cfg['exp_name'],
                                         output_cfg,
                                         keep_original=keep_original)

        print(f"{segment} -> {final_video_path}")
        break

    print('Over!')
