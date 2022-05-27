import os
import os.path as osp
import cv2
import moviepy.editor as mp
from pydub import AudioSegment


def video2images(video_path, outimages_path, zero_fill=8):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    isOpened = cap.isOpened()
    assert isOpened, "Can't find video"

    for index in range(video_length):
        (flag, data) = cap.read()
        file_name = "{}.jpg".format(str(index).zfill(zero_fill))  # start from zero
        file_path = osp.join(outimages_path, file_name)
        if flag:
            cv2.imwrite(file_path, data, [cv2.IMWRITE_JPEG_QUALITY, 100])

    return {'width': width, 'height': height, 'fps': fps}


def image2video(image_dir, save_path, save_name, fps=25):
    image_path_list = os.listdir(image_dir)
    image_path_list = [osp.join(image_dir, x) for x in image_path_list]
    image_path_list.sort()
    temp = cv2.imread(image_path_list[0])
    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(osp.join(save_path, f"{save_name}.mp4"), fourcc, fps, size)
    for image_path in image_path_list:
        if image_path.endswith(".jpg"):
            image_data_temp = cv2.imread(image_path)
            video.write(image_data_temp)
    print("Video done！")

    return osp.join(save_path, f"{save_name}.mp4")


def add_audio_to_video(src_audio_path, new_video_path, output_dir, save_name, segment, fps):
    inst_id, start_frame, end_frame, length = segment
    audio_start = start_frame / fps * 1000  # ms
    audio_end = (end_frame + 1) / fps * 1000  # ms
    sound = AudioSegment.from_wav(src_audio_path)
    sound[audio_start:audio_end].export(osp.join(output_dir, f"{save_name}.wav"), format="wav")

    video = mp.VideoFileClip(new_video_path)
    audio = mp.AudioFileClip(osp.join(output_dir, f"{save_name}.wav"))
    video = video.set_audio(audio)
    video.write_videofile(osp.join(output_dir, f"{save_name}_final.mp4"))

    return osp.join(output_dir, f"{save_name}_final.mp4")
