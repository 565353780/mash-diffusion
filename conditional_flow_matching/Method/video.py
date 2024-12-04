import os
import cv2
from tqdm import tqdm

from conditional_flow_matching.Method.path import createFileFolder, removeFile

def sampleImagesToVideo(
    iter_root_folder_path: str,
    rel_image_file_path: str,
    save_video_file_path: str,
    video_width: int = 540,
    video_height: int = 540,
    video_fps: int = 24,
    overwrite: bool = False
) -> bool:
    if os.path.exists(save_video_file_path):
        if not overwrite:
            return True

        removeFile(save_video_file_path)

    iter_folder_name_list = os.listdir(iter_root_folder_path)
    iter_folder_name_list.sort(key=lambda x: int(x.split('_')[1]))

    createFileFolder(save_video_file_path)

    video_size = (video_width, video_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_video_file_path, fourcc, video_fps, video_size)

    print('[INFO][sampe_to_video::sampleImagesToVideo]')
    print('\t start convert sample images to video...')
    print('\t', rel_image_file_path)
    print('\t -->')
    print('\t', save_video_file_path)

    for iter_folder_name in tqdm(iter_folder_name_list):
        iter_image_file_path = iter_root_folder_path + iter_folder_name + '/' + rel_image_file_path

        if not os.path.exists(iter_image_file_path):
            continue

        frame = cv2.imread(iter_image_file_path)

        if (frame.shape[1], frame.shape[0]) != video_size:
            frame = cv2.resize(frame, video_size)

        video.write(frame)

    video.release()

    return True

def sampleImagesFolderToVideos(
    iter_root_folder_path: str,
    save_video_folder_path: str,
    video_width: int = 540,
    video_height: int = 540,
    video_fps: int = 24,
    overwrite: bool = False
) -> bool:
    iter_folder_name_list = os.listdir(iter_root_folder_path)

    rel_image_file_path_set = set()

    for iter_folder_name in iter_folder_name_list:
        iter_folder_path = iter_root_folder_path + iter_folder_name + '/'

        if not os.path.exists(iter_folder_path):
            continue

        for root, _, files in os.walk(iter_folder_path):
            for file in files:
                if file[-4:] not in ['.jpg', '.png']:
                    continue

                rel_image_file_path = os.path.relpath(root, iter_folder_path) + '/' + file

                rel_image_file_path_set.add(rel_image_file_path)

    rel_image_file_path_list = list(rel_image_file_path_set)
    rel_image_file_path_list.sort()

    for rel_image_file_path in rel_image_file_path_list:
        save_video_file_path = save_video_folder_path + rel_image_file_path[:-4] + '.mp4'

        if not sampleImagesToVideo(
            iter_root_folder_path,
            rel_image_file_path,
            save_video_file_path,
            video_width,
            video_height,
            video_fps,
            overwrite):
            print('[ERROR][sample_to_video::sampleImagesFolderToVideos]')
            print('\t sampleImagesToVideo failed!')

            continue

    return True
