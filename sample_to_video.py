from conditional_flow_matching.Method.video import sampleImagesFolderToVideos


if __name__ == "__main__":
    iter_root_folder_path = './output/render_sample/20241204_18:06:56/'
    save_video_folder_path = './output/video_sample/20241204_18:06:56/'
    video_width = 540
    video_height = 540
    video_fps = 10
    back_ground = [255, 255, 255]
    overwrite = False

    sampleImagesFolderToVideos(
        iter_root_folder_path,
        save_video_folder_path,
        video_width,
        video_height,
        video_fps,
        back_ground,
        overwrite
    )
