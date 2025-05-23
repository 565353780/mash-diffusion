from mash_diffusion.Method.video import sampleImagesFolderToVideos


if __name__ == "__main__":
    timestamp = '48depth_600epoch'
    iter_root_folder_path = './output/render_sample/' + timestamp + '/'
    save_video_folder_path = './output/video_sample/' + timestamp + '/'
    video_width = 540
    video_height = 540
    video_fps = 30
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
