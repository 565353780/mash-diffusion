import os

from mash_diffusion.Dataset.tos_image import TOSImageDataset


if __name__ == "__main__":
    paths_file_path = "./data/tos_paths_list.npy"
    overwrite = True

    if not os.path.exists(paths_file_path):
        TOSImageDataset(
            mash_bucket="mm-data-general-model-trellis",
            mash_folder_key="mash/",
            image_bucket="mm-data-general-model-v1",
            image_folder_key="rendering/orient_cam72_base/",
            transform=None,
        ).savePaths(paths_file_path, overwrite)
