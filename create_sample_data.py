import MFSClient

import io
import os
import random
from tqdm import tqdm

from mash_diffusion.Dataset.tos_image import TOSImageDataset
from mash_diffusion.Method.time import getCurrentTime


if __name__ == "__main__":
    save_folder_path = "./output/sample/" + getCurrentTime() + "/"

    sample_shape_num = 100

    client = MFSClient.MFSClient2()

    dataset = TOSImageDataset(
        client,
        mash_bucket="mm-data-general-model-trellis",
        mash_folder_key="mash/",
        image_bucket="mm-data-general-model-v1",
        image_folder_key="rendering/orient_cam72_base/",
        transform=None,
        paths_file_path="./data/tos_paths_list.npy",
        return_raw_data=True,
    )

    all_shape_num = len(dataset)
    if sample_shape_num >= all_shape_num:
        random_shape_idxs = list(range(all_shape_num))
    else:
        random_shape_idxs = random.sample(range(all_shape_num), sample_shape_num)

    print("[INFO][create_sample_data::demo]")
    print("\t start sample condition data...")
    for shape_idx in tqdm(random_shape_idxs):
        data_dict = dataset[shape_idx]

        current_save_folder_path = save_folder_path + str(shape_idx) + "/"
        os.makedirs(current_save_folder_path, exist_ok=True)

        with open(current_save_folder_path + "condition_image.jpg", "wb") as f:
            f.write(io.BytesIO(data_dict["image_data"]).read())

        with open(current_save_folder_path + "gt_mash.npy", "wb") as f:
            f.write(io.BytesIO(data_dict["mash_data"]).read())
