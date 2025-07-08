import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mash_diffusion.Method.io import loadMashTensor


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        mash_folder_name: str,
        image_folder_name: str,
        transform,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.transform = transform
        self.split = split
        self.dtype = dtype

        self.mash_folder_path = self.dataset_root_folder_path + mash_folder_name + "/"
        self.image_root_folder_path = (
            self.dataset_root_folder_path + image_folder_name + "/"
        )

        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.image_root_folder_path)

        self.output_error = False

        self.invalid_image_file_path_list = []

        self.paths_list = []

        print("[INFO][ImageDataset::__init__]")
        print("\t start load mash and image datasets...")
        for root, _, files in os.walk(self.image_root_folder_path):
            if len(files) == 0:
                continue

            rel_folder_path = os.path.relpath(root, self.image_root_folder_path)

            mash_file_path = self.mash_folder_path + rel_folder_path + ".npy"

            if not os.path.exists(mash_file_path):
                continue

            image_file_path_list = []
            for file in files:
                if not file.endswith(".jpg") or file.endswith("_tmp.jpg"):
                    continue

                image_file_path_list.append(root + "/" + file)

            if len(image_file_path_list) == 0:
                continue

            image_file_path_list.sort()

            self.paths_list.append([mash_file_path, image_file_path_list])

        self.paths_list.sort(key=lambda x: x[0])

        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, image_file_path_list = self.paths_list[index]

        if not os.path.exists(mash_file_path):
            if self.output_error:
                print("[ERROR][ImageDataset::__getitem__]")
                print("\t this image file is not valid!")
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        image_file_idx = np.random.choice(len(image_file_path_list))

        image_file_path = image_file_path_list[image_file_idx]

        if image_file_path in self.invalid_image_file_path_list:
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        try:
            image = Image.open(image_file_path)
        except KeyboardInterrupt:
            print("[INFO][imageDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][imageDataset::__getitem__]")
                print("\t this npy file is not valid!")
                print("\t image_file_path:", image_file_path)
                print("\t error info:", e)

            self.invalid_image_file_path_list.append(image_file_path)
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        image = image.convert("RGB")

        image = self.transform(image)

        mash_params = loadMashTensor(mash_file_path)
        assert mash_params is not None, (
            "[ERROR][ImageDataset::__getitem__] mash_params is None!"
        )

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        data = {
            "mash_params": mash_params.to(self.dtype),
            "image": image.to(self.dtype),
        }

        return data
