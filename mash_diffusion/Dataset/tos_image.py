import os
import io
import tos
import torch
import random
import numpy as np
from PIL import Image
from typing import Union
from random import choice
from torch.utils.data import Dataset

from mash_diffusion.Method.io import toMashTensor
from mash_diffusion.Method.tos import listdirTOS, isFileExist, filterExistFiles

view_candidates = [
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    64,
    65,
    66,
    67,
]


def loadImageBucketDict() -> Union[dict, None]:
    bucket_txt_file_path = "./data/uids_bucket.txt"
    if not os.path.exists(bucket_txt_file_path):
        print("[ERROR][TOSImageDataset::loadImageBucketDict]")
        print("\t bucket txt file not exist!")
        print("\t bucket_txt_file_path:", bucket_txt_file_path)
        return None

    image_bucket_dict = {}
    with open(bucket_txt_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        shape_id, bucket = line.split()
        image_bucket_dict[shape_id] = bucket

    return image_bucket_dict


def getMashFileKey(rel_mash_folder_path: str, shape_id: str) -> str:
    return rel_mash_folder_path + shape_id + ".npy"


def getRandomImageFileKey(rel_image_folder_path: str, shape_id: str) -> str:
    random_view_idx = choice(view_candidates)

    rel_image_file_path = (
        rel_image_folder_path
        + shape_id
        + "/View"
        + str(random_view_idx)
        + "_FinalColor.jpg"
    )
    return rel_image_file_path


class TOSImageDataset(Dataset):
    def __init__(
        self,
        mash_bucket: str,
        mash_folder_key: str,
        image_bucket: str,
        image_folder_key: str,
        transform,
        split: str = "train",
        dtype=torch.float32,
        empty: bool = False,
    ) -> None:
        self.mash_bucket = mash_bucket
        self.mash_folder_key = mash_folder_key
        self.image_bucket = image_bucket
        self.image_folder_key = image_folder_key
        self.transform = transform
        self.split = split
        self.dtype = dtype

        self.client = None

        self.createClient()
        if not isinstance(self.client, tos.TosClientV2):
            print("[ERROR][TOSImageDataset::__init__]")
            print("\t createClient failed!")
            exit()

        if not empty:
            image_bucket_dict = loadImageBucketDict()
            if image_bucket_dict is None:
                print("[ERROR][TOSImageDataset::__init__]")
                print("\t loadBucketDict failed!")
                exit()

            mash_file_keys = listdirTOS(
                self.client, self.mash_bucket, self.mash_folder_key
            )

            shape_id_list = []
            for mash_file_key in mash_file_keys:
                shape_id_list.append(
                    mash_file_key.split(self.mash_folder_key)[1].split(".")[0]
                )

            shape_image_pairs = []
            for shape_id in shape_id_list:
                image_file_key = getRandomImageFileKey(self.image_folder_key, shape_id)

                image_bucket = image_bucket_dict[shape_id]
                if image_bucket.endswith("v2"):
                    image_file_key = "data/" + image_file_key

                shape_image_pairs.append([shape_id, image_file_key])

            valid_shape_id_list = filterExistFiles(
                self.client, image_bucket_dict, shape_image_pairs, max_workers=64
            )

            self.paths_list = []
            for valid_shape_id in valid_shape_id_list:
                mash_file_key = getMashFileKey(self.mash_folder_key, valid_shape_id)
                image_file_key = getRandomImageFileKey(
                    self.image_folder_key, valid_shape_id
                )

                image_bucket = image_bucket_dict[valid_shape_id]
                if image_bucket.endswith("v2"):
                    image_file_key = "data/" + image_file_key

                self.paths_list.append([mash_file_key, image_bucket, image_file_key])

            print(len(shape_id_list), "mashes found")
            print(len(self.paths_list), "valid mash image pairs found")
            if len(self.paths_list) == 0:
                print("[ERROR][TOSImageDataset::__init__]")
                print("\t valid mash image pairs not found!")
                exit()

        self.output_error = True

        self.invalid_image_file_keys = []
        return

    def createClient(self) -> bool:
        ak = os.getenv("TOS_ACCESS_KEY")
        sk = os.getenv("TOS_SECRET_KEY")
        assert isinstance(ak, str) and isinstance(sk, str)
        endpoint = "tos-cn-beijing.volces.com"
        region = "cn-beijing"

        self.client = tos.TosClientV2(ak, sk, endpoint, region)
        return True

    def closeClient(self) -> bool:
        if isinstance(self.client, tos.TosClientV2):
            self.client.close()
        return True

    def __len__(self):
        return len(self.paths_list)

    def getRandomItem(self):
        random_idx = random.randint(0, len(self.paths_list) - 1)
        return self.__getitem__(random_idx)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_key, image_bucket, image_file_key = self.paths_list[index]

        if not isFileExist(self.client, image_bucket, image_file_key):
            if self.output_error:
                print("[ERROR][TOSImageDataset::__getitem__]")
                print("\t this image file is not valid!")
            return self.getRandomItem()

        if image_file_key in self.invalid_image_file_keys:
            return self.getRandomItem()

        mash_stream = self.client.get_object(self.mash_bucket, mash_file_key)
        image_stream = self.client.get_object(image_bucket, image_file_key)

        mash_data = mash_stream.read()

        mash_params_dict = np.load(io.BytesIO(mash_data), allow_pickle=True).item()
        mash_params = toMashTensor(mash_params_dict)
        assert mash_params is not None, (
            "[ERROR][TOSImageDataset::__getitem__] mash_params is None!"
        )

        image_data = image_stream.read()

        try:
            image = Image.open(io.BytesIO(image_data))
        except KeyboardInterrupt:
            print("[INFO][TOSImageDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][TOSImageDataset::__getitem__]")
                print("\t image file is not valid!")
                print("\t image_file_path:", image_file_key)
                print("\t error info:", e)

            self.invalid_image_file_keys.append(image_file_key)
            return self.getRandomItem()

        image = image.convert("RGB")

        image = self.transform(image)

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        data = {
            "mash_params": mash_params.to(self.dtype),
            "image": image.to(self.dtype),
        }

        return data
