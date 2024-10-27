import sys
sys.path.append('../ma-sh/')

import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Model.mash import Mash


class ImageEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        preload: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.preload = preload

        self.mash_folder_path = self.dataset_root_folder_path + "MashV4/"
        self.image_embedding_folder_path = self.dataset_root_folder_path + "ImageEmbedding/"

        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.image_embedding_folder_path)

        self.path_dict_list = []

        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)
            # FIXME: for detect test only
            if self.split == "test":
                # categories = ["02691156"]
                categories = ["03001627"]

            for i, category in enumerate(categories):
                class_folder_path = dataset_folder_path + category + "/"

                mash_filename_list = os.listdir(class_folder_path)

                print("[INFO][ImageEmbeddingDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(i + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for mash_filename in mash_filename_list:
                    path_dict = {}
                    mash_file_path = class_folder_path + mash_filename

                    if not os.path.exists(mash_file_path):
                        continue

                    image_embedding_file_path = self.image_embedding_folder_path + dataset_name + '/' + \
                        category + '/' + mash_filename

                    if not os.path.exists(image_embedding_file_path):
                        continue

                    if self.preload:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        image_embedding = np.load(image_embedding_file_path, allow_pickle=True).item()
                        path_dict['mash'] = mash_params
                        path_dict['image_embedding'] = image_embedding
                    else:
                        path_dict['mash'] = mash_file_path
                        path_dict['image_embedding'] = image_embedding_file_path

                    self.path_dict_list.append(path_dict)
        return

    def __len__(self):
        return len(self.path_dict_list)

    def __getitem__(self, index):
        index = index % len(self.path_dict_list)

        data = {}

        path_dict = self.path_dict_list[index]

        if self.preload:
            mash_params = path_dict['mash']
            image_embedding = path_dict['image_embedding']
        else:
            mash_file_path = path_dict['mash']
            image_embedding_file_path = path_dict['image_embedding']
            mash_params = np.load(mash_file_path, allow_pickle=True).item()
            image_embedding = np.load(image_embedding_file_path, allow_pickle=True).item()

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        if self.split == "train" and False:
            scale_range = [0.8, 1.2]
            move_range = [-0.2, 0.2]

            random_scale = (
                scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
            )
            random_translate = move_range[0] + (
                move_range[1] - move_range[0]
            ) * np.random.rand(3)

            positions = positions * random_scale + random_translate
            sh_params = sh_params * random_scale

        permute_idxs = np.random.permutation(rotate_vectors.shape[0])

        rotate_vectors = rotate_vectors[permute_idxs]
        positions = positions[permute_idxs]
        mask_params = mask_params[permute_idxs]
        sh_params = sh_params[permute_idxs]

        mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')
        mash.loadParams(mask_params, sh_params, rotate_vectors, positions)

        ortho_poses_tensor = mash.toOrtho6DPoses().float()
        positions_tensor = torch.tensor(positions).float()
        mask_params_tesnor = torch.tensor(mask_params).float()
        sh_params_tensor = torch.tensor(sh_params).float()

        cfm_mash_params = torch.cat((ortho_poses_tensor, positions_tensor, mask_params_tesnor, sh_params_tensor), dim=1)

        data['cfm_mash_params'] = cfm_mash_params

        image_embedding_tensor = {}

        for key, item in image_embedding.items():
            image_embedding_tensor[key] = torch.from_numpy(item).float()

        data['image_embedding'] = image_embedding_tensor
        return data
