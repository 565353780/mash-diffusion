import sys
sys.path.append('../ma-sh/')

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from conditional_flow_matching.Config.shapenet import CATEGORY_IDS

from ma_sh.Model.mash import Mash


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        preload_data: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.preload_data = preload_data

        self.mash_folder_path = self.dataset_root_folder_path + "MashV4/"
        self.split_folder_path = self.dataset_root_folder_path + "SplitMashCFM/"
        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.split_folder_path)

        self.paths_list = []

        dataset_name_list = os.listdir(self.split_folder_path)

        for dataset_name in dataset_name_list:
            mash_split_folder_path = self.split_folder_path + dataset_name + "/"

            categories = os.listdir(mash_split_folder_path)
            # FIXME: for detect test only
            if self.split == "test":
                # categories = ["02691156"]
                categories = ["03001627"]

            for j, category in enumerate(categories):
                modelid_list_file_path = (
                    mash_split_folder_path + category + "/" + self.split + ".txt"
                )
                if not os.path.exists(modelid_list_file_path):
                    continue

                with open(modelid_list_file_path, "r") as f:
                    modelid_list = f.read().split()

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(j + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for modelid in tqdm(modelid_list):
                    mash_file_path = (
                        self.mash_folder_path
                        + dataset_name
                        + "/"
                        + category
                        + "/"
                        + modelid
                        + ".npy"
                    )

                    category_id = CATEGORY_IDS[category]

                    if self.preload_data:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        self.paths_list.append([mash_params, category_id])
                    else:
                        self.paths_list.append([mash_file_path, category_id])

        return

    def __len__(self):
        if self.split == "train":
            return len(self.paths_list)
        else:
            return len(self.paths_list)

    def __getitem__(self, index: int):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        if self.preload_data:
            mash_params, category_id = self.paths_list[index]
        else:
            mash_file_path, category_id = self.paths_list[index]
            mash_params = np.load(mash_file_path, allow_pickle=True).item()

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

        data = {
            'cfm_mash_params': cfm_mash_params,
            'category_id': category_id,
        }

        return data
