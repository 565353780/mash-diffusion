import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor

from distribution_manage.Module.transformer import Transformer

from conditional_flow_matching.Config.shapenet import CATEGORY_IDS


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split

        self.mash_folder_path = self.dataset_root_folder_path + "MashV4/"
        assert os.path.exists(self.mash_folder_path)

        self.paths_list = []

        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)
            # FIXME: for detect test only
            if self.split == "test":
                # categories = ["02691156"]
                categories = ["03001627"]

            categories = ["03001627"]

            print("[INFO][MashDataset::__init__]")
            print("\t start load dataset [" + dataset_name + "]...")
            for category in tqdm(categories):
                category_id = CATEGORY_IDS[category]

                class_folder_path = dataset_folder_path + category + "/"

                mash_filename_list = os.listdir(class_folder_path)

                for mash_filename in mash_filename_list:
                    mash_file_path = class_folder_path + mash_filename

                    self.paths_list.append([mash_file_path, category_id])

        self.transformer = Transformer('../ma-sh/output/multi_linear_transformers.pkl')
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index: int):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, category_id = self.paths_list[index]

        mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        mash_params = self.normalize(mash_params)

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        data = {
            'mash_params': mash_params,
            'category_id': category_id,
        }

        return data
