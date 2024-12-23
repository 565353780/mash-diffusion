import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor

from conditional_flow_matching.Config.shapenet import CATEGORY_IDS


class SingleShapeDataset(Dataset):
    def __init__(
        self,
        mash_file_path: str,
    ) -> None:
        assert os.path.exists(mash_file_path)

        self.category_id = CATEGORY_IDS['03636649']

        self.mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        self.mash_params = self.normalize(self.mash_params)
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        self.min = torch.min(mash_params, dim=0, keepdim=True)[0]
        self.max = torch.max(mash_params, dim=0, keepdim=True)[0]

        return (mash_params - self.min) / (self.max - self.min)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return mash_params * (self.max - self.min) + self.min

    def __len__(self):
        return 10000

    def __getitem__(self, index: int):
        permute_idxs = np.random.permutation(self.mash_params.shape[0])

        data = {
            'mash_params': self.mash_params[permute_idxs],
            'category_id': self.category_id,
        }

        return data
