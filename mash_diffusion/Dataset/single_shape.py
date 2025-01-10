import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor


class SingleShapeDataset(Dataset):
    def __init__(
        self,
        mash_file_path: str,
        size: int = 10000,
        dtype = torch.float32,
    ) -> None:
        assert os.path.exists(mash_file_path)
        self.size = size
        self.dtype = dtype

        self.category_id = 0

        self.mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        self.mash_params = self.normalize(self.mash_params)

        self.mash_params = self.mash_params.to(self.dtype)
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return mash_params

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return mash_params

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        permute_idxs = np.random.permutation(self.mash_params.shape[0])

        data = {
            'mash_params': self.mash_params[permute_idxs],
            'category_id': self.category_id,
        }

        return data
