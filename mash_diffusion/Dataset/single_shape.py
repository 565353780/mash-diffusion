import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor

from distribution_manage.Module.transformer import Transformer

from mash_diffusion.Config.shapenet import CATEGORY_IDS


class SingleShapeDataset(Dataset):
    def __init__(
        self,
        mash_file_path: str,
    ) -> None:
        assert os.path.exists(mash_file_path)

        self.category_id = CATEGORY_IDS['03636649']

        self.mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        self.transformer = Transformer('../ma-sh/output/multi_linear_transformers.pkl')

        self.mash_params = self.normalize(self.mash_params)
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return 10000

    def __getitem__(self, index: int):
        permute_idxs = np.random.permutation(self.mash_params.shape[0])

        data = {
            'mash_params': self.mash_params[permute_idxs],
            'category_id': self.category_id,
        }

        return data
