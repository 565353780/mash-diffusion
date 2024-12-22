import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Model.mash import Mash

from conditional_flow_matching.Config.shapenet import CATEGORY_IDS


class SingleShapeDataset(Dataset):
    def __init__(
        self,
        mash_file_path: str,
    ) -> None:
        assert os.path.exists(mash_file_path)

        self.category_id = CATEGORY_IDS['03636649']
        mash_params = np.load(mash_file_path, allow_pickle=True).item()

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')
        mash.loadParams(mask_params, sh_params, rotate_vectors, positions)

        ortho_poses_tensor = mash.toOrtho6DPoses().float()
        positions_tensor = torch.tensor(positions).float()
        mask_params_tesnor = torch.tensor(mask_params).float()
        sh_params_tensor = torch.tensor(sh_params).float()

        self.mash_params = torch.cat((ortho_poses_tensor, positions_tensor, mask_params_tesnor, sh_params_tensor), dim=1)

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
