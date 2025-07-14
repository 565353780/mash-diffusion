import os
import torch
import numpy as np
from typing import Union


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def toMashTensor(mash_params_dict: dict) -> torch.Tensor:
    positions = torch.from_numpy(mash_params_dict["positions"]).to(torch.float32)
    ortho_poses = torch.from_numpy(mash_params_dict["ortho_poses"]).to(torch.float32)
    mask_params = torch.from_numpy(mash_params_dict["mask_params"]).to(torch.float32)
    sh_params = torch.from_numpy(mash_params_dict["sh_params"]).to(torch.float32)

    mash_params = torch.cat([positions, ortho_poses, mask_params, sh_params], dim=-1)
    return mash_params


def loadMashTensor(mash_file_path: str) -> Union[torch.Tensor, None]:
    if not os.path.exists(mash_file_path):
        print("[ERROR][io::loadMashTensor]")
        print("\t mash file not exist!")
        print("\t mash_file_path:", mash_file_path)
        return None

    mash_params_dict = np.load(mash_file_path, allow_pickle=True).item()

    mash_params = toMashTensor(mash_params_dict)
    return mash_params
