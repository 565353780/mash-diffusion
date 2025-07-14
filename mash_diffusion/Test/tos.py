import sys

sys.path.append("../ma-sh/")

import torch

from mash_diffusion.Dataset.tos_image import TOSImageDataset


def test():
    tos_image_dataset = TOSImageDataset(
        bucket="mm-data-general-model-trellis",
        mash_folder_key="mash/",
        image_folder_key="rendering/orient_cam72_base/",
        transform=None,
        split="train",
        dtype=torch.float32,
    )

    data = tos_image_dataset[0]
    return True
