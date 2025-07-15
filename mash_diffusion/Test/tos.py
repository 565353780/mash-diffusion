import sys

sys.path.append("../ma-sh/")

import torch

from mash_diffusion.Dataset.tos_image import TOSImageDataset


def test():
    tos_image_dataset = TOSImageDataset(
        mash_bucket="mm-data-general-model-trellis",
        mash_folder_key="mash/",
        image_bucket="mm-data-general-model-v1",
        image_folder_key="rendering/orient_cam72_base/",
        transform=None,
        split="train",
        dtype=torch.float32,
    )

    data = tos_image_dataset[0]
    return True
