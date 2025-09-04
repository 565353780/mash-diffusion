import torch
from torch.amp import autocast

from mash_diffusion.Model.hy3ddit import Hunyuan3DDiT


def test():
    dtype = torch.float32
    device = "cuda:0"

    model = Hunyuan3DDiT(
        in_channels=25,
        context_in_dim=1024,
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        depth=16,
        depth_single_blocks=32,
        axes_dim=[64],
        theta=10_000,
        qkv_bias=True,
        time_factor=1000,
    ).to(device)

    data_dict = {
        "mash_params": torch.randn([2, 400, 25], dtype=dtype, device=device),
        "condition": torch.randn([2, 1397, 1024], dtype=dtype, device=device),
        "t": torch.randn([2], dtype=dtype, device=device),
    }

    with autocast("cuda", dtype=torch.float16):
        y = model(
            data_dict["mash_params"], data_dict["t"], {"main": data_dict["condition"]}
        )

    print(y.shape)
    return True
