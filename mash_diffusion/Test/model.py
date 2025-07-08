import torch
from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer


def test():
    device = "cpu"

    model = CFMLatentTransformer(
        n_latents=400,
        mask_degree=3,
        sh_degree=2,
        context_dim=1024,
        n_heads=16,
        d_head=64,
        depth=24,
    ).to(device)

    data_dict = {
        "mash_params": torch.randn([2, 400, 25], device=device),
        "condition": torch.randn([2, 1397, 1024], device=device),
        "t": torch.randn([2], device=device),
        "drop_prob": 0.0,
    }

    result_dict = model(data_dict)

    return True
