from mash_diffusion.Model.cfm_hy3ddit import CFMHunyuan3DDiT


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test():
    model = CFMHunyuan3DDiT(
        n_latents=8192,
        mask_degree=2,
        sh_degree=2,
        context_dim=1024,
        n_heads=8,
        d_head=16,
        depth=16,
    )

    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    return True
