from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test():
    model = CFMLatentTransformer(
        n_latents=8192,
        mask_degree=2,
        sh_degree=2,
        context_dim=1024,
        n_heads=8,
        d_head=64,
        depth=24,
    )

    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    return True
