from mash_diffusion.Model.base_edm import BaseEDM
from mash_diffusion.Model.Transformer.latent_array import LatentArrayTransformer


class EDMLatentTransformer(BaseEDM):
    def __init__(
        self,
        n_latents=512,
        channels=8,
        n_heads=8,
        d_head=64,
        depth=12,
        context_dim=512,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
    ):
        super().__init__(sigma_min, sigma_max, sigma_data)
        self.n_latents = n_latents
        self.channels = channels

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )
        return
