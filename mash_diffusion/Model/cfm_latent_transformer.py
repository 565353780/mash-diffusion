from mash_diffusion.Model.base_cfm import BaseCFM
from mash_diffusion.Model.Transformer.latent_array import LatentArrayTransformer


class CFMLatentTransformer(BaseCFM):
    def __init__(
        self,
        n_latents=400,
        mask_degree: int = 3,
        sh_degree: int = 2,
        context_dim=1024,
        n_heads=8,
        d_head=64,
        depth=24,
    ):
        super().__init__()
        self.n_latents = n_latents

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.channels = 9 + self.mask_dim + self.sh_dim

        self.model = LatentArrayTransformer(
            in_channels=self.channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )
        return
