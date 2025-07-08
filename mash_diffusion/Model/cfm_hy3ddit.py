from mash_diffusion.Model.base_cfm import BaseCFM
from mash_diffusion.Model.hy3ddit import Hunyuan3DDiT


class CFMHunyuan3DDiT(BaseCFM):
    def __init__(
        self,
        n_latents=8192,
        mask_degree: int = 2,
        sh_degree: int = 2,
        context_dim=1024,
        n_heads=16,
        d_head=64,
        depth=16,
    ):
        super().__init__()
        self.n_latents = n_latents

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.channels = 9 + self.mask_dim + self.sh_dim

        hidden_size = d_head * n_heads
        self.model = Hunyuan3DDiT(
            in_channels=self.channels,
            context_in_dim=context_dim,
            hidden_size=hidden_size,
            mlp_ratio=4.0,
            num_heads=n_heads,
            depth=depth,
            depth_single_blocks=32,
            axes_dim=[d_head],
            theta=10_000,
            qkv_bias=True,
            time_factor=1000,
            guidance_embed=False,
        )
        return
