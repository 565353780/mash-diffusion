import torch

from mash_diffusion.Model.hy3ddit import Hunyuan3DDiT


class CFMHunyuan3DDiT(torch.nn.Module):
    def __init__(
        self,
        n_latents=8192,
        mask_degree: int = 2,
        sh_degree: int = 2,
        context_dim=1024,
        num_heads=16,
        dim_head=64,
        depth=16,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.channels = 9 + self.mask_dim + self.sh_dim

        hidden_size = dim_head * num_heads
        self.model = Hunyuan3DDiT(
            in_channels=self.channels,
            context_in_dim=context_dim,
            hidden_size=hidden_size,
            mlp_ratio=4.0,
            num_heads=num_heads,
            depth=depth,
            depth_single_blocks=32,
            axes_dim=[dim_head],
            theta=10_000,
            qkv_bias=True,
            time_factor=1000,
            guidance_embed=False,
        )
        return

    def forwardCondition(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> dict:
        vt = self.model(xt, t, {"main": condition})

        result_dict = {"vt": vt}

        return result_dict

    def forwardData(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        result_dict = self.forwardCondition(xt, condition, t)

        vt = result_dict["vt"]

        return vt

    def forward(self, data_dict: dict) -> dict:
        xt = data_dict["xt"]
        t = data_dict["t"]
        condition = data_dict["condition"]
        drop_prob = data_dict["drop_prob"]

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        if drop_prob > 0:
            drop_mask = torch.rand_like(condition) <= drop_prob
            condition[drop_mask] = 0

        result_dict = self.forwardCondition(xt, condition, t)

        return result_dict

    def forwardWithFixedAnchors(
        self,
        xt: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        fixed_anchor_mask: torch.Tensor,
    ):
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        result_dict = self.forwardCondition(xt, condition, t)

        vt = result_dict["vt"]

        vt[fixed_anchor_mask] = 0.0

        """
        sum_data = torch.sum(torch.sum(mash_params_noise, dim=2), dim=0)
        valid_tag = torch.where(sum_data == 0.0)[0] == fixed_anchor_idxs
        is_valid = torch.all(valid_tag)
        print(valid_tag)
        print(is_valid)
        assert is_valid
        """

        return vt
