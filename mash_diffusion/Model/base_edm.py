import torch
from torch import nn


class BaseEDM(nn.Module):
    def __init__(self, sigma_min=0, sigma_max=float("inf"), sigma_data=1):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.model = nn.Module()
        return

    def forwardCondition(self, x, sigma, condition):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=condition)
        assert F_x.dtype == dtype

        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        result_dict = {
            "D_x": D_x,
        }

        return result_dict

    def forwardData(
        self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        result_dict = self.forwardCondition(x, sigma, condition)

        return result_dict["D_x"]

    def forward(self, data_dict: dict):
        x = data_dict["noise"]
        sigma = data_dict["sigma"]
        condition = data_dict["condition"]
        drop_prob = data_dict["drop_prob"]
        fixed_prob = data_dict["fixed_prob"]

        if drop_prob > 0:
            drop_mask = torch.rand_like(condition) <= drop_prob
            condition[drop_mask] = 0

        if fixed_prob > 0:
            mash_params = data_dict["mash_params"]

            fixed_mask = torch.rand_like(mash_params) <= fixed_prob

            x[fixed_mask] = mash_params[fixed_mask]

        return self.forwardCondition(x, sigma, condition)
