import torch
from torch import nn


class BaseCFM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        return

    def forwardCondition(
        self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor
    ) -> dict:
        vt = self.model(xt, t, cond=condition)

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
