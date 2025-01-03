import torch
import torch.nn as nn

from mash_diffusion.Model.Transformer.latent_array import LatentArrayTransformer


class CFMLatentTransformer(torch.nn.Module):
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

        self.category_emb = nn.Embedding(55, context_dim)

        self.model = LatentArrayTransformer(
            in_channels=self.channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )

        self.final_linear = True

        if self.final_linear:
            self.to_outputs = nn.Linear(self.channels, self.channels)
        return

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forwardCondition(self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> dict:
        vt = self.model(xt, t, cond=condition)

        if self.final_linear:
            vt = self.to_outputs(vt)

        result_dict = {
            'vt': vt
        }

        return result_dict

    def forwardData(self, xt: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([xt.shape[0]], dtype=torch.long, device=xt.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        result_dict = self.forwardCondition(xt, condition, t)

        vt = result_dict['vt']

        return vt

    def forward(self, data_dict: dict) -> dict:
        xt = data_dict['xt']
        t = data_dict['t']
        condition = data_dict['condition']
        drop_prob = data_dict['drop_prob']

        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([xt.shape[0]], dtype=torch.long, device=xt.device))
        else:
            condition = self.emb_category(condition)

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
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([xt.shape[0]], dtype=torch.long, device=xt.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        result_dict = self.forwardCondition(xt, condition, t)

        vt = result_dict['vt']

        vt[fixed_anchor_mask] = 0.0

        '''
        sum_data = torch.sum(torch.sum(mash_params_noise, dim=2), dim=0)
        valid_tag = torch.where(sum_data == 0.0)[0] == fixed_anchor_idxs
        is_valid = torch.all(valid_tag)
        print(valid_tag)
        print(is_valid)
        assert is_valid
        '''

        return vt
