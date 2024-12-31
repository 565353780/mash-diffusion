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

    def forwardCondition(self, mash_params: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> dict:
        mash_params_noise = self.model(mash_params, t, cond=condition)

        if self.final_linear:
            mash_params_noise = self.to_outputs(mash_params_noise)

        result_dict = {
            'vt': mash_params_noise
        }

        return result_dict

    def forwardData(self, mash_params: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, condition_drop_prob: float = 0.0) -> torch.Tensor:
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
        condition = condition * context_mask

        result_dict = self.forwardCondition(mash_params, condition, t)

        return result_dict['vt']

    def forward(self, data_dict: dict) -> dict:
        mash_params = data_dict['mash_params']
        condition = data_dict['condition']
        t = data_dict['t']
        condition_drop_prob = data_dict['drop_prob']

        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
        condition = condition * context_mask

        return self.forwardCondition(mash_params, condition, t)

    def forwardWithFixedAnchors(
        self,
        mash_params: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        fixed_anchor_mask: torch.Tensor,
        condition_drop_prob: float = 0.0
    ):
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
        condition = condition * context_mask

        result_dict = self.forwardCondition(mash_params, condition, t)

        mash_params_noise = result_dict['vt']

        mash_params_noise[fixed_anchor_mask] = 0.0

        '''
        sum_data = torch.sum(torch.sum(mash_params_noise, dim=2), dim=0)
        valid_tag = torch.where(sum_data == 0.0)[0] == fixed_anchor_idxs
        is_valid = torch.all(valid_tag)
        print(valid_tag)
        print(is_valid)
        assert is_valid
        '''

        return mash_params_noise
