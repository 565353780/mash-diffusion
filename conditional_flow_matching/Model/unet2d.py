import torch
from torch import nn

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

class MashUNet(nn.Module):
    def __init__(self,
                 condition_dim: int = 768) -> None:
        super().__init__()

        self.category_emb = nn.Embedding(55, condition_dim)

        self.unet = UNet2DConditionModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(32, 64, 128, 128),
        cross_attention_dim=condition_dim
        )
        return

    def emb_category(self, class_labels: torch.Tensor) -> torch.Tensor:
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, mash_params: torch.Tensor, condition: torch.Tensor, t: torch.Tensor, condition_drop_prob: float = 0.0) -> torch.Tensor:
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        if condition_drop_prob > 0:
            # dropout context with some probability
            context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
            condition = condition * context_mask

        mash_params = mash_params.unsqueeze(1)

        mash_noise = self.unet(mash_params, t, condition).sample

        mash_noise = mash_noise.squeeze(1)
        return mash_noise
