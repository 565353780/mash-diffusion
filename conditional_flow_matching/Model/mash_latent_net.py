import torch
import torch.nn as nn

from conditional_flow_matching.Model.Layer.point_embed import PointEmbed
from conditional_flow_matching.Model.Transformer.latent_array import LatentArrayTransformer


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x

class MashLatentNet(torch.nn.Module):
    def __init__(
        self,
        n_latents=400,
        mask_degree: int = 3,
        sh_degree: int = 2,
        embed_dim: int = 512,
        context_dim=1024,
        n_heads=8,
        d_head=64,
        depth=24,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        assert embed_dim % 4 == 0
        self.per_embed_dim = int(embed_dim / 4)
        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.channels = embed_dim

        self.category_emb = nn.Embedding(55, context_dim)

        self.rotation_encoder = nn.Sequential(
            nn.Linear(6, self.per_embed_dim),
            Swish(),
            nn.Linear(self.per_embed_dim, self.per_embed_dim)
        )

        self.position_encoder = PointEmbed(3, 48, self.per_embed_dim)

        self.mask_encoder = nn.Sequential(
            nn.Linear(self.mask_dim, self.per_embed_dim),
            Swish(),
            nn.Linear(self.per_embed_dim, self.per_embed_dim)
        )

        self.sh_encoder = nn.Sequential(
            nn.Linear(self.sh_dim, self.per_embed_dim),
            Swish(),
            nn.Linear(self.per_embed_dim, self.per_embed_dim)
        )

        self.model = LatentArrayTransformer(
            in_channels=self.channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )

        self.to_outputs = nn.Linear(self.channels, 9 + self.mask_dim + self.sh_dim)
        return

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def encodeMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_feature = self.rotation_encoder(mash_params[:, :, :6])
        position_feature = self.position_encoder(mash_params[:, :, 6:9])
        mask_feature = self.mask_encoder(mash_params[:, :, 9:9 + self.mask_dim])
        sh_feature = self.sh_encoder(mash_params[:, :, 9 + self.mask_dim:])

        mash_feature = torch.cat([rotation_feature, position_feature, mask_feature, sh_feature], dim=2)
        return mash_feature

    def forwardCondition(self, mash_params, condition, t):
        mash_feature = self.encodeMash(mash_params)
        mash_params_noise = self.model(mash_feature, t, cond=condition)
        mash_params_noise = self.to_outputs(mash_params_noise)
        return mash_params_noise

    def forward(self, mash_params, condition, t, condition_drop_prob: float = 0.0):
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
