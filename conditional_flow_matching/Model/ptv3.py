import sys
sys.path.append('../point-cept/')

import torch
from torch import nn

from pointcept.models.point_transformer_v3 import PointTransformerV3

from mash_autoencoder.Model.Layer.point_embed import PointEmbed

class PTV3(nn.Module):
    def __init__(self,
                 mask_degree: int = 3,
                 sh_degree: int = 2,
                 positional_embedding_dim: int = 48) -> None:
        super().__init__()

        mask_dim = 2 * mask_degree + 1
        sh_dim = (sh_degree + 1) ** 2

        ptv3_output_dim = 64

        self.embedding = PointEmbed(hidden_dim=positional_embedding_dim, dim=1)

        self.feature_encoder = PointTransformerV3(
            in_channels=self.embedding.embedding_dim,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(ptv3_output_dim, 64, 128, 256),
            dec_patch_size=(1024, 1024, 1024, 1024),
        )

        if False:
            self.ortho_poses_decoder = nn.Sequential(
                nn.Linear(ptv3_output_dim, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, 3),
            )

            self.positions_decoder = nn.Sequential(
                nn.Linear(ptv3_output_dim, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, 3),
            )

            self.mask_params_decoder = nn.Sequential(
                nn.Linear(ptv3_output_dim, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, mask_dim),
            )

            self.sh_params_decoder = nn.Sequential(
                nn.Linear(ptv3_output_dim, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, ptv3_output_dim // 2),
                nn.Linear(ptv3_output_dim // 2, sh_dim),
            )
        else:
            self.cfm_mash_params_decoder = nn.Linear(ptv3_output_dim, 9 + mask_dim + sh_dim)

        return

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        surface_points = data['surface_points']

        if drop_prob > 0.0:
            mask = surface_points.new_empty(*surface_points.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            surface_points = surface_points * mask.unsqueeze(-1).expand_as(surface_points).type(surface_points.dtype)

        surface_point_embeddings = self.embedding.embed(surface_points, self.embedding.basis)

        input_dict = {
            'coord': surface_points.reshape(-1, 3),
            'feat': surface_point_embeddings.reshape(-1, self.embedding.embedding_dim),
            'batch': torch.cat([torch.ones(surface_points.shape[1], dtype=torch.long, device=surface_points.device) * i for i in range(surface_points.shape[0])]),
            'grid_size': 0.01,
        }

        points = self.feature_encoder(input_dict)

        x = points.feat.reshape(surface_points.shape[0], surface_points.shape[1], -1)

        ortho_poses = self.ortho_poses_decoder(x)
        delta_positions = self.positions_decoder(x)
        positions = surface_points + delta_positions
        mask_params = self.mask_params_decoder(x)
        sh_params = self.sh_params_decoder(x)

        mash_dict = {
            'ortho_poses': ortho_poses,
            'positions': positions,
            'mask_params': mask_params,
            'sh_params': sh_params,
        }

        return {"mash_params_dict": mash_dict}
