import torch
import numpy as np
from math import ceil
from torch import nn


class PointEmbed(nn.Module):
    def __init__(self, data_dim=3, hidden_dim=48, dim=128):
        super().__init__()

        cos_sin_data_dim = 2 * data_dim

        self.embedding_dim = ceil(hidden_dim // cos_sin_data_dim) * cos_sin_data_dim

        e = (
            torch.pow(2, torch.arange(self.embedding_dim // cos_sin_data_dim)).float()
            * np.pi
        )
        basis = torch.zeros(
            [data_dim, data_dim * self.embedding_dim // cos_sin_data_dim]
        )
        for i in range(data_dim):
            basis[i, e.shape[0] * i : e.shape[0] * (i + 1)] = e

        # data_dim x (embedding_dim // cos_sin_data_dim)
        self.register_buffer("basis", basis)

        self.mlp = nn.Linear(self.embedding_dim + data_dim, dim)
        return

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x cos_sin_data_dim
        embed = self.embed(input, self.basis)

        embed = torch.cat([embed, input], dim=2)

        # B x N x C
        embed = self.mlp(embed)
        return embed
