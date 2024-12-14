import torch.nn as nn
import torch.nn.functional as F

from conditional_flow_matching.Model.positional_encoding import PositionalEncoding
from conditional_flow_matching.Model.positional_embedding import PositionalEmbedding
from conditional_flow_matching.Model.Transformer.basic_block import BasicTransformerBlock


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        t_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.map_noise = PositionalEncoding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # ###
        # self.pos_emb = nn.Embedding(512, inner_dim)
        # ###

    def forward(self, x, t, cond=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)

        # ###
        # x = x + self.pos_emb.weight[None]
        # ###

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)

        x = self.proj_out(x)
        return x
