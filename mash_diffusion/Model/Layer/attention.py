from torch import nn
from einops import rearrange

from flash_attn import flash_attn_kvpacked_func


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        assert isinstance(context_dim, int)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, window_size=-1):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        kv = self.to_kv(context)

        q = rearrange(q, "b n (h d) -> b n h d", h=h)
        kv = rearrange(kv, "b n (p h d) -> b n p h d", h=h, p=2)

        out = flash_attn_kvpacked_func(
            q.bfloat16(), kv.bfloat16(), window_size=(window_size, window_size)
        )
        out = out.to(x.dtype)

        return self.to_out(rearrange(out, "b n h d -> b n (h d)"))
