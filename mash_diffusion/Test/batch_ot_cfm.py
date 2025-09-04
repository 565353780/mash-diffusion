import torch
import numpy as np
from tqdm import trange

from mash_diffusion.Module.batch_ot_cfm import (
    BatchExactOptimalTransportConditionalFlowMatcher,
)


def test():
    sigma = 0.0
    target_dim = None
    parallel = True
    dtype = torch.float32

    x0 = torch.randn([6, 8192, 23], dtype=dtype)

    for _ in trange(10):
        permute_idxs = np.random.permutation(x0.shape[0])
        x1 = x0[permute_idxs] * 4.0

        batch_ot_cfm = BatchExactOptimalTransportConditionalFlowMatcher(
            sigma, target_dim, parallel
        )

        t, xt, ut = batch_ot_cfm.sample_location_and_conditional_flow(x0, x1)

    print(t)
    print(t.shape)
    print(xt.shape)
    print(ut.shape)

    exit()
    return True
