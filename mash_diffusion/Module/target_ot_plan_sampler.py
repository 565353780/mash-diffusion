import torch
import warnings
import ot as pot
import numpy as np
from typing import Union

from torchcfm.optimal_transport import OTPlanSampler


class TargetOTPlanSampler(OTPlanSampler):
    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
        target_dim: Union[list, None] = None,
    ) -> None:
        super().__init__(method, reg, reg_m, normalize_cost, num_threads, warn)
        self.target_dim = target_dim
        return

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        if self.target_dim is not None:
            target_x0 = x0[:, self.target_dim]
            target_x1 = x1[:, self.target_dim]
        else:
            target_x0 = x0
            target_x1 = x1

        M = torch.cdist(target_x0, target_x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().to(torch.float32).numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p
