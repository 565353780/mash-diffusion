import torch
from typing import Union
from torchcfm.conditional_flow_matching import pad_t_like_x

from mash_diffusion.Module.target_ot_plan_sampler import TargetOTPlanSampler


class BatchExactOptimalTransportConditionalFlowMatcher(object):
    def __init__(self, sigma: Union[float, int] = 0.0, target_dim: Union[list, None]=None):
        self.sigma = sigma
        self.ot_sampler = TargetOTPlanSampler(method="exact", target_dim=target_dim)
        return

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = pad_t_like_x(self.sigma, x0)
        return mu_t + sigma_t * epsilon

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        x0_list, x1_list = [], []
        for i in range(x0.shape[0]):
            curr_x0, curr_x1 = self.ot_sampler.sample_plan(x0[i], x1[i])
            x0_list.append(curr_x0)
            x1_list.append(curr_x1)

        batch_x0 = torch.stack(x0_list, dim=0)
        batch_x1 = torch.stack(x1_list, dim=0)

        if t is None:
            t = torch.rand(batch_x0.shape[0]).type_as(batch_x0)
        assert len(t) == batch_x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(batch_x0)
        xt = self.sample_xt(batch_x0, batch_x1, t, eps)
        ut = batch_x1 - batch_x0
        return t, xt, ut

    def guided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None
    ):
        x0_list, x1_list, y0_list, y1_list = [], [], [], []
        for i in range(x0.shape[0]):
            curr_x0, curr_x1, curr_y0, curr_y1 = self.ot_sampler.sample_plan_with_labels(
                x0[i],
                x1[i],
                y0[i] if y0 is not None else None,
                y1[i] if y1 is not None else None)
            x0_list.append(curr_x0)
            x1_list.append(curr_x1)
            y0_list.append(curr_y0)
            y1_list.append(curr_y1)

        batch_x0 = torch.stack(x0_list, dim=0)
        batch_x1 = torch.stack(x1_list, dim=0)

        if t is None:
            t = torch.rand(batch_x0.shape[0]).type_as(batch_x0)
        assert len(t) == batch_x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(batch_x0)
        xt = self.sample_xt(batch_x0, batch_x1, t, eps)
        ut = batch_x1 - batch_x0
        return t, xt, ut, y0, y1
