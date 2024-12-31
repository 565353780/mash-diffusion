import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Union, Tuple


def toTSteps(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: int = 7,
) -> torch.Tensor:
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = torch.cat(
        [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    return t_steps

def addNoise(
    x_cur: torch.Tensor,
    t_cur: torch.Tensor,
    num_steps: int,
    randn_like=torch.randn_like,
    S_churn: int = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if S_churn == 0:
        return x_cur, t_cur

    # Increase noise temporarily.
    gamma = (
        min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    )

    if gamma == 0.0:
        return x_cur, t_cur

    t_hat = t_cur + gamma * t_cur
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

    return x_hat, t_hat

def toMaskedNoise(
    latents: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    fixed_mask: Union[torch.Tensor, None] = None,
    randn_like = torch.randn_like,
) -> torch.Tensor:
    if fixed_mask is None:
        return x

    fixed_latents = latents[fixed_mask]

    noise = randn_like(latents) * t

    fixed_x = fixed_latents + noise[fixed_mask]

    x[fixed_mask] = fixed_x

    return x

def deNoise(
    net: nn.Module,
    latents: torch.Tensor,
    x_hat: torch.Tensor,
    t_hat: torch.Tensor,
    t_next: torch.Tensor,
    condition: Union[torch.Tensor, None] = None,
    apply_second_order_correction: bool = False,
    fixed_mask: Union[torch.Tensor, None] = None,
    randn_like = torch.randn_like,
) -> torch.Tensor:
    x_hat = toMaskedNoise(latents, x_hat, t_hat, fixed_mask, randn_like)

    # Euler step.
    denoised = net.forwardData(x_hat, t_hat, condition).to(torch.float64)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if apply_second_order_correction:
        x_next = toMaskedNoise(latents, x_next, t_next, fixed_mask, randn_like)
        denoised = net.forwardData(x_next, t_next, condition).to(torch.float64)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def edm_sampler(
    net: nn.Module,
    latents: torch.Tensor,
    condition: Union[torch.Tensor, None] = None,
    randn_like=torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: int = 7,
    # S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
    S_churn: int = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    fixed_mask: Union[torch.Tensor, None] = None,
) -> list:
    x_list = []

    latents = latents.to(torch.float64)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    t_steps = toTSteps(num_steps, sigma_min, sigma_max, rho).to(latents.device)

    # Main sampling loop.
    x_next = randn_like(latents) * t_steps[0]

    x_list.append(x_next.detach().clone())

    # 0, ..., N-1
    for i, (t_cur, t_next) in enumerate(zip(tqdm(t_steps[:-1]), t_steps[1:])):
        x_cur = x_next

        x_hat, t_hat = addNoise(x_cur, t_cur, num_steps, randn_like, S_churn, S_min, S_max, S_noise)

        apply_second_order_correction = i < num_steps - 1
        x_next = deNoise(net, latents, x_hat, t_hat, t_next, condition, apply_second_order_correction, fixed_mask, randn_like)

        x_list.append(x_next.detach().clone())

    return x_list


def ablation_sampler(
    net,
    latents,
    condition=None,
    randn_like=torch.randn_like,
    num_steps=512,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="euler",
    discretization="vp",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    replace_noise=None,
):
    assert solver in ["euler", "heun"]
    assert discretization in ["vp", "ve", "iddpm", "edm"]
    assert schedule in ["vp", "ve", "linear"]
    assert scaling in ["vp", "none"]

    # Helper functions for VP & VE noise level schedules.
    def vp_sigma(beta_d, beta_min):
        return lambda t: (np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1) ** 0.5

    def vp_sigma_deriv(beta_d, beta_min):
        return lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))

    def vp_sigma_inv(beta_d, beta_min):
        return (
            lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )

    def ve_sigma(t):
        return t.sqrt()

    def ve_sigma_deriv(t):
        return 0.5 / t.sqrt()

    def ve_sigma_inv(sigma):
        return sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)

        def alpha_bar(j):
            return (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2

        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        assert discretization == "edm"
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == "linear"

        def sigma(t):
            return t

        def sigma_deriv(t):
            return 1

        def sigma_inv(sigma):
            return sigma

    # Define scaling schedule.
    if scaling == "vp":

        def s(t):
            return 1 / (1 + sigma(t) ** 2).sqrt()

        def s_deriv(t):
            return -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == "none"

        def s(t):
            return 1

        def s_deriv(t):
            return 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(torch.as_tensor(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    if replace_noise is None:
        noise_list = []

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    # 0, ..., N-1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= sigma(t_cur) <= S_max
            else 0
        )
        t_hat = sigma_inv(torch.as_tensor(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        if replace_noise is None:
            denoised = net(x_hat / s(t_hat), sigma(t_hat), condition).to(
                torch.float64
            )
            noise_list.append(denoised)
        else:
            if i >= 0 and i < 5:
                print("replace", i)
                denoised = replace_noise[i]

        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == "heun"
            denoised = net(x_prime / s(t_prime), sigma(t_prime), condition).to(
                torch.float64
            )
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    if replace_noise is None:
        # noise_list#.reverse()
        return x_next, noise_list
    else:
        return x_next
