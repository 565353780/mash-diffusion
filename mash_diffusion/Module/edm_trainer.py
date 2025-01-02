import torch
import numpy as np
from torch import nn
from typing import Union

from mash_diffusion.Loss.edm import EDMLoss
from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.edm_latent_transformer import EDMLatentTransformer
from mash_diffusion.Module.base_diffusion_trainer import BaseDiffusionTrainer
from mash_diffusion.Module.stacked_random_generator import StackedRandomGenerator
from mash_diffusion.Method.sample import edm_sampler


class EDMTrainer(BaseDiffusionTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        dataset_json_file_path_dict: dict = {},
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "cuda:0",
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
    ) -> None:
        self.context_dim = 1024
        self.n_heads = 16
        self.d_head = 64
        self.depth = 24

        self.fix_params = False

        self.loss_func = EDMLoss()

        super().__init__(
            dataset_root_folder_path,
            dataset_json_file_path_dict,
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            device,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
        )
        return

    def createModel(self) -> bool:
        model_id = 2
        if model_id == 1:
            self.model = MashUNet(self.context_dim).to(self.device)
        elif model_id == 2:
            self.model = EDMLatentTransformer(
                n_latents=self.anchor_num,
                channels=self.anchor_channel,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
                context_dim=self.context_dim,
            ).to(self.device)
        return True

    def preProcessDiffusionData(self, data_dict: dict, is_training: bool = False) -> dict:
        mash_params = data_dict["mash_params"]

        noise, sigma, weight = self.loss_func(mash_params, not is_training)

        data_dict['noise'] = noise
        data_dict['sigma'] = sigma
        data_dict['weight'] = weight

        if is_training and self.fix_params:
            fixed_prob = 2.0 * np.random.rand() - 1.0
            fixed_prob = max(fixed_prob, 0.0)
            data_dict['fixed_prob'] = fixed_prob
        else:
            data_dict['fixed_prob'] = 0.0

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        inputs = data_dict['mash_params']
        D_yn = result_dict['D_x']
        weight = data_dict['weight']

        loss = weight * ((D_yn - inputs) ** 2)

        loss = loss.mean()

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleMashData(self, model: nn.Module, condition: torch.Tensor, sample_num: int) -> torch.Tensor:
        timestamp_num = 18

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(condition.device, batch_seeds)
        latents = rnd.randn([sample_num, self.anchor_num, self.anchor_channel], device=condition.device)


        sampled_array = edm_sampler(
            model,
            latents,
            condition,
            randn_like=rnd.randn_like,
            num_steps=timestamp_num,
            fixed_mask=None)[-1]

        return sampled_array
