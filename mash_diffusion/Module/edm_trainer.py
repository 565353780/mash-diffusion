import torch
from torch import nn
from typing import Union

from base_diffusion_trainer.Module.base_edm_trainer import BaseEDMTrainer

from mash_diffusion.Loss.edm import EDMLoss
from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.edm_latent_transformer import EDMLatentTransformer
from mash_diffusion.Module.common_func import CommonFunc


class EDMTrainer(CommonFunc, BaseEDMTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
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
        quick_test: bool = False,
    ) -> None:
        self.loss_func = EDMLoss()

        CommonFunc.__init__(self, dataset_root_folder_path)

        BaseEDMTrainer.__init__(
            self,
            dataset_root_folder_path,
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
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
            quick_test,
        )
        return

    def createModel(self) -> bool:
        model_id = 2
        if model_id == 1:
            self.model = MashUNet(self.context_dim).to(self.device, dtype=self.dtype)
        elif model_id == 2:
            self.model = EDMLatentTransformer(
                n_latents=self.anchor_num,
                channels=self.anchor_channel,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
                context_dim=self.context_dim,
            ).to(self.device, dtype=self.dtype)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        data_dict = CommonFunc.preProcessData(self, data_dict, is_training)
        data_dict = self.preProcessDiffusionData(data_dict, 'mash_params', is_training)
        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        loss_diffusion = self.getDiffusionLossDict(data_dict, 'mash_params', result_dict)

        loss_dict = {
            "Loss": loss_diffusion,
        }

        return loss_dict

    @torch.no_grad()
    def sampleMashData(
        self, model: nn.Module, condition: torch.Tensor, sample_num: int
    ) -> torch.Tensor:
        timestamp_num = 2

        data_shape = [sample_num, self.anchor_num, self.anchor_channel]

        sampled_array = self.sampleData(
            model,
            condition,
            data_shape,
            sample_num,
            timestamp_num,
        )

        return sampled_array
