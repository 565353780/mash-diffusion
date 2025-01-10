import torch
import torchdiffeq
import numpy as np
from torch import nn
from typing import Union

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer
from mash_diffusion.Module.base_diffusion_trainer import BaseDiffusionTrainer
from mash_diffusion.Module.batch_ot_cfm import BatchExactOptimalTransportConditionalFlowMatcher
from mash_diffusion.Module.stacked_random_generator import StackedRandomGenerator


class CFMTrainer(BaseDiffusionTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        training_mode: str = 'dino',
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "cuda:0",
        dtype = torch.float32,
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
        fm_id = 2
        if fm_id == 1:
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        elif fm_id == 2:
            self.FM = BatchExactOptimalTransportConditionalFlowMatcher(
                sigma=0.0, target_dim=None
            )
        elif fm_id == 3:
            self.FM = AffineProbPath(scheduler=CondOTScheduler())

        super().__init__(
            dataset_root_folder_path,
            training_mode,
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
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
            self.model = MashUNet(
                self.context_dim
            ).to(self.device, dtype=self.dtype)
        elif model_id == 2:
            self.model = CFMLatentTransformer(
                n_latents=self.anchor_num,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
            ).to(self.device, dtype=self.dtype)
        return True

    def preProcessDiffusionData(self, data_dict: dict, is_training: bool = False) -> dict:
        mash_params = data_dict["mash_params"]

        init_mash_params = torch.randn_like(mash_params)

        if is_training and self.fix_params:
            fixed_prob = 2.0 * np.random.rand() - 1.0
            fixed_prob = max(fixed_prob, 0.0)

            if fixed_prob > 0:
                fixed_mask = torch.rand_like(mash_params) <= fixed_prob
                init_mash_params[fixed_mask] = mash_params[fixed_mask]

        if isinstance(self.FM, ExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_mash_params, mash_params
            )
        elif isinstance(self.FM, BatchExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_mash_params, mash_params
            )
        elif isinstance(self.FM, AffineProbPath):
            t = torch.rand(mash_params.shape[0]).to(mash_params.device, dtype=mash_params.dtype)
            t = torch.pow(t, 0.5)
            path_sample = self.FM.sample(t=t, x_0=init_mash_params, x_1=mash_params)
            t = path_sample.t
            xt = path_sample.x_t
            ut = path_sample.dx_t
        else:
            print("[ERROR][CFMTrainer::preProcessDiffusionData]")
            print("\t FM not valid!")
            exit()

        data_dict["ut"] = ut
        data_dict["t"] = t
        data_dict["xt"] = xt

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        ut = data_dict["ut"]
        vt = result_dict["vt"]

        loss = torch.pow(vt - ut, 2).mean()

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleMashData(self, model: nn.Module, condition: torch.Tensor, sample_num: int) -> torch.Tensor:
        timestamp_num = 2

        query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 1.0 / 2.0)

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        x_init = rnd.randn([sample_num, self.anchor_num, self.anchor_channel], device=self.device)

        traj = torchdiffeq.odeint(
            lambda t, x: model.forwardData(x, condition, t),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        sampled_array = traj.cpu()[-1]

        return sampled_array
