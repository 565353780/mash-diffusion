import os
import torch
import numpy as np
from typing import Union

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from ma_sh.Model.mash import Mash
from ma_sh.Method.random_mash import sampleRandomMashParams

from conditional_flow_matching.Model.unet2d import MashUNet
from conditional_flow_matching.Model.mash_net import MashNet
from conditional_flow_matching.Model.mash_latent_net import MashLatentNet

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        assert 'condition' in extras
        return self.model(x, extras['condition'], t)

class FMSampler(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.encoded_mash_channel = 25
        self.mask_degree = 3
        self.sh_degree = 2
        self.embed_dim = 256
        self.context_dim = 1024
        self.n_heads = 4
        self.d_head = 64
        self.depth = 24

        self.use_ema = use_ema
        self.device = device

        model_id = 3
        if model_id == 1:
            self.model = MashUNet(self.context_dim).to(self.device)
        elif model_id == 2:
            self.model = MashNet(
                n_latents=self.mash_channel,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth
            ).to(self.device)
        elif model_id == 3:
            self.model = MashLatentNet(
                n_latents=self.mash_channel,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                embed_dim=self.embed_dim,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth
            ).to(self.device)

        self.wrapped_model = WrappedModel(self.model)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def toInitialMashModel(self, device: Union[str, None]=None) -> Mash:
        if device is None:
            device = self.device

        mash_model = Mash(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            20,
            800,
            0.4,
            dtype=torch.float64,
            device=device,
        )
        return mash_model

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            print("[ERROR][Sampler::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        if self.use_ema:
            self.wrapped_model.model.load_state_dict(model_dict["ema_model"])
        else:
            self.wrapped_model.model.load_state_dict(model_dict["model"])

        print("[INFO][Sampler::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        timestamp_num: int = 20,
        ) -> np.ndarray:
        self.wrapped_model.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][Sampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        step_size = 1.0 / timestamp_num

        T = torch.linspace(0,1,timestamp_num).to(self.device)

        x_init = sampleRandomMashParams(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            sample_num,
            'cpu',
            'randn',
            False).type(torch.float32).to(self.device)

        solver = ODESolver(velocity_model=self.wrapped_model)
        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method='midpoint',
            step_size=step_size,
            return_intermediates=True,
            condition=condition_tensor,
        )

        return sol.cpu().numpy()
