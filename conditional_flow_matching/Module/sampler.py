import os
import torch
import torchdiffeq
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash

from conditional_flow_matching.Model.unet2d import MashUNet
from conditional_flow_matching.Model.mash_net import MashNet


class Sampler(object):
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
        self.context_dim = 512
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        self.use_ema = use_ema
        self.device = device

        model_id = 2
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
            print("[ERROR][MashSampler::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][MashSampler::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        ) -> np.ndarray: 
        self.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][Sampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, condition_tensor, t),
            torch.randn(condition_tensor.shape[0], 400, 25, device=self.device),
            torch.linspace(0, 1, 10, device=self.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        return traj.cpu().numpy()
