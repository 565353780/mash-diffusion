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
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.encoded_mash_channel = 25
        self.mask_degree = 3
        self.sh_degree = 2
        self.context_dim = 768
        self.n_heads = 8
        self.d_head = 64
        self.depth = 12
        self.device = device

        model_id = 2
        if model_id == 1:
            self.model = MashUNet(self.context_dim).to(self.device)
        elif model_id == 2:
            self.model = MashNet(n_latents=self.mash_channel, mask_degree=self.mask_degree, sh_degree=self.sh_degree, context_dim=self.context_dim,n_heads=self.n_heads, d_head=self.d_head,depth=self.depth).to(self.device)

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
            10,
            400,
            0.4,
            dtype=torch.float64,
            device=device,
        )
        return mash_model

    def loadModel(self, model_file_path, resume_model_only=True):
        if not os.path.exists(model_file_path):
            print("[ERROR][MashSampler::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        self.model.load_state_dict(model_dict["model"])

        if not resume_model_only:
            # self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        print("[INFO][MashSampler::loadModel]")
        print("\t load model success!")
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        category_id: int = 0,
        ) -> np.ndarray: 
        self.model.eval()

        condition = torch.ones([sample_num]).long().to(self.device) * category_id

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, condition, t),
            torch.randn(condition.shape[0], 400, 25, device=self.device),
            torch.linspace(0, 1, 10, device=self.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        return traj.cpu().numpy()
