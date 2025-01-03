import os
import torch
import torchdiffeq
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Module.local_editor import LocalEditor

from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer


class CFMSampler(object):
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
            self.model = CFMLatentTransformer(
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

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][CFMSampler::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][CFMSampler::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        timestamp_num: int = 10,
        ) -> np.ndarray:
        self.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][CFMSampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        query_t = torch.linspace(0,1,timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 1.0 / 2.0)

        x_init = torch.randn(condition_tensor.shape[0], 400, 25, device=self.device)

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forwardData(x, condition_tensor, t),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        return traj.cpu().numpy()

    @torch.no_grad()
    def sampleWithFixedAnchors(
        self,
        mash_file_path_list: list,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        timestamp_num: int = 10,
    ) -> Union[np.ndarray, None]:
        self.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][CFMSampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        query_t = torch.linspace(0,1,timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 1.0 / 2.0)

        '''
        local_editor = LocalEditor(self.device)
        if not local_editor.loadMashFiles(mash_file_path_list):
            print('[ERROR][CFMSampler::sampleWithFixedAnchors]')
            print('\t loadMashFiles failed!')
            return None

        combined_mash = local_editor.toCombinedMash()
        if combined_mash is None:
            print('[ERROR][CFMSampler::sampleWithFixedAnchors]')
            print('\t toCombinedMash failed!')
            return None
        '''
        combined_mash = Mash.fromParamsFile(
            mash_file_path_list[0],
            10,
            10,
            1.0,
            torch.int64,
            torch.float64,
            self.device,
        )

        fixed_ortho_poses = combined_mash.toOrtho6DPoses().detach().clone().float()
        fixed_positions = combined_mash.positions.detach().clone().float()
        fixed_mask_params = combined_mash.mask_params.detach().clone().float()
        fixed_sh_params = combined_mash.sh_params.detach().clone().float()

        fixed_x_init = torch.cat((
            fixed_ortho_poses,
            fixed_positions,
            fixed_mask_params,
            fixed_sh_params,
        ), dim=1).view(1, combined_mash.anchor_num, 25).expand(condition_tensor.shape[0], combined_mash.anchor_num, 25)

        random_x_init = torch.randn(condition_tensor.shape[0], 400 - combined_mash.anchor_num, 25, device=self.device)

        x_init = torch.cat((fixed_x_init, random_x_init), dim=1)

        fixed_anchor_mask = torch.zeros_like(x_init, dtype=torch.bool)
        fixed_anchor_mask[:, :combined_mash.anchor_num, :] = True

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forwardWithFixedAnchors(x, condition_tensor, t, fixed_anchor_mask),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        return traj.cpu().numpy()
