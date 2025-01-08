import os
import torch
import torchdiffeq
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from math import sqrt, ceil
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Method.transformer import getTransformer
from ma_sh.Module.local_editor import LocalEditor

from ulip_manage.Module.detector import Detector as ULIPDetector
from dino_v2_detect.Module.detector import Detector as DINODetector

from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer


class CFMSampler(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        device: str = "cpu",
        transformer_id: str = 'Objaverse_82K',
        ulip_model_file_path: Union[str, None] = None,
        open_clip_model_file_path: Union[str, None] = None,
        dino_model_file_path: Union[str, None] = None,
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

        self.transformer = getTransformer(transformer_id)
        assert self.transformer is not None

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

        self.ulip_detector = None
        if ulip_model_file_path is not None and open_clip_model_file_path is not None:
            self.ulip_detector = ULIPDetector(ulip_model_file_path, open_clip_model_file_path, device)

        self.dino_detector = None
        if dino_model_file_path is not None:
            self.dino_detector = DINODetector('large', dino_model_file_path, device)
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

        fixed_x_init = self.transformer.transform(fixed_x_init)

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

    def samplePipeline(
        self,
        save_folder_path: str,
        condition_type: str = 'category',
        condition_value: Union[int, str, np.ndarray] = 18,
        sample_num: int = 9,
        timestamp_num: int = 10,
        save_results_only: bool = True,
        mash_file_path_list: Union[list, None]=None,
    ) -> bool:
        assert condition_type in ['category', 'dino', 'ulip-image', 'ulip-points', 'ulip-text']

        if condition_type == 'category':
            assert isinstance(condition_value, int)

            condition = condition_value
        elif condition_type == 'dino':
            assert self.dino_detector is not None
            assert isinstance(condition_value, str)

            image_file_path = condition_value
            if not os.path.exists(image_file_path):
                print('[ERROR][CFMSampler::demoCondition]')
                print('\t condition image file not exist!')
                return False

            condition = self.dino_detector.detectFile(image_file_path)
        elif condition_type == 'ulip-image':
            assert self.ulip_detector is not None
            assert isinstance(condition_value, str)

            image_file_path = condition_value
            if not os.path.exists(image_file_path):
                print('[ERROR][CFMSampler::demoCondition]')
                print('\t condition image file not exist!')
                return False

            condition = (
                self.ulip_detector.encodeImageFile(image_file_path).cpu().numpy()
            )
        elif condition_type == 'ulip-points':
            assert self.ulip_detector is not None
            assert isinstance(condition_value, np.ndarray)

            points = condition_value
            condition = (
                self.ulip_detector.encodePointCloud(points).cpu().numpy()
            )
        elif condition_type == 'ulip-text':
            assert self.ulip_detector is not None
            assert isinstance(condition_value, str)

            text = condition_value
            condition = (
                self.ulip_detector.encodeText(condition_value).cpu().numpy()
            )
        else:
            print('[ERROR][CFMSampler::demoCondition]')
            print('\t condition type not valid!')
            return False

        print("start diffuse", sample_num, "mashs....")
        if mash_file_path_list is None:
            sampled_array = self.sample(sample_num, condition, timestamp_num)
        else:
            sampled_array = self.sampleWithFixedAnchors(mash_file_path_list, sample_num, condition, timestamp_num)

        object_dist = [0, 0, 0]

        row_num = ceil(sqrt(sample_num))

        mash_model = self.toInitialMashModel()

        for j in range(sampled_array.shape[0]):
            if save_results_only:
                if j != sampled_array.shape[0] - 1:
                    continue

            current_save_folder_path = save_folder_path + 'iter_' + str(j) + '/'

            os.makedirs(current_save_folder_path, exist_ok=True)

            if condition_type == 'image':
                copyfile(image_file_path, current_save_folder_path + 'condition_image.png')
            elif condition_type == 'points':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(current_save_folder_path + 'condition_pcd.ply', pcd)
            elif condition_type == 'text':
                with open(current_save_folder_path + 'condition_text.txt', 'w') as f:
                    f.write(text)

            print("start create mash files,", j + 1, '/', sampled_array.shape[0], "...")
            for i in tqdm(range(sample_num)):

                mash_params = sampled_array[j][i]

                mash_params = self.transformer.inverse_transform(mash_params)

                sh2d = 2 * self.mask_degree + 1
                ortho_poses = mash_params[:, :6]
                positions = mash_params[:, 6:9]
                mask_params = mash_params[:, 9 : 9 + sh2d]
                sh_params = mash_params[:, 9 + sh2d :]

                mash_model.loadParams(
                    mask_params=mask_params,
                    sh_params=sh_params,
                    positions=positions,
                    ortho6d_poses=ortho_poses
                )

                translate = [
                    int(i / row_num) * object_dist[0],
                    (i % row_num) * object_dist[1],
                    j * object_dist[2],
                ]

                mash_model.translate(translate)

                mash_model.saveParamsFile(current_save_folder_path + 'mash/sample_' + str(i+1) + '_mash.npy', True)
                mash_model.saveAsPcdFile(current_save_folder_path + 'pcd/sample_' + str(i+1) + '_pcd.ply', True)

        return True
