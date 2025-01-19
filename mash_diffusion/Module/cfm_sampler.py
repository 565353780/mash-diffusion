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
from ma_sh.Method.io import loadMashFileParamsTensor
from ma_sh.Method.transformer import getTransformer
from ma_sh.Module.local_editor import LocalEditor

from mash_occ_decoder.Module.detector import Detector as OCCDetector

from wn_nc.Module.wnnc_reconstructor import WNNCReconstructor
from wn_nc.Module.mesh_smoother import MeshSmoother

from ulip_manage.Module.detector import Detector as ULIPDetector

from dino_v2_detect.Module.detector import Detector as DINODetector

from blender_manage.Module.blender_renderer import BlenderRenderer

from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer
from mash_diffusion.Method.path import removeFile, createFileFolder


class CFMSampler(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        occ_model_file_path: Union[str, None] = None,
        cfm_use_ema: bool = True,
        occ_use_ema: bool = True,
        device: str = "cpu",
        transformer_id: str = 'Objaverse_82K',
        ulip_model_file_path: Union[str, None] = None,
        open_clip_model_file_path: Union[str, None] = None,
        dino_model_file_path: Union[str, None] = None,
        occ_batch_size: int = 1200000,
        recon_wnnc: bool = True,
        recon_occ: bool = True,
        smooth_wnnc: bool = True,
        smooth_occ: bool = True,
        render_pcd: bool = True,
        render_wnnc: bool = True,
        render_wnnc_smooth: bool = True,
        render_occ: bool = True,
        render_occ_smooth: bool = True,

    ) -> None:
        self.mash_channel = 400
        self.encoded_mash_channel = 25
        self.mask_degree = 3
        self.sh_degree = 2

        self.recon_wnnc = recon_wnnc
        self.recon_occ = recon_occ
        self.smooth_wnnc = smooth_wnnc
        self.smooth_occ = smooth_occ
        self.render_pcd = render_pcd
        self.render_wnnc = render_wnnc
        self.render_wnnc_smooth = render_wnnc_smooth
        self.render_occ = render_occ
        self.render_occ_smooth = render_occ_smooth

        if transformer_id in ['ShapeNet', 'ShapeNet_03001627']:
            self.context_dim = 512
            self.n_heads = 8
            self.d_head = 64
            self.depth = 24
        elif transformer_id == 'Objaverse_82K':
            self.context_dim = 768
            self.n_heads = 8
            self.d_head = 64
            self.depth = 24

        self.use_ema = cfm_use_ema
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
            self.dino_detector = DINODetector('base', dino_model_file_path, 'auto', device)

        self.occ_detector = None
        if occ_model_file_path is not None:
            self.occ_detector = OCCDetector(
                model_file_path=occ_model_file_path,
                use_ema=occ_use_ema,
                batch_size=occ_batch_size,
                resolution=128,
                transformer_id='Objaverse_82K',
                device=device)

        self.wnnc_reconstructor = WNNCReconstructor()
        self.mesh_smoother = MeshSmoother()

        self.blender_renderer = BlenderRenderer(
            workers_per_cpu=4,
            workers_per_gpu=0,
            is_background=True,
            mute=True,
            gpu_id_list=[0],
        )
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

    def getCondition(self,
                     condition: Union[int, np.ndarray, torch.Tensor],
                     batch_size: int = 1,
                     ) -> Union[torch.Tensor, None]:
        if isinstance(condition, int):
            condition_tensor = torch.ones([batch_size]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device)

            if condition_tensor.ndim == 1:
                condition_tensor = condition_tensor.view(1, 1, -1)
            elif condition_tensor.ndim == 2:
                condition_tensor = condition_tensor.unsqueeze(1)
            elif condition_tensor.ndim == 4:
                condition_tensor = torch.squeeze(condition_tensor, dim=1)

            condition_tensor = condition_tensor.repeat(*([batch_size] + [1] * (condition_tensor.ndim - 1)))
        elif isinstance(condition, torch.Tensor):
            condition_tensor = condition.type(torch.float32).to(self.device)

            if condition_tensor.ndim == 1:
                condition_tensor = condition_tensor.view(1, 1, -1)
            elif condition_tensor.ndim == 2:
                condition_tensor = condition_tensor.unsqueeze(1)
            elif condition_tensor.ndim == 4:
                condition_tensor = torch.squeeze(condition_tensor, dim=1)

            condition_tensor = condition_tensor.repeat(*([batch_size] + [1] * (condition_tensor.ndim - 1)))
        else:
            print('[ERROR][CFMSampler::getCondition]')
            print('\t condition type not valid!')
            return None

        return condition_tensor

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray, torch.Tensor] = 0,
        timestamp_num: int = 10,
        ) -> np.ndarray:
        self.model.eval()

        condition_tensor = self.getCondition(condition, sample_num)

        if condition_tensor is None:
            print('[ERROR][CFMSampler::sample]')
            print('\t getCondition failed!')
            return np.ndarray([])

        query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 0.5)

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

        condition_tensor = self.getCondition(condition, sample_num)

        if condition_tensor is None:
            print('[ERROR][CFMSampler::sampleWithFixedAnchors]')
            print('\t getCondition failed!')
            return np.ndarray([])

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

        mash_params_list = []
        for mash_file_path in mash_file_path_list:
            mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')
            mash_params_list.append(mash_params)

        fixed_mash_params = torch.cat(mash_params_list, dim=0).view(1, -1, 25)
        fixed_mash_params = fixed_mash_params.repeat(*([condition_tensor.shape[0]] + [1] * (fixed_mash_params.ndim - 1)))

        fixed_x_init = self.transformer.transform(fixed_mash_params).to(self.device)

        random_x_init = torch.randn(condition_tensor.shape[0], 400 - fixed_x_init.shape[1], 25, device=self.device)

        x_init = torch.cat((fixed_x_init, random_x_init), dim=1)

        fixed_anchor_mask = torch.zeros_like(x_init, dtype=torch.bool)
        fixed_anchor_mask[:, :fixed_x_init.shape[1], :] = True

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

            if condition_type in ['dino', 'ulip-image']:
                copyfile(image_file_path, current_save_folder_path + 'condition_image.' + image_file_path.split('.')[-1])
            elif condition_type == 'ulip-points':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(current_save_folder_path + 'condition_pcd.ply', pcd)
            elif condition_type == 'ulip-text':
                with open(current_save_folder_path + 'condition_text.txt', 'w') as f:
                    f.write(text)

            current_save_mash_folder_path = current_save_folder_path + 'mash/'
            current_save_pcd_folder_path = current_save_folder_path + 'pcd/'
            current_save_wnnc_normal_folder_path = current_save_folder_path + 'wnnc_normal/'
            current_save_wnnc_folder_path = current_save_folder_path + 'wnnc/'
            current_save_wnnc_smooth_folder_path = current_save_folder_path + 'wnnc_smooth/'
            current_save_occ_folder_path = current_save_folder_path + 'occ/'
            current_save_occ_smooth_folder_path = current_save_folder_path + 'occ_smooth/'
            current_save_render_pcd_folder_path = current_save_folder_path + 'render_pcd/'
            current_save_render_wnnc_folder_path = current_save_folder_path + 'render_wnnc/'
            current_save_render_wnnc_smooth_folder_path = current_save_folder_path + 'render_wnnc_smooth/'
            current_save_render_occ_folder_path = current_save_folder_path + 'render_occ/'
            current_save_render_occ_smooth_folder_path = current_save_folder_path + 'render_occ_smooth/'

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

                current_save_mash_file_path = current_save_mash_folder_path + 'sample_' + str(i+1) + '_mash.npy'
                current_save_pcd_file_path = current_save_pcd_folder_path + 'sample_' + str(i+1) + '_pcd.ply'
                current_save_wnnc_xyz_file_path = current_save_wnnc_normal_folder_path + 'sample_' + str(i+1) + '_pcd.xyz'
                current_save_wnnc_mesh_file_path = current_save_wnnc_folder_path + 'sample_' + str(i+1) + '_mesh.ply'
                current_save_wnnc_smooth_mesh_file_path = current_save_wnnc_smooth_folder_path + 'sample_' + str(i+1) + '_mesh.ply'
                current_save_occ_mesh_file_path = current_save_occ_folder_path + 'sample_' + str(i+1) + '_mesh.ply'
                current_save_occ_smooth_mesh_file_path = current_save_occ_smooth_folder_path + 'sample_' + str(i+1) + '_mesh.ply'

                mash_model.saveParamsFile(current_save_mash_file_path, True)
                mash_model.saveAsPcdFile(current_save_pcd_file_path, True)

                if self.recon_wnnc:
                    if os.path.exists(current_save_pcd_file_path):
                        tmp_xyz_file_path = './output/tmp_pcd.xyz'
                        removeFile(tmp_xyz_file_path)
                        createFileFolder(tmp_xyz_file_path)
                        pcd = o3d.io.read_point_cloud(current_save_pcd_file_path)
                        o3d.io.write_point_cloud(tmp_xyz_file_path, pcd, write_ascii=True)
                        self.wnnc_reconstructor.autoReconstructSurface(
                            tmp_xyz_file_path,
                            current_save_wnnc_xyz_file_path,
                            current_save_wnnc_mesh_file_path,
                            width_tag='l1',
                            wsmin=0.01,
                            wsmax=0.04,
                            iters=40,
                            use_gpu=True,
                            print_progress=True,
                            overwrite=True)

                if self.recon_occ:
                    if os.path.exists(current_save_pcd_file_path):
                        if self.occ_detector is not None:
                            mesh = self.occ_detector.detectFile(current_save_mash_file_path)
                            createFileFolder(current_save_occ_mesh_file_path)
                            mesh.export(current_save_occ_mesh_file_path)

                if self.smooth_wnnc:
                    if os.path.exists(current_save_wnnc_mesh_file_path):
                        self.mesh_smoother.smoothMesh(
                            current_save_wnnc_mesh_file_path,
                            current_save_wnnc_smooth_mesh_file_path,
                            n_iter=10,
                            pass_band=0.01,
                            edge_angle=15.0,
                            feature_angle=45.0,
                            overwrite=True)

                if self.smooth_occ:
                    if os.path.exists(current_save_occ_mesh_file_path):
                        self.mesh_smoother.smoothMesh(
                            current_save_occ_mesh_file_path,
                            current_save_occ_smooth_mesh_file_path,
                            n_iter=10,
                            pass_band=0.01,
                            edge_angle=15.0,
                            feature_angle=45.0,
                            overwrite=True)

                if self.blender_renderer.isValid():
                    overwrite = False

                    if condition_type == 'dino':
                        render_image_num = 12

                        if self.render_pcd:
                            if os.path.exists(current_save_pcd_folder_path):
                                self.blender_renderer.renderAroundFile(
                                    shape_file_path=current_save_pcd_file_path,
                                    render_image_num=render_image_num,
                                    save_image_folder_path=current_save_render_pcd_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_wnnc:
                            if os.path.exists(current_save_wnnc_folder_path):
                                self.blender_renderer.renderAroundFile(
                                    shape_file_path=current_save_wnnc_mesh_file_path,
                                    render_image_num=render_image_num,
                                    save_image_folder_path=current_save_render_wnnc_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_wnnc_smooth:
                            if os.path.exists(current_save_wnnc_smooth_folder_path):
                                self.blender_renderer.renderAroundFile(
                                    shape_file_path=current_save_wnnc_smooth_mesh_file_path,
                                    render_image_num=render_image_num,
                                    save_image_folder_path=current_save_render_wnnc_smooth_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_occ:
                            if os.path.exists(current_save_occ_folder_path):
                                self.blender_renderer.renderAroundFile(
                                    shape_file_path=current_save_occ_mesh_file_path,
                                    render_image_num=render_image_num,
                                    save_image_folder_path=current_save_render_occ_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_occ_smooth:
                            if os.path.exists(current_save_occ_smooth_folder_path):
                                self.blender_renderer.renderAroundFile(
                                    shape_file_path=current_save_occ_smooth_mesh_file_path,
                                    render_image_num=render_image_num,
                                    save_image_folder_path=current_save_render_occ_smooth_folder_path,
                                    overwrite=overwrite,
                                )
                    else:
                        if self.render_pcd:
                            if os.path.exists(current_save_pcd_folder_path):
                                self.blender_renderer.renderFile(
                                    shape_file_path=current_save_pcd_file_path,
                                    save_image_file_path=current_save_render_pcd_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_wnnc:
                            if os.path.exists(current_save_wnnc_folder_path):
                                self.blender_renderer.renderFile(
                                    shape_file_path=current_save_wnnc_mesh_file_path,
                                    save_image_file_path=current_save_render_wnnc_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_wnnc_smooth:
                            if os.path.exists(current_save_wnnc_smooth_folder_path):
                                self.blender_renderer.renderFile(
                                    shape_file_path=current_save_wnnc_smooth_mesh_file_path,
                                    save_image_file_path=current_save_render_wnnc_smooth_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_occ:
                            if os.path.exists(current_save_occ_folder_path):
                                self.blender_renderer.renderFile(
                                    shape_file_path=current_save_occ_mesh_file_path,
                                    save_image_file_path=current_save_render_occ_folder_path,
                                    overwrite=overwrite,
                                )

                        if self.render_occ_smooth:
                            if os.path.exists(current_save_occ_smooth_folder_path):
                                self.blender_renderer.renderFile(
                                    shape_file_path=current_save_occ_smooth_mesh_file_path,
                                    save_image_file_path=current_save_render_occ_smooth_folder_path,
                                    overwrite=overwrite,
                                )

        return True

    def waitRender(self) -> bool:
        self.blender_renderer.waitWorkers()
        return True
