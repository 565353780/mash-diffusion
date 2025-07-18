import io
import os
import torch
import torchdiffeq
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from math import sqrt, ceil

from ma_sh.Model.mash import Mash

from mash_occ_decoder.Module.detector import Detector as OCCDetector

from wn_nc.Module.wnnc_reconstructor import WNNCReconstructor
from wn_nc.Module.mesh_smoother import MeshSmoother

from dino_v2_detect.Module.detector import Detector as DINODetector

from blender_manage.Module.blender_renderer import BlenderRenderer

from mash_diffusion.Model.cfm_latent_transformer import CFMLatentTransformer
from mash_diffusion.Model.cfm_hy3ddit import CFMHunyuan3DDiT
from mash_diffusion.Method.path import removeFile, createFileFolder


class CFMSampler(object):
    def __init__(
        self,
        model_file_path: str,
        dino_model_file_path: str,
        occ_model_file_path: Union[str, None] = None,
        cfm_use_ema: bool = True,
        occ_use_ema: bool = True,
        device: str = "cpu",
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
        self.anchor_num = 8192
        self.encoded_mash_channel = 23
        self.mask_degree = 2
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

        self.context_dim = 1024
        self.n_heads = 8
        self.d_head = 64
        self.depth = 8
        self.depth_single_blocks = 16

        self.use_ema = cfm_use_ema
        self.device = device

        model_id = 2
        if model_id == 1:
            self.model = CFMLatentTransformer(
                n_latents=self.anchor_num,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
            ).to(self.device)
        elif model_id == 2:
            self.model = CFMHunyuan3DDiT(
                n_latents=self.anchor_num,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
                depth_single_blocks=self.depth_single_blocks,
            ).to(self.device)

        self.loadModel(model_file_path)

        model_type = "large"
        dtype = "auto"
        self.dino_detector = DINODetector(
            model_type, dino_model_file_path, dtype, device
        )

        self.occ_detector = None
        if occ_model_file_path is not None:
            self.occ_detector = OCCDetector(
                model_file_path=occ_model_file_path,
                use_ema=occ_use_ema,
                batch_size=occ_batch_size,
                resolution=128,
                device=device,
            )

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

    def toInitialMashModel(self, device: Union[str, None] = None) -> Mash:
        if device is None:
            device = self.device

        mash_model = Mash(
            self.anchor_num,
            self.mask_degree,
            self.sh_degree,
            4,
            4,
            dtype=torch.float32,
            device=device,
        )
        return mash_model

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][CFMSampler::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device), weights_only=False
        )

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][CFMSampler::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    def getCondition(
        self,
        condition: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        condition_tensor = condition.type(torch.float32).to(self.device)

        if condition_tensor.ndim == 1:
            condition_tensor = condition_tensor.view(1, 1, -1)
        elif condition_tensor.ndim == 2:
            condition_tensor = condition_tensor.unsqueeze(1)
        elif condition_tensor.ndim == 4:
            condition_tensor = torch.squeeze(condition_tensor, dim=1)

        condition_tensor = condition_tensor.repeat(
            *([batch_size] + [1] * (condition_tensor.ndim - 1))
        )

        return condition_tensor

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: torch.Tensor,
        timestamp_num: int = 10,
        step_size: Union[float, None] = None,
    ) -> np.ndarray:
        self.model.eval()

        condition_tensor = self.getCondition(condition, sample_num)

        query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 0.5)

        x_init = torch.randn(
            condition_tensor.shape[0],
            self.model.n_latents,
            self.model.channels,
            device=self.device,
        )

        if step_size is None:
            method = "dopri5"
            ode_opts = None
        else:
            method = "euler"
            ode_opts = {"step_size": step_size}

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forwardData(x, condition_tensor, t),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method=method,
            options=ode_opts,
        )

        return traj.cpu().numpy()

    def samplePipeline(
        self,
        condition_image_file_path: str,
        save_folder_path: str,
        sample_num: int = 9,
        timestamp_num: int = 10,
        save_results_only: bool = True,
        step_size: Union[float, None] = None,
    ) -> bool:
        if not os.path.exists(condition_image_file_path):
            print("[ERROR][CFMSampler::samplePipeline]")
            print("\t condition image file not exist!")
            print("\t condition_image_file_path:", condition_image_file_path)
            return False

        os.makedirs(save_folder_path, exist_ok=True)

        condition = self.dino_detector.detectFile(condition_image_file_path)

        print("start diffuse", sample_num, "mashs....")
        sampled_array = self.sample(
            sample_num,
            condition,
            timestamp_num,
            step_size=step_size,
        )

        object_dist = [0, 0, 0]

        row_num = ceil(sqrt(sample_num))

        mash_model = self.toInitialMashModel()

        for j in range(sampled_array.shape[0]):
            if save_results_only:
                if j != sampled_array.shape[0] - 1:
                    continue

            current_save_folder_path = save_folder_path + "iter_" + str(j) + "/"

            os.makedirs(current_save_folder_path, exist_ok=True)

            current_save_mash_folder_path = current_save_folder_path + "mash/"
            current_save_pcd_folder_path = current_save_folder_path + "pcd/"
            current_save_wnnc_normal_folder_path = (
                current_save_folder_path + "wnnc_normal/"
            )
            current_save_wnnc_folder_path = current_save_folder_path + "wnnc/"
            current_save_wnnc_smooth_folder_path = (
                current_save_folder_path + "wnnc_smooth/"
            )
            current_save_occ_folder_path = current_save_folder_path + "occ/"
            current_save_occ_smooth_folder_path = (
                current_save_folder_path + "occ_smooth/"
            )
            current_save_render_pcd_folder_path = (
                current_save_folder_path + "render_pcd/"
            )
            current_save_render_wnnc_folder_path = (
                current_save_folder_path + "render_wnnc/"
            )
            current_save_render_wnnc_smooth_folder_path = (
                current_save_folder_path + "render_wnnc_smooth/"
            )
            current_save_render_occ_folder_path = (
                current_save_folder_path + "render_occ/"
            )
            current_save_render_occ_smooth_folder_path = (
                current_save_folder_path + "render_occ_smooth/"
            )

            print("start create mash files,", j + 1, "/", sampled_array.shape[0], "...")
            for i in tqdm(range(sample_num)):
                mash_params = sampled_array[j][i]

                sh2d = 2 * self.mask_degree + 1
                positions = mash_params[:, :3]
                ortho_poses = mash_params[:, 3:9]
                mask_params = mash_params[:, 9 : 9 + sh2d]
                sh_params = mash_params[:, 9 + sh2d :]

                mash_model.loadParams(
                    mask_params=mask_params,
                    sh_params=sh_params,
                    ortho_poses=ortho_poses,
                    positions=positions,
                )

                translate = [
                    int(i / row_num) * object_dist[0],
                    (i % row_num) * object_dist[1],
                    j * object_dist[2],
                ]

                mash_model.translate(translate)

                current_save_mash_file_path = (
                    current_save_mash_folder_path + "sample_" + str(i + 1) + "_mash.npy"
                )
                current_save_pcd_file_path = (
                    current_save_pcd_folder_path + "sample_" + str(i + 1) + "_pcd.ply"
                )
                current_save_wnnc_xyz_file_path = (
                    current_save_wnnc_normal_folder_path
                    + "sample_"
                    + str(i + 1)
                    + "_pcd.xyz"
                )
                current_save_wnnc_mesh_file_path = (
                    current_save_wnnc_folder_path + "sample_" + str(i + 1) + "_mesh.ply"
                )
                current_save_wnnc_smooth_mesh_file_path = (
                    current_save_wnnc_smooth_folder_path
                    + "sample_"
                    + str(i + 1)
                    + "_mesh.ply"
                )
                current_save_occ_mesh_file_path = (
                    current_save_occ_folder_path + "sample_" + str(i + 1) + "_mesh.ply"
                )
                current_save_occ_smooth_mesh_file_path = (
                    current_save_occ_smooth_folder_path
                    + "sample_"
                    + str(i + 1)
                    + "_mesh.ply"
                )

                mash_model.saveParamsFile(current_save_mash_file_path, True)
                # FIXME: for faster running only, you can activate this if you need to render results
                # it's the best way to save mash npy file only, and use follow script in ma-sh git package to visualize the result
                # python view.py <mash-npy-file-path>
                # mash_model.saveAsPcdFile(current_save_pcd_file_path, True)

                if self.recon_wnnc:
                    if os.path.exists(current_save_pcd_file_path):
                        tmp_xyz_file_path = "./output/tmp_pcd.xyz"
                        removeFile(tmp_xyz_file_path)
                        createFileFolder(tmp_xyz_file_path)
                        pcd = o3d.io.read_point_cloud(current_save_pcd_file_path)
                        o3d.io.write_point_cloud(
                            tmp_xyz_file_path, pcd, write_ascii=True
                        )
                        self.wnnc_reconstructor.autoReconstructSurface(
                            tmp_xyz_file_path,
                            current_save_wnnc_xyz_file_path,
                            current_save_wnnc_mesh_file_path,
                            width_tag="l1",
                            wsmin=0.01,
                            wsmax=0.04,
                            iters=40,
                            use_gpu=True,
                            print_progress=True,
                            overwrite=True,
                        )

                if self.recon_occ:
                    if os.path.exists(current_save_pcd_file_path):
                        if self.occ_detector is not None:
                            mesh = self.occ_detector.detectFile(
                                current_save_mash_file_path
                            )
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
                            overwrite=True,
                        )

                if self.smooth_occ:
                    if os.path.exists(current_save_occ_mesh_file_path):
                        self.mesh_smoother.smoothMesh(
                            current_save_occ_mesh_file_path,
                            current_save_occ_smooth_mesh_file_path,
                            n_iter=10,
                            pass_band=0.01,
                            edge_angle=15.0,
                            feature_angle=45.0,
                            overwrite=True,
                        )

                if self.blender_renderer.isValid():
                    overwrite = False

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
        return True

    def waitRender(self) -> bool:
        self.blender_renderer.waitWorkers()
        return True
