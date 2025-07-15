import MFSClient

import os
import torch
from torch import nn
from tqdm import trange
from typing import Union
from abc import abstractmethod

from ma_sh.Model.mash import Mash

from base_trainer.Module.base_trainer import BaseTrainer

from dino_v2_detect.Module.detector import Detector as DINODetector

from mash_diffusion.Dataset.image import ImageDataset
from mash_diffusion.Dataset.tos_image import TOSImageDataset


class BaseDiffusionTrainer(BaseTrainer):
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
        self.dataset_root_folder_path = dataset_root_folder_path

        self.anchor_num = 8192
        self.mask_degree = 2
        self.sh_degree = 2

        self.context_dim = 1024
        self.n_heads = 8  # 16
        self.d_head = 64
        self.depth = 8  # 16
        self.depth_single_blocks = 16  # 32

        self.gt_sample_added_to_logger = False

        mask_dim = 2 * self.mask_degree + 1
        sh_dim = (self.sh_degree + 1) ** 2
        self.anchor_channel = 9 + mask_dim + sh_dim

        super().__init__(
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

    def createDatasets(self) -> bool:
        model_type = "large"
        model_file_path = "./data/dinov2_vitl14_reg4_pretrain.pth"
        dtype = "auto"

        if not os.path.exists(model_file_path):
            print("[ERROR][BaseDiffusionTrainer::createDatasets]")
            print("\t DINOv2 model not found!")
            print("\t model_file_path:", model_file_path)
            exit()

        self.dino_detector = DINODetector(
            model_type, model_file_path, dtype, self.device
        )

        use_tos = True
        # FIXME: skip eval for faster training on slurm
        eval = False
        if use_tos:
            client = MFSClient.MFSClient2()

            paths_file_path = "./data/tos_paths_list.npy"

            self.dataloader_dict["dino"] = {
                "dataset": TOSImageDataset(
                    client,
                    mash_bucket="mm-data-general-model-trellis",
                    mash_folder_key="mash/",
                    image_bucket="mm-data-general-model-v1",
                    image_folder_key="rendering/orient_cam72_base/",
                    transform=self.dino_detector.transform,
                    split="train",
                    dtype=self.dtype,
                    paths_file_path=paths_file_path,
                ),
                "repeat_num": 1,
            }

            if eval:
                self.dataloader_dict["eval"] = {
                    "dataset": TOSImageDataset(
                        client,
                        mash_bucket="mm-data-general-model-trellis",
                        mash_folder_key="mash/",
                        image_bucket="mm-data-general-model-v1",
                        image_folder_key="rendering/orient_cam72_base/",
                        transform=self.dino_detector.transform,
                        split="eval",
                        dtype=self.dtype,
                        paths_file_path=paths_file_path,
                    ),
                }
        else:
            self.dataloader_dict["dino"] = {
                "dataset": ImageDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/manifold_mash",
                    "Objaverse_82K/render_jpg_v2",
                    self.dino_detector.transform,
                    "train",
                    self.dtype,
                ),
                "repeat_num": 1,
            }

            if eval:
                self.dataloader_dict["eval"] = {
                    "dataset": ImageDataset(
                        self.dataset_root_folder_path,
                        "Objaverse_82K/manifold_mash",
                        "Objaverse_82K/render_jpg_v2",
                        self.dino_detector.transform,
                        "eval",
                        self.dtype,
                    ),
                }

        if "eval" in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:4]

        return True

    def getCondition(self, data_dict: dict) -> dict:
        if "category_id" in data_dict.keys():
            data_dict["condition"] = data_dict["category_id"]
        elif "image" in data_dict.keys():
            image = data_dict["image"]
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(self.device)

            dino_feature = self.dino_detector.detect(image)

            data_dict["condition"] = dino_feature
        elif "embedding" in data_dict.keys():
            embedding = data_dict["embedding"]

            if embedding.ndim == 1:
                embedding = embedding.view(1, 1, -1)
            if embedding.ndim == 2:
                embedding = embedding.unsqueeze(1)
            elif embedding.ndim == 4:
                embedding = torch.squeeze(embedding, dim=1)

            data_dict["condition"] = embedding.to(self.device)
        else:
            print("[ERROR][BaseDiffusionTrainer::getCondition]")
            print("\t valid condition type not found!")
            exit()

        return data_dict

    @abstractmethod
    def preProcessDiffusionData(
        self, data_dict: dict, is_training: bool = False
    ) -> dict:
        """
        if is_training:
            data_dict[new_name] = new_value
        return data_dict
        """
        pass

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        data_dict = self.getCondition(data_dict)

        if is_training:
            data_dict["drop_prob"] = 0.1
        else:
            data_dict["drop_prob"] = 0.0

        data_dict = self.preProcessDiffusionData(data_dict, is_training)

        return data_dict

    @abstractmethod
    @torch.no_grad()
    def sampleMashData(
        self, model: nn.Module, condition: torch.Tensor, sample_num: int
    ) -> torch.Tensor:
        """
        mash_params = sample_func(model, condition, sample_num)
        return mash_params
        """
        pass

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        # FIXME: skip this since it will occur NCCL error
        return True

        sample_num = 3
        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)
        data_dict = self.getCondition(data_dict)

        condition = data_dict["condition"]

        if isinstance(condition, int):
            condition = torch.ones([sample_num]).long().to(self.device) * condition
        else:
            condition = (
                condition.type(self.dtype)
                .to(self.device)
                .repeat(*([sample_num] + [1] * (condition.ndim - 1)))
            )

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample", sample_num, "mashs....")

        sampled_array = self.sampleMashData(model, condition, sample_num)

        mash_model = Mash(
            self.anchor_num,
            self.mask_degree,
            self.sh_degree,
            10,
            10,
            dtype=torch.float32,
            device=self.device,
        )

        if not self.gt_sample_added_to_logger:
            gt_mash = data_dict["mash_params"]

            sh2d = 2 * self.mask_degree + 1
            positions = gt_mash[:, :3]
            ortho_poses = gt_mash[:, 3:9]
            mask_params = gt_mash[:, 9 : 9 + sh2d]
            sh_params = gt_mash[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                ortho_poses=ortho_poses,
                positions=positions,
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        for i in trange(sample_num):
            mash_params = sampled_array[i]

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

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
