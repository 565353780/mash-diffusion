import os
import torch
from torch import nn
from tqdm import trange
from typing import Union
from abc import abstractmethod

from ma_sh.Model.mash import Mash

from base_trainer.Module.base_trainer import BaseTrainer

from mash_diffusion.Dataset.mash import MashDataset
from mash_diffusion.Dataset.embedding import EmbeddingDataset
from mash_diffusion.Dataset.single_shape import SingleShapeDataset


class BaseDiffusionTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        dataset_json_file_path_dict: dict = {},
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "cuda:0",
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
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.dataset_json_file_path_dict = dataset_json_file_path_dict

        self.anchor_num = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.anchor_channel = int(
            9 + (2 * self.mask_degree + 1) + ((self.sh_degree + 1) ** 2)
        )

        self.gt_sample_added_to_logger = False

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            device,
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
        )
        return

    def createDatasets(self) -> bool:
        if False:
            mash_file_path = (
                os.environ["HOME"]
                + "/Dataset/MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy"
            )
            self.dataloader_dict["single_shape"] = {
                "dataset": SingleShapeDataset(mash_file_path),
                "repeat_num": 1,
            }

        if False:
            self.dataloader_dict["mash"] = {
                "dataset": MashDataset(self.dataset_root_folder_path, "train"),
                "repeat_num": 1,
            }

        if True:
            self.dataloader_dict["dino"] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/render_dino",
                    "dino",
                    "train",
                    self.dataset_json_file_path_dict.get("dino"),
                ),
                "repeat_num": 1,
            }

        if True:
            self.dataloader_dict["eval"] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/render_dino",
                    "dino",
                    "eval",
                    self.dataset_json_file_path_dict.get("dino"),
                ),
            }

            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:64]

        return True

    def getCondition(self, data_dict: dict) -> dict:
        if "category_id" in data_dict.keys():
            data_dict["condition"] = data_dict["category_id"]
        elif "embedding" in data_dict.keys():
            embedding = data_dict["embedding"]

            if embedding.ndim == 2:
                embedding = embedding.unsqueeze(1)
            elif embedding.ndim == 4:
                embedding = torch.squeeze(embedding, dim=1)

            data_dict["condition"] = embedding.to(self.device)
        else:
            print("[ERROR][BaseDiffusionTrainer::toCondition]")
            print("\t valid condition type not found!")
            exit()

        return data_dict

    @abstractmethod
    def preProcessDiffusionData(self, data_dict: dict, is_training: bool = False) -> dict:
        '''
        if is_training:
            data_dict[new_name] = new_value
        return data_dict
        '''
        pass

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        data_dict = self.getCondition(data_dict)

        if is_training:
            data_dict["drop_prob"] = 0.0
        else:
            data_dict["drop_prob"] = 0.0

        data_dict = self.preProcessDiffusionData(data_dict, is_training)

        return data_dict

    @abstractmethod
    @torch.no_grad()
    def sampleMashData(self, model: nn.Module, condition: torch.Tensor, sample_num: int) -> torch.Tensor:
        '''
        mash_params = sample_func(model, condition, sample_num)
        return mash_params
        '''
        pass

    @torch.no_grad()
    def sampleModelStep(self, model: nn.Module, model_name: str) -> bool:
        if self.local_rank != 0:
            return True

        sample_num = 3
        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(0)
        data_dict = self.getCondition(data_dict)

        condition = data_dict['condition']

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start diffuse", sample_num, "mashs....")

        sampled_array = self.sampleMashData(model, condition, sample_num)

        mash_model = Mash(
            self.anchor_num,
            self.mask_degree,
            self.sh_degree,
            20,
            800,
            0.4,
            dtype=torch.float64,
            device=self.device,
        )

        if not self.gt_sample_added_to_logger:
            gt_mash = data_dict['mash_params']

            gt_mash = dataset.normalizeInverse(gt_mash)

            sh2d = 2 * self.mask_degree + 1
            ortho_poses = gt_mash[:, :6]
            positions = gt_mash[:, 6:9]
            mask_params = gt_mash[:, 9 : 9 + sh2d]
            sh_params = gt_mash[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses,
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud("GT_MASH/gt_mash", pcd, self.step)

            self.gt_sample_added_to_logger = True

        for i in trange(sample_num):
            mash_params = sampled_array[i]

            mash_params = dataset.normalizeInverse(mash_params)

            sh2d = 2 * self.mask_degree + 1
            ortho_poses = mash_params[:, :6]
            positions = mash_params[:, 6:9]
            mask_params = mash_params[:, 9 : 9 + sh2d]
            sh_params = mash_params[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses,
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud(model_name + "/pcd_" + str(i), pcd, self.step)

        return True
