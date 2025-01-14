import torch
from torch import nn
from tqdm import trange
from typing import Union
from abc import abstractmethod

from ma_sh.Config.custom_path import toModelRootPath
from ma_sh.Model.mash import Mash

from base_trainer.Module.base_trainer import BaseTrainer

from dino_v2_detect.Module.detector import Detector as DINODetector

from mash_diffusion.Dataset.image import ImageDataset
from mash_diffusion.Dataset.mash import MashDataset
from mash_diffusion.Dataset.embedding import EmbeddingDataset
from mash_diffusion.Dataset.single_category import SingleCategoryDataset
from mash_diffusion.Dataset.single_shape import SingleShapeDataset


class BaseDiffusionTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        training_mode: str = 'dino',
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
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
        self.dataset_root_folder_path = dataset_root_folder_path
        self.training_mode = training_mode

        self.anchor_num = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.anchor_channel = int(
            9 + (2 * self.mask_degree + 1) + ((self.sh_degree + 1) ** 2)
        )

        if training_mode in ['single_shape', 'single_category', 'category', 'multi_modal']:
            self.context_dim = 512
            self.n_heads = 8
            self.d_head = 64
            self.depth = 24
            self.fix_params = False
        elif training_mode in ['dino']:
            self.context_dim = 768
            self.n_heads = 8
            self.d_head = 64
            self.depth = 24
            self.fix_params = False

        self.gt_sample_added_to_logger = False

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
        if self.training_mode in ['dino']:
            model_root_path = toModelRootPath()
            assert model_root_path is not None

            model_type = 'base'
            model_file_path = model_root_path + 'DINOv2/dinov2_vitb14_reg4_pretrain.pth'
            dtype = 'auto'

            self.dino_detector = DINODetector(model_type, model_file_path, dtype, self.device)

        if self.training_mode in ['single_shape']:
            mash_file_path = self.dataset_root_folder_path + \
                "MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy"
            self.dataloader_dict["single_shape"] = {
                "dataset": SingleShapeDataset(
                    mash_file_path,
                    10000,
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['single_category']:
            self.dataloader_dict["single_category"] = {
                "dataset": SingleCategoryDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    '03001627',
                    "train",
                    'ShapeNet_03001627',
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['category', 'multi_modal']:
            self.dataloader_dict['category'] = {
                "dataset": MashDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    "train",
                    'ShapeNet',
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['dino']:
            self.dataloader_dict["dino"] = {
                "dataset": ImageDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/manifold_mash",
                    "Objaverse_82K/render_jpg",
                    self.dino_detector.transform,
                    "train",
                    'Objaverse_82K',
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['image', 'multi_modal']:
            self.dataloader_dict['image'] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    "ImageEmbedding_ulip/ShapeNet",
                    "random",
                    "train",
                    'ShapeNet',
                    True,
                    None,
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['point', 'multi_modal']:
            self.dataloader_dict['point'] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    "PointsEmbedding/ShapeNet",
                    "random",
                    "train",
                    'ShapeNet',
                    True,
                    None,
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        if self.training_mode in ['text', 'multi_modal']:
            self.dataloader_dict['text'] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    "TextEmbedding_ShapeGlot/ShapeNet",
                    "random",
                    "train",
                    'ShapeNet',
                    True,
                    None,
                    self.dtype,
                ),
                "repeat_num": 10,
            }

        if self.training_mode in ['single_category']:
            self.dataloader_dict["eval"] = {
                "dataset": SingleCategoryDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    '03001627',
                    "eval",
                    'ShapeNet_03001627',
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        elif self.training_mode in ['single_shape', 'category']:
            self.dataloader_dict["eval"] = {
                "dataset": MashDataset(
                    self.dataset_root_folder_path,
                    'MashV4/ShapeNet',
                    "eval",
                    'ShapeNet',
                    self.dtype,
                ),
            }

        elif self.training_mode in ['multi_modal']:
            self.dataloader_dict['eval'] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path,
                    "MashV4/ShapeNet",
                    "PointsEmbedding/ShapeNet",
                    "random",
                    "eval",
                    'ShapeNet',
                    True,
                    None,
                    self.dtype,
                ),
                "repeat_num": 1,
            }

        elif self.training_mode in ['dino']:
            self.dataloader_dict["eval"] = {
                "dataset": ImageDataset(
                    self.dataset_root_folder_path,
                    "Objaverse_82K/manifold_mash",
                    "Objaverse_82K/render_jpg",
                    self.dino_detector.transform,
                    "eval",
                    'Objaverse_82K',
                    self.dtype,
                ),
            }

        if 'eval' in self.dataloader_dict.keys():
            self.dataloader_dict["eval"]["dataset"].paths_list = self.dataloader_dict[
                "eval"
            ]["dataset"].paths_list[:64]

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
            data_dict["drop_prob"] = 0.1
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
        if self.training_mode == 'multi_modal':
            dataset = self.dataloader_dict['image']["dataset"]
        else:
            dataset = self.dataloader_dict[self.training_mode]["dataset"]

        model.eval()

        data_dict = dataset.__getitem__(1)
        data_dict = self.getCondition(data_dict)
 
        condition = data_dict['condition']

        if isinstance(condition, int):
            condition = torch.ones([sample_num]).long().to(self.device) * condition
        else:
            condition = condition.type(self.dtype).to(self.device).repeat(
                *([sample_num] + [1] * (condition.ndim - 1))
            )

        print("[INFO][BaseDiffusionTrainer::sampleModelStep]")
        print("\t start sample", sample_num, "mashs....")

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
