import os
import torch
import torchdiffeq
from tqdm import trange
from typing import Union

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from ma_sh.Model.mash import Mash

from base_trainer.Module.base_trainer import BaseTrainer

from mash_diffusion.Dataset.mash import MashDataset
from mash_diffusion.Dataset.embedding import EmbeddingDataset
from mash_diffusion.Dataset.single_shape import SingleShapeDataset
from mash_diffusion.Model.unet2d import MashUNet
from mash_diffusion.Model.mash_net import MashNet
from mash_diffusion.Model.mash_latent_net import MashLatentNet
from mash_diffusion.Model.image2mash_latent_net import Image2MashLatentNet
from mash_diffusion.Module.batch_ot_cfm import BatchExactOptimalTransportConditionalFlowMatcher
from mash_diffusion.Module.stacked_random_generator import StackedRandomGenerator


class Trainer(BaseTrainer):
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

        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.embed_dim = 1024
        self.context_dim = 1024
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        fm_id = 2
        if fm_id == 1:
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        elif fm_id == 2:
            self.FM = BatchExactOptimalTransportConditionalFlowMatcher(
                sigma=0.0, target_dim=None
            )
        elif fm_id == 3:
            self.FM = AffineProbPath(scheduler=CondOTScheduler())

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

        if False:
            self.dataloader_dict["points"] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path, "PointsEmbedding", "train"
                ),
                "repeat_num": 1,
            }

        if False:
            self.dataloader_dict["text"] = {
                "dataset": EmbeddingDataset(
                    self.dataset_root_folder_path, "TextEmbedding_ShapeGlot", "train"
                ),
                "repeat_num": 10,
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

    def createModel(self) -> bool:
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
                depth=self.depth,
            ).to(self.device)
        elif model_id == 3:
            self.model = MashLatentNet(
                n_latents=self.mash_channel,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
            ).to(self.device)
        elif model_id == 4:
            self.model = Image2MashLatentNet(
                n_latents=self.mash_channel,
                mask_degree=self.mask_degree,
                sh_degree=self.sh_degree,
                embed_dim=self.embed_dim,
                context_dim=self.context_dim,
                n_heads=self.n_heads,
                d_head=self.d_head,
                depth=self.depth,
            ).to(self.device)

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
            print("[ERROR][Trainer::toCondition]")
            print("\t valid condition type not found!")
            exit()

        return data_dict

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        mash_params = data_dict["mash_params"]

        init_mash_params = torch.randn_like(mash_params)

        data_dict = self.getCondition(data_dict)

        if isinstance(self.FM, ExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_mash_params, mash_params
            )
        elif isinstance(self.FM, BatchExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut = self.FM.sample_location_and_conditional_flow(
                init_mash_params, mash_params
            )
        elif isinstance(self.FM, AffineProbPath):
            t = torch.rand(mash_params.shape[0]).to(self.device)
            t = torch.pow(t, 1.0 / 2.0)
            path_sample = self.FM.sample(t=t, x_0=init_mash_params, x_1=mash_params)
            t = path_sample.t
            xt = path_sample.x_t
            ut = path_sample.dx_t
        else:
            print("[ERROR][Trainer::trainStep]")
            print("\t FM not valid!")
            exit()

        data_dict["ut"] = ut
        data_dict["t"] = t
        data_dict["xt"] = xt

        if is_training:
            data_dict["drop_prob"] = 0.0
        else:
            data_dict["drop_prob"] = 0.0

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        ut = data_dict["ut"]
        vt = result_dict["vt"]

        loss = torch.pow(vt - ut, 2).mean()

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: torch.nn.Module, model_name: str) -> bool:
        if self.local_rank != 0:
            return True

        sample_gt = False
        sample_num = 3
        timestamp_num = 2
        dataset = self.dataloader_dict["dino"]["dataset"]

        model.eval()

        data = dataset.__getitem__(0)
        gt_mash = data["mash_params"]

        data = self.getCondition(data)
        condition = data["condition"]

        if sample_gt:
            gt_mash = dataset.normalizeInverse(gt_mash)

        print("[INFO][Trainer::sampleModelStep]")
        print("\t start diffuse", sample_num, "mashs....")

        query_t = torch.linspace(0, 1, timestamp_num).to(self.device)
        query_t = torch.pow(query_t, 1.0 / 2.0)

        # x_init = torch.randn(sample_num, 400, 25, device=self.device)

        batch_seeds = torch.arange(sample_num)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        x_init = rnd.randn([sample_num, self.mash_channel, 25], device=self.device)

        traj = torchdiffeq.odeint(
            lambda t, x: model.forwardData(x, condition, t),
            x_init,
            query_t,
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

        sampled_array = traj.cpu()[-1]

        mash_model = Mash(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            20,
            800,
            0.4,
            dtype=torch.float64,
            device=self.device,
        )

        if sample_gt and not self.gt_sample_added_to_logger:
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
