import os
import torch
import torchsde
import torchdiffeq
from torch import nn
from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from torchcfm.conditional_flow_matching import *

from conditional_flow_matching.Dataset.mash import MashDataset
from conditional_flow_matching.Method.time import getCurrentTime
from conditional_flow_matching.Method.path import createFileFolder
from conditional_flow_matching.Module.logger import Logger


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 400,
        accum_iter: int = 1,
        num_workers: int = 4,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
        warm_epoch_step_num: int = 20,
        warm_epoch_num: int = 10,
        finetune_step_num: int = 400,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        factor: float = 0.9,
        patience: int = 1,
        min_lr: float = 1e-4,
        drop_prob: float = 0.75,
        deterministic: bool = False,
        kl_weight: float = 1.0,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.deterministic = deterministic
        self.loss_kl_weight = kl_weight

        self.loss_ortho_poses_weight = 1.0
        self.loss_positions_weight = 1.0
        self.loss_mask_params_weight = 1.0
        self.loss_sh_params_weight = 1.0

        self.accum_iter = accum_iter
        self.dtype = dtype
        self.device = device

        self.warm_epoch_step_num = warm_epoch_step_num
        self.warm_epoch_num = warm_epoch_num

        self.finetune_step_num = finetune_step_num

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.drop_prob = drop_prob

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.train_loader = DataLoader(
            MashDataset(dataset_root_folder_path, 'train'),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model_id = 1
        if model_id == 1:
            sigma = 0.0
            self.model = UNetModel(
                dim=(1, 400, 25), num_channels=32, num_res_blocks=1, num_classes=55, class_cond=True
            ).to(device)
            self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            self.node = NeuralODE(self.model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        self.loss_fn = nn.MSELoss()

        self.initRecords()

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.min_lr_reach_time = 0
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_state_dict["model"])
        return True

    def getLr(self, optimizer) -> float:
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def toTrainStepNum(self, scheduler: LRScheduler) -> int:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            return self.finetune_step_num

        if scheduler.T_mult == 1:
            warm_epoch_num = scheduler.T_0 * self.warm_epoch_num
        else:
            warm_epoch_num = int(
                scheduler.T_mult
                * (1.0 - pow(scheduler.T_mult, self.warm_epoch_num))
                / (1.0 - scheduler.T_mult)
            )

        return self.warm_epoch_step_num * warm_epoch_num

    def trainStep(
        self,
        data: dict,
        optimizer: Optimizer,
    ) -> dict:
        cfm_mash_params = data['cfm_mash_params'].to(self.device)
        category_id = data['category_id'].to(self.device)

        cfm_mash_params_noise = torch.randn_like(cfm_mash_params)

        t, xt, ut, _, y1 = self.FM.guided_sample_location_and_conditional_flow(cfm_mash_params_noise, cfm_mash_params, y1=category_id)

        vt = self.model(t, xt, y1)

        loss = self.loss_fn(vt, ut)

        accum_loss = loss / self.accum_iter
        accum_loss.backward()

        if (self.step + 1) % self.accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_dict = {
            "Loss": loss.item(),
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["Loss"])

            if self.getLr(optimizer) == self.min_lr:
                self.min_lr_reach_time += 1

            return self.min_lr_reach_time > self.patience

        current_warm_epoch = self.step / self.warm_epoch_step_num
        scheduler.step(current_warm_epoch)

        return current_warm_epoch >= self.warm_epoch_num

    def train(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> bool:
        train_step_num = self.toTrainStepNum(scheduler)
        final_step = self.step + train_step_num

        print("[INFO][Trainer::train]")
        print("\t start training ...")

        loss_dict_list = []
        while self.step < final_step:
            self.model.train()

            pbar = tqdm(total=len(self.train_loader))
            for data in self.train_loader:
                train_loss_dict = self.trainStep(data, optimizer)

                loss_dict_list.append(train_loss_dict)

                lr = self.getLr(optimizer)

                if (self.step + 1) % self.accum_iter == 0:
                    for key in train_loss_dict.keys():
                        value = 0
                        for i in range(len(loss_dict_list)):
                            value += loss_dict_list[i][key]
                        value /= len(loss_dict_list)
                        self.logger.addScalar("Train/" + key, value, self.step)
                    self.logger.addScalar("Train/Lr", lr, self.step)

                    loss_dict_list = []

                pbar.set_description(
                    "LOSS %.6f LR %.4f"
                    % (
                        train_loss_dict["Loss"],
                        self.getLr(optimizer) / self.lr,
                    )
                )

                self.step += 1
                pbar.update(1)

                if self.checkStop(optimizer, scheduler, train_loss_dict):
                    break

                if self.step >= final_step:
                    break

            pbar.close()

            self.autoSaveModel(train_loss_dict['Loss'])

        return True

    def autoTrain(
        self,
    ) -> bool:
        print("[INFO][Trainer::autoTrain]")
        print("\t start auto train mash occ decoder...")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        warm_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
        finetune_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )

        self.train(optimizer, warm_scheduler)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr
        self.train(optimizer, finetune_scheduler)

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.state_dict(),
            "loss_min": self.loss_min,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, value: float, check_lower: bool = True) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + "model_last.pth"

        self.saveModel(save_last_model_file_path)

        if self.loss_min == float("inf"):
            if not check_lower:
                self.loss_min = -float("inf")

        if check_lower:
            if value > self.loss_min:
                return False
        else:
            if value < self.loss_min:
                return False

        self.loss_min = value

        save_best_model_file_path = self.save_result_folder_path + "model_best.pth"

        self.saveModel(save_best_model_file_path)

        return True
