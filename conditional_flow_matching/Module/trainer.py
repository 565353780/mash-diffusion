import os
import torch
from tqdm import tqdm
from copy import deepcopy
from typing import Union
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from torchcfm.conditional_flow_matching import (
    TargetConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

from conditional_flow_matching.Dataset.mash import MashDataset
from conditional_flow_matching.Dataset.embedding import EmbeddingDataset
from conditional_flow_matching.Model.unet2d import MashUNet
from conditional_flow_matching.Model.mash_net import MashNet
from conditional_flow_matching.Method.time import getCurrentTime
from conditional_flow_matching.Method.path import createFileFolder, removeFile, renameFile
from conditional_flow_matching.Module.logger import Logger


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 24,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "cuda:0",
        warm_step_num: int = 5000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        ema_start_step: int = 20000,
        ema_decay: float = 0.9999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.accum_iter = accum_iter
        self.device = device
        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num
        self.lr = lr
        self.ema_start_step = ema_start_step
        self.ema_decay = ema_decay

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.step = 0

        self.logger = Logger()

        self.dataloader_dict = {}

        if True:
            self.dataloader_dict['mash'] =  {
                'dataset': MashDataset(dataset_root_folder_path, 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['image'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'ImageEmbedding_ulip', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['points'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'PointsEmbedding', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['text'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'TextEmbedding_ShapeGlot', 'train'),
                'repeat_num': 10,
            }

        for key, item in self.dataloader_dict.items():
            self.dataloader_dict[key]['dataloader'] = DataLoader(
                item['dataset'],
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        model_id = 2
        if model_id == 1:
            self.model = MashUNet(768).to(device)
        elif model_id == 2:
            self.model = MashNet().to(device)

        self.ema_model = deepcopy(self.model)
        self.ema_loss = None

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.FM = TargetConditionalFlowMatcher(sigma=0.0)
        # self.FM = VariancePreservingConditionalFlowMatcher(sigma=0.0)

        self.initRecords()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def initRecords(self) -> bool:
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
        self.ema_model.load_state_dict(model_state_dict["ema_model"])
        return True

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        return min(step, self.warm_step_num) / self.warm_step_num

    def ema(self) -> bool:
        if self.step <= self.ema_start_step:
            source_dict = self.model.state_dict()
            target_dict = self.ema_model.state_dict()
            for key in source_dict.keys():
                target_dict[key].data.copy_(source_dict[key].data)

            return True

        source_dict = self.model.state_dict()
        target_dict = self.ema_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * self.ema_decay + source_dict[key].data * (1 - self.ema_decay)
            )
        return True

    def trainStep(
        self,
        data: dict,
    ) -> dict:
        cfm_mash_params = data['cfm_mash_params'].to(self.device)
        condition = data['condition'].to(self.device)

        cfm_mash_params_noise = torch.randn_like(cfm_mash_params)

        if isinstance(self.FM, ExactOptimalTransportConditionalFlowMatcher):
            t, xt, ut, _, y1 = self.FM.guided_sample_location_and_conditional_flow(cfm_mash_params_noise, cfm_mash_params, y1=condition)
        else:
            t, xt, ut = self.FM.sample_location_and_conditional_flow(cfm_mash_params_noise, cfm_mash_params)
            y1 = condition

        vt = self.model(xt, y1, t)

        loss = torch.mean((vt - ut) ** 2)

        accum_loss = loss / self.accum_iter
        accum_loss.backward()

        if (self.step + 1) % self.accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.sched.step()
            self.ema()
            self.optim.zero_grad()

        loss_dict = {
            "Loss": loss.item(),
        }

        return loss_dict

    def toCondition(self, data: dict) -> Union[torch.Tensor, None]:
        if 'category_id' in data.keys():
            return data['category_id']

        if 'embedding' in data.keys():
            return data["embedding"]

        print('[ERROR][Trainer::toCondition]')
        print('\t valid condition type not found!')
        return None

    def train(self) -> bool:
        final_step = self.step + self.finetune_step_num

        print("[INFO][Trainer::train]")
        print("\t start training ...")

        loss_dict_list = []

        epoch_idx = 1
        while self.step < final_step or self.finetune_step_num < 0:
            self.model.train()

            for data_name, dataloader_dict in self.dataloader_dict.items():

                dataloader = dataloader_dict['dataloader']
                repeat_num = dataloader_dict['repeat_num']

                for i in range(repeat_num):
                    print('[INFO][Trainer::train]')
                    print('\t start training on dataset [', data_name, '] ,', i + 1, '/', repeat_num, '...')

                    pbar = tqdm(total=len(dataloader))
                    for data in dataloader:
                        condition = self.toCondition(data)
                        if condition is None:
                            print('[ERROR][Trainer::train]')
                            print('\t toCondition failed!')
                            continue

                        conditional_data = {
                            'cfm_mash_params': data['cfm_mash_params'],
                            'condition': condition,
                        }

                        train_loss_dict = self.trainStep(conditional_data)

                        loss_dict_list.append(train_loss_dict)

                        lr = self.getLr()

                        if (self.step + 1) % self.accum_iter == 0:
                            for key in train_loss_dict.keys():
                                value = 0
                                for i in range(len(loss_dict_list)):
                                    value += loss_dict_list[i][key]
                                value /= len(loss_dict_list)
                                self.logger.addScalar("Train/" + key, value, self.step)
                            self.logger.addScalar("Train/Lr", lr, self.step)

                            if self.ema_loss is None or self.step <= self.ema_start_step:
                                self.ema_loss = train_loss_dict["Loss"]
                            else:
                                self.ema_loss = self.ema_loss * self.ema_decay + train_loss_dict["Loss"] * (1 - self.ema_decay)
                            self.logger.addScalar("Train/EMALoss", self.ema_loss, self.step)

                            loss_dict_list = []

                        pbar.set_description(
                            "EPOCH %d LOSS %.6f LR %.4f"
                            % (
                                epoch_idx,
                                train_loss_dict["Loss"],
                                self.getLr() / self.lr,
                            )
                        )

                        self.step += 1
                        pbar.update(1)

                    pbar.close()

                self.autoSaveModel("total")

                epoch_idx += 1

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, name: str) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + name + "_model_last.pth"

        tmp_save_last_model_file_path = save_last_model_file_path[:-4] + "_tmp.pth"

        self.saveModel(tmp_save_last_model_file_path)

        removeFile(save_last_model_file_path)
        renameFile(tmp_save_last_model_file_path, save_last_model_file_path)

        return True
