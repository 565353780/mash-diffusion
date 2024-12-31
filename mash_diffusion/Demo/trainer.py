import os
import sys
import time
import json
import math
import torch
import datetime
import numpy as np
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter

from td_shape_to_vec_set.Data.smoothed_value import SmoothedValue
from td_shape_to_vec_set.Loss.edm import EDMLoss
from td_shape_to_vec_set.Dataset.mash import MashDataset
from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond
from td_shape_to_vec_set.Method.distributed import (
    init_distributed_mode,
    get_rank,
    get_world_size,
    is_main_process,
)
from td_shape_to_vec_set.Method.lr_sched import adjust_learning_rate
from td_shape_to_vec_set.Method.misc import load_model, save_model, all_reduce_mean
from td_shape_to_vec_set.Module.Logger.metric import MetricLogger
from td_shape_to_vec_set.Optimizer.native_scaler import (
    NativeScalerWithGradNormCount as NativeScaler,
)


class Trainer(object):
    def __init__(self) -> None:
        self.mash_channel = 40
        self.sh_2d_degree = 4
        self.sh_3d_degree = 4
        self.channels = int(
            6 + (2 * self.sh_2d_degree + 1) + ((self.sh_3d_degree + 1) ** 2)
        )
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        dataset_folder_path_list = [
            "/home/chli/Dataset/",
            "/data2/lch/Dataset/",
        ]
        for dataset_folder_path in dataset_folder_path_list:
            if not os.path.exists(dataset_folder_path):
                continue

            self.mash_dataset_folder_path = dataset_folder_path
            break

        self.batch_size = 64
        self.accum_iter = 1
        self.warmup_epochs = 4
        self.epochs = 10000
        self.clip_grad = None
        self.weight_decay = 0.05
        self.lr = None
        self.blr = 1e-4
        self.layer_decay = 0.75
        self.min_lr = 1e-6
        self.output_dir = "./output/"
        self.log_dir = "./logs/"
        self.device = "cuda"
        self.seed = 0
        self.resume = ""
        self.start_epoch = 0
        self.eval = False
        self.dist_eval = False
        self.num_workers = 60
        self.pin_mem = True
        self.no_pin_mem = False
        self.world_size = 1
        self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = "env://"

        self.distributed = True
        self.save_freq = 10
        return

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        log_writer=None,
    ):
        model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch)
        print_freq = 20

        accum_iter = self.accum_iter

        optimizer.zero_grad()

        if log_writer is not None:
            print("log_dir: {}".format(log_writer.log_dir))

        for data_iter_step, data in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, self
                )

            mash_params = data["mash_params"].to(device, non_blocking=True)
            categories = data["category_id"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(model, mash_params, categories)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_iter_step + 1) % accum_iter == 0,
            )
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            min_lr = 10.0
            max_lr = 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)

            loss_value_reduce = all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", max_lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, data_loader, model, criterion, device):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        # switch to evaluation mode
        model.eval()

        for data in metric_logger.log_every(data_loader, 50, header):
            mash_params = data["mash_params"].to(device, non_blocking=True)
            categories = data["category_id"].to(device, non_blocking=True)
            # compute output

            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(model, mash_params, categories)

            metric_logger.update(loss=loss.item())

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self):
        init_distributed_mode(self)

        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

        device = torch.device(self.device)

        # fix the seed for reproducibility
        seed = self.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset_train = MashDataset(self.mash_dataset_folder_path)
        dataset_val = MashDataset(self.mash_dataset_folder_path)

        if True:  # self.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if self.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                        "This will slightly alter validation results as extra duplicate entries are added to achieve "
                        "equal num of samples per-process."
                    )
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and self.log_dir is not None and not self.eval:
            os.makedirs(self.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=self.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            # prefetch_factor=2,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            # batch_size=self.batch_size,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
        )

        model = EDMPrecond(
            n_latents=self.mash_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
        )
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print("number of params (M): %.2f" % (n_parameters / 1.0e6))

        eff_batch_size = self.batch_size * self.accum_iter * get_world_size()

        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 256

        print("base lr: %.2e" % (self.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % self.lr)

        print("accumulate grad iterations: %d" % self.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        # # build optimizer with layer-wise lr decay (lrd)
        # param_groups = lrd.param_groups_lrd(model_without_ddp, self.weight_decay,
        #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
        #     layer_decay=self.layer_decay
        # )
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=self.lr)
        loss_scaler = NativeScaler()

        criterion = EDMLoss()

        print("criterion = %s" % str(criterion))

        load_model(
            args=self,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

        if self.eval:
            test_stats = self.evaluate(data_loader_val, model, device)
            print(
                f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}"
            )
            exit(0)

        print(f"Start training for {self.epochs} epochs")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                self.clip_grad,
                log_writer=log_writer,
            )
            if self.output_dir and (
                epoch % self.save_freq == 0 or epoch + 1 == self.epochs
            ):
                save_model(
                    args=self,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            if epoch % 5 == 0 or epoch + 1 == self.epochs:
                test_stats = self.evaluate(data_loader_val, model, criterion, device)
                print(
                    f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}"
                )

                if log_writer is not None:
                    log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                # if self.output_dir and is_main_process():
                #     if log_writer is not None:
                #         log_writer.flush()
                #     with open(os.path.join(self.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                #         f.write(json.dumps(log_stats) + "\n")

            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

            if self.output_dir and is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(
                    os.path.join(self.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
