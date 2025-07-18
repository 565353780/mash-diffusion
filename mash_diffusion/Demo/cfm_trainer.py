import MFSClient

import sys

sys.path.append("../ma-sh/")
sys.path.append("../base-trainer/")
sys.path.append("../dino-v2-detect/")

import os
import torch

from mash_diffusion.Module.cfm_trainer import CFMTrainer


def demo():
    dataset_root_folder_path = os.environ["HOME"] + "/chLi/Dataset/"
    assert dataset_root_folder_path is not None
    print(dataset_root_folder_path)

    batch_size = 6
    accum_iter = 6
    num_workers = 16
    model_file_path = (
        "../../output/cfm-Objaverse_82K-single_image-v10/model_last.pth".replace(
            "../../", "./"
        )
    )
    model_file_path = None
    weights_only = False
    device = "auto"
    dtype = torch.float32
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 1e-5
    lr_batch_size = 1024
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.9999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 1
    use_amp = True
    quick_test = False

    cfm_trainer = CFMTrainer(
        dataset_root_folder_path,
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

    cfm_trainer.train()
    return True
