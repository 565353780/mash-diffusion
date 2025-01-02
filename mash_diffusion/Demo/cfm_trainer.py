import sys
sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")
sys.path.append("../base-trainer/")

from ma_sh.Config.custom_path import toDatasetRootPath

from mash_diffusion.Module.cfm_trainer import CFMTrainer


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    dataset_json_file_path_dict = {
        "dino": dataset_root_folder_path + "Objaverse_82K/render_dino.pkl",
    }
    batch_size = 2
    accum_iter = 16
    num_workers = 2
    model_file_path = None
    # model_file_path = "../../output/cfm-20241230_16:16:14/model_last.pth".replace('../../', './')
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 2e-4
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 1
    use_amp = False

    cfm_trainer = CFMTrainer(
        dataset_root_folder_path,
        dataset_json_file_path_dict,
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

    cfm_trainer.train()
    return True
