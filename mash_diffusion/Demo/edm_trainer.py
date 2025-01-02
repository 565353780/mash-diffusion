import sys
sys.path.append("../ma-sh/")
sys.path.append("../distribution-manage/")
sys.path.append("../base-trainer/")

from ma_sh.Config.custom_path import toDatasetRootPath

from mash_diffusion.Module.edm_trainer import EDMTrainer


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None

    dataset_json_file_path_dict = {
        "dino": dataset_root_folder_path + "Objaverse_82K/render_dino.pkl",
    }
    training_mode = 'category'
    batch_size = 24
    accum_iter = 2
    num_workers = 16
    model_file_path = None
    model_file_path = "../../output/edm-ShapeNet_03001627-24depth-v3/model_last.pth".replace('../../', './')
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 2e-5
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 50
    use_amp = False
    quick_test = False

    edm_trainer = EDMTrainer(
        dataset_root_folder_path,
        dataset_json_file_path_dict,
        training_mode,
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
        quick_test,
    )

    edm_trainer.train()
    return True
