import sys
sys.path.append("../ma-sh/")
sys.path.append("../base-trainer/")
sys.path.append("../dino-v2-detect/")
sys.path.append("../distribution-manage/")

from ma_sh.Config.custom_path import toDatasetRootPath

from mash_diffusion.Module.edm_trainer import EDMTrainer


def demo():
    dataset_root_folder_path = toDatasetRootPath()
    assert dataset_root_folder_path is not None
    print(dataset_root_folder_path)

    training_mode = 'dino'
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
    ema_decay = 0.9999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = None
    is_metric_lower_better = True
    sample_results_freq = 50
    use_amp = False
    quick_test = False

    if training_mode == 'single_category':
        batch_size = 24
        accum_iter = 2
        model_file_path = "../../output/edm-ShapeNet_03001627-v3/model_last.pth".replace('../../', './')
        lr = 2e-6
    elif training_mode == 'category':
        batch_size = 24
        accum_iter = 5
        model_file_path = "../../output/edm-ShapeNet-category-v2/model_last.pth".replace('../../', './')
        lr = 2e-6
    elif training_mode == 'multi_modal':
        batch_size = 24
        accum_iter = 2
        model_file_path = "../../output/edm-ShapeNet-multi_modal-v1/model_last.pth".replace('../../', './')
        lr = 2e-5
    elif training_mode == 'dino':
        batch_size = 12
        accum_iter = 2
        model_file_path = "../../output/edm-Objaverse_82K-single_image-v2/model_last.pth".replace('../../', './')
        model_file_path = None
        lr = 2e-4
    else:
        exit()

    edm_trainer = EDMTrainer(
        dataset_root_folder_path,
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
