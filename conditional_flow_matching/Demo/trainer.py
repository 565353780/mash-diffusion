import sys
sys.path.append('../ma-sh/')

from conditional_flow_matching.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    batch_size = 48
    accum_iter = 1
    num_workers = 16
    model_file_path = "./output/cfm_24depth_24h_class-image/total_model_last.pth"
    model_file_path = None
    device = "cuda:0"
    warm_step_num = 5000
    finetune_step_num = -1
    lr = 1e-4
    ema_decay = 0.9999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        model_file_path,
        device,
        warm_step_num,
        finetune_step_num,
        lr,
        ema_decay,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.train()
    return True
