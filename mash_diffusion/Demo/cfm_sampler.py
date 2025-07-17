import MFSClient

import sys

sys.path.append("../ma-sh/")
sys.path.append("../wn-nc/")
sys.path.append("../ulip-manage/")
sys.path.append("../blender-manage/")
sys.path.append("../dino-v2-detect/")
sys.path.append("../mash-occ-decoder/")
sys.path.append("../distribution-manage/")

import torch
import random

from mash_diffusion.Dataset.tos_image import TOSImageDataset
from mash_diffusion.Method.time import getCurrentTime
from mash_diffusion.Module.cfm_sampler import CFMSampler


def demo():
    cfm_model_file_path = "/vepfs-cnbja62d5d769987/lichanghao/github/MASH/mash-diffusion/output/test/model_last.pth"
    occ_model_file_path = None
    cfm_use_ema = False
    occ_use_ema = True
    device = "cuda:0"
    dino_model_file_path = "./data/dinov2_vitl14_reg4_pretrain.pth"

    occ_batch_size = 1200000  # 24G GPU Memory required
    # occ_batch_size = 500000 # 12G GPU Memory required

    save_folder_path = "./output/sample/" + getCurrentTime() + "/"

    sample_shape_num = 100
    sample_mash_per_shape = 4

    timestamp_num = 2
    save_results_only = True

    recon_wnnc = False
    recon_occ = False
    render_pcd = False

    smooth_wnnc = True and recon_wnnc
    smooth_occ = True and recon_occ
    render_wnnc = True and recon_wnnc
    render_wnnc_smooth = True and recon_wnnc and smooth_wnnc
    render_occ = True and recon_occ
    render_occ_smooth = True and recon_occ and smooth_occ

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        dino_model_file_path,
        occ_model_file_path,
        cfm_use_ema,
        occ_use_ema,
        device,
        occ_batch_size,
        recon_wnnc,
        recon_occ,
        smooth_wnnc,
        smooth_occ,
        render_pcd,
        render_wnnc,
        render_wnnc_smooth,
        render_occ,
        render_occ_smooth,
    )

    client = MFSClient.MFSClient2()

    dataset = TOSImageDataset(
        client,
        mash_bucket="mm-data-general-model-trellis",
        mash_folder_key="mash/",
        image_bucket="mm-data-general-model-v1",
        image_folder_key="rendering/orient_cam72_base/",
        transform=cfm_sampler.dino_detector.transform,
        paths_file_path="./data/tos_paths_list.npy",
        return_raw_data=True,
    )

    all_shape_num = len(dataset)
    if sample_shape_num >= all_shape_num:
        random_shape_idxs = list(range(all_shape_num))
    else:
        random_shape_idxs = random.sample(range(all_shape_num), sample_shape_num)

    for shape_idx in random_shape_idxs:
        print("start sample for shape No." + str(shape_idx) + " ...")
        data_dict = dataset[shape_idx]
        cfm_sampler.samplePipeline(
            data_dict,
            save_folder_path + str(shape_idx) + "/",
            sample_mash_per_shape,
            timestamp_num,
            save_results_only,
        )

    cfm_sampler.waitRender()

    torch.cuda.empty_cache()
    return True
