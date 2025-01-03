import sys
sys.path.append("../ma-sh/")
sys.path.append('../distribution-manage/')

import os
from tqdm import tqdm
from math import sqrt, ceil

from mash_diffusion.Config.shapenet import CATEGORY_IDS
from mash_diffusion.Method.time import getCurrentTime
from mash_diffusion.Module.edm_sampler import EDMSampler

def postProcess(time_stamp: str) -> bool:
    return True

def demo():
    model_file_path = "../../output/20250102_21:12:31/model_last.pth".replace('../.', '')
    transformer_id = 'ShapeNet_03001627'
    use_ema = True
    device = "cpu"
    sample_num = 9
    condition_name = '03001627'
    diffuse_steps = 18
    sample_category = False
    sample_fixed_anchors = True
    save_results_only = True

    condition = CATEGORY_IDS[condition_name]
    condition_info = 'category/' + condition_name

    time_stamp = getCurrentTime()
    save_folder_path = None

    print(model_file_path)
    edm_sampler = EDMSampler(model_file_path, use_ema, device, transformer_id)

    print("start diffuse", sample_num, "mashs....")
    if sample_category:
        sampled_array = edm_sampler.sample(sample_num, condition, diffuse_steps)
    elif sample_fixed_anchors:
        mash_file_path_list = [
            '../ma-sh/output/combined_mash.npy',
        ]
        sampled_array = edm_sampler.sampleWithFixedAnchors(mash_file_path_list, sample_num, condition, diffuse_steps)
    else:
        return True

    object_dist = [0, 0, 0]

    row_num = ceil(sqrt(sample_num))

    mash_model = edm_sampler.toInitialMashModel()

    for j in range(len(sampled_array)):
        if save_results_only:
            if j != len(sampled_array) - 1:
                continue

        if save_folder_path is None:
            save_folder_path = './output/sample/' + time_stamp + '/'
        current_save_folder_path = save_folder_path + 'iter_' + str(j) + '/' + condition_info + '/'

        os.makedirs(current_save_folder_path, exist_ok=True)

        print("start create mash files,", j + 1, '/', len(sampled_array), "...")
        for i in tqdm(range(sample_num)):
            mash_params = sampled_array[j][i]

            mash_params = edm_sampler.transformer.inverse_transform(mash_params)

            sh2d = 2 * edm_sampler.mask_degree + 1
            ortho_poses = mash_params[:, :6]
            positions = mash_params[:, 6:9]
            mask_params = mash_params[:, 9 : 9 + sh2d]
            sh_params = mash_params[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses
            )

            translate = [
                int(i / row_num) * object_dist[0],
                0 * object_dist[1],
                (i % row_num) * object_dist[2],
            ]

            mash_model.translate(translate)

            mash_model.saveParamsFile(current_save_folder_path + 'mash/sample_' + str(i+1) + '_mash.npy', True)
            mash_model.saveAsPcdFile(current_save_folder_path + 'pcd/sample_' + str(i+1) + '_pcd.ply', True)

    return True
