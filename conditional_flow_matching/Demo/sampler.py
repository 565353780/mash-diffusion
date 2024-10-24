import sys
from typing import Union

sys.path.append("../ma-sh/")

import os
import open3d as o3d
from tqdm import tqdm
from math import sqrt, ceil

from conditional_flow_matching.Module.sampler import Sampler


def demo(save_folder_path: Union[str, None] = None):
    output_folder_path = './output/'
    model_folder_name_list = os.listdir(output_folder_path)

    valid_model_folder_name_list = []
    valid_model_folder_name_list.append('2023')
    for model_folder_name in model_folder_name_list:
        if "2024" not in model_folder_name:
            continue
        if not os.path.isdir(output_folder_path + model_folder_name + "/"):
            continue

        valid_model_folder_name_list.append(model_folder_name)

    valid_model_folder_name_list.sort()
    model_folder_path = valid_model_folder_name_list[-1]
    #model_folder_path = 'pretrain-single-v1'
    model_file_path = output_folder_path + model_folder_path + "/model_last.pth"

    device = "cuda:0"

    sample_num = 9
    category_id = 18

    print(model_file_path)
    sampler = Sampler(model_file_path, device)

    print("start diffuse", sample_num, "mashs....")
    sampled_array = sampler.sample(sample_num, category_id)

    object_dist = [2, 2, 2]

    row_num = ceil(sqrt(sample_num))

    mash_model = sampler.toInitialMashModel('cpu')

    for j in range(sampled_array.shape[0]):
        if j != sampled_array.shape[0] -  1:
            continue

        if save_folder_path is None:
            save_folder_path = './output/sample/save_itr_' + str(j) + '/'
        os.makedirs(save_folder_path, exist_ok=True)

        for i in tqdm(range(sample_num)):

            mash_params = sampled_array[j][i]

            sh2d = 2 * sampler.mask_degree + 1
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
            mash_pcd = mash_model.toSamplePcd()

            if True:
                translate = [
                    int(i / row_num) * object_dist[0],
                    (i % row_num) * object_dist[1],
                    j * object_dist[2],
                ]

                mash_pcd.translate(translate)

            o3d.io.write_point_cloud(
                save_folder_path + 'sample_' + str(i) + '.ply',
                mash_pcd,
                write_ascii=True,
            )
    return True
