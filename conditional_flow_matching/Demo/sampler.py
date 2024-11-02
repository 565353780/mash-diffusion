import sys
from threading import current_thread
from typing import Union

sys.path.append("../ma-sh/")

import os
import gc
import clip
import torch
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from math import sqrt, ceil
from shutil import copyfile

from conditional_flow_matching.Method.time import getCurrentTime
from conditional_flow_matching.Module.sampler import Sampler


def demoCondition(
    use_ema: bool = True,
    condition_value: Union[int, str] = 18,
    sample_num: int = 9,
    device: str = 'cuda:0',
    save_folder_path: Union[str, None] = None,
    condition_type: str = 'categoty'):
    assert condition_type in ['category', 'image']

    if condition_type == 'category':
        assert isinstance(condition_value, int)

        condition = condition_value
    elif condition_type == 'image':
        assert isinstance(condition_value, str)

        image_file_path = condition_value
        if not os.path.exists(image_file_path):
            print('[ERROR][sampler::demoCondition]')
            print('\t condition image file not exist!')
            return False

        clip_model_id: str = "ViT-L/14"
        model, preprocess = clip.load(clip_model_id, device=device)
        model.eval()

        image = Image.open(image_file_path)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            condition = (
                model.encode_image(image).detach().clone().cpu().numpy()
            )
    else:
        print('[ERROR][sampler::demoCondition]')
        print('\t condition type not valid!')
        return False

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
    model_file_path = output_folder_path + model_folder_path + "/total_model_last.pth"

    print(model_file_path)
    sampler = Sampler(model_file_path, use_ema, device)

    print("start diffuse", sample_num, "mashs....")
    sampled_array = sampler.sample(sample_num, condition)

    object_dist = [2, 2, 2]

    row_num = ceil(sqrt(sample_num))

    mash_model = sampler.toInitialMashModel('cpu')

    for j in range(sampled_array.shape[0]):
        if j != sampled_array.shape[0] -  1:
            continue

        if save_folder_path is None:
            current_time = getCurrentTime()
            save_folder_path = './output/sample/' + current_time + '/iter-' + str(j) + '/'

        if use_ema:
            ema_state = 'ema'
        else:
            ema_state = 'normal'
        save_folder_path += ema_state + '/' + condition_type + '/'
        os.makedirs(save_folder_path, exist_ok=True)

        if condition_type == 'image':
            copyfile(image_file_path, save_folder_path + 'condition_image.png')

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

    del sampler
    del sampled_array
    del mash_model
    gc.collect()
    torch.cuda.empty_cache()
    return True

def demo(save_folder_path: Union[str, None] = None):
    sample_num = 9
    device = 'cuda:0'

    categoty_id = 18
    demoCondition(True, categoty_id, sample_num, device, save_folder_path, 'category')
    demoCondition(False, categoty_id, sample_num, device, save_folder_path, 'category')

    image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/03001627/1a74a83fa6d24b3cacd67ce2c72c02e/y_5_x_3.png'
    demoCondition(True, image_file_path, sample_num, device, save_folder_path, 'image')
    demoCondition(False, image_file_path, sample_num, device, save_folder_path, 'image')

    return True
