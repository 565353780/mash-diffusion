import sys
sys.path.append("../ma-sh/")

import os
import gc
import clip
import torch
from PIL import Image
from tqdm import tqdm
from typing import Union
from math import sqrt, ceil
from shutil import copyfile

from conditional_flow_matching.Method.time import getCurrentTime
from conditional_flow_matching.Module.sampler import Sampler

global current_time

current_time = None


def demoCondition(
    model_file_path: str,
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

    sampler = Sampler(model_file_path, use_ema, device)

    print("start diffuse", sample_num, "mashs....")
    sampled_array = sampler.sample(sample_num, condition)

    object_dist = [0, 0, 0]

    row_num = ceil(sqrt(sample_num))

    mash_model = sampler.toInitialMashModel('cpu')

    global current_time

    for j in range(sampled_array.shape[0]):
        if j != sampled_array.shape[0] -  1:
            continue

        if save_folder_path is None:
            if current_time is None:
                current_time = getCurrentTime()
            save_folder_path = './output/sample/' + current_time + '/iter-' + str(j) + '/'

        if use_ema:
            ema_state = 'ema'
        else:
            ema_state = 'normal'
        save_folder_path += ema_state + '/'

        if condition_type == 'category':
            condition_info = 'category/' + str(condition)
        elif condition_type == 'image':
            condition_info = 'image/' + image_file_path.split('/ShapeNet/')[1].split('/y_5_x_3.png')[0]
        save_folder_path += condition_info + '/'

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

            translate = [
                int(i / row_num) * object_dist[0],
                (i % row_num) * object_dist[1],
                j * object_dist[2],
            ]

            mash_model.translate(translate)

            mash_model.saveParamsFile(save_folder_path + 'mash/sample_' + str(i+1) + '.npy', True)
            mash_model.saveAsPcdFile(save_folder_path + 'pcd/sample_' + str(i+1) + '.ply', True)

    del sampler
    del sampled_array
    del mash_model
    gc.collect()
    torch.cuda.empty_cache()
    return True

def demo(save_folder_path: Union[str, None] = None):
    model_file_path = './output/6depth_1124epoch/total_model_last.pth'
    sample_num = 9
    device = 'cuda:0'

    categoty_id = 18
    # 0: airplane
    # 2: bag
    # 6: bench
    # 18: chair
    # 22: monitor
    # 23: earphone
    # 24: spigot
    # 26: guitar
    # 30: lamp
    # 46: skateboard
    # 47: sofa
    # 49: table
    # 53: watercraft
    for categoty_id in [0, 2, 6, 18, 22, 23, 24, 26, 30, 46, 47, 49, 53]:
        print('start sample for category ' + str(categoty_id) + '...')
        demoCondition(model_file_path, True, categoty_id, sample_num, device, save_folder_path, 'category')

    # image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/02691156/1adb40469ec3636c3d64e724106730cf'
    image_id_list = [
        '03001627/1a74a83fa6d24b3cacd67ce2c72c02e',
        '03001627/1a38407b3036795d19fb4103277a6b93',
        '03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7',
        '02691156/1a6ad7a24bb89733f412783097373bdc',
        '02691156/1a32f10b20170883663e90eaf6b4ca52',
        '02691156/1abe9524d3d38a54f49a51dc77a0dd59',
        '02691156/1adb40469ec3636c3d64e724106730cf',
    ]
    for image_id in image_id_list:
        print('start sample for image ' + image_id + '...')
        image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/' + image_id + '/y_5_x_3.png'
        demoCondition(model_file_path, True, image_file_path, sample_num, device, save_folder_path, 'image')

    return True
