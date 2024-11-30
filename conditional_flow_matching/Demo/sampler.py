import sys

sys.path.append("../ma-sh/")
sys.path.append("../ulip-manage/")

import os
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from math import sqrt, ceil
from shutil import copyfile

from ma_sh.Data.mesh import Mesh
from ulip_manage.Module.detector import Detector

from conditional_flow_matching.Method.time import getCurrentTime
from conditional_flow_matching.Module.sampler import Sampler


def demoCondition(
    sampler: Sampler,
    detector: Detector,
    time_stamp: str,
    condition_value: Union[int, str, np.ndarray] = 18,
    sample_num: int = 9,
    save_folder_path: Union[str, None] = None,
    condition_type: str = 'category',
    condition_name: str = '0'):
    assert condition_type in ['category', 'image', 'points', 'text']

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

        condition = (
            detector.encodeImageFile(image_file_path).cpu().numpy()
        )
    elif condition_type == 'points':
        assert isinstance(condition_value, np.ndarray)

        points = condition_value
        condition = (
            detector.encodePointCloud(points).cpu().numpy()
        )
    elif condition_type == 'text':
        assert isinstance(condition_value, str)

        text = condition_value
        condition = (
            detector.encodeText(condition_value).cpu().numpy()
        )
    else:
        print('[ERROR][sampler::demoCondition]')
        print('\t condition type not valid!')
        return False

    condition_info = condition_type + '/' + condition_name

    print("start diffuse", sample_num, "mashs....")
    sampled_array = sampler.sample(sample_num, condition)

    object_dist = [0, 0, 0]

    row_num = ceil(sqrt(sample_num))

    mash_model = sampler.toInitialMashModel('cpu')

    for j in range(sampled_array.shape[0]):
        if j != sampled_array.shape[0] -  1:
            continue

        if save_folder_path is None:
            save_folder_path = './output/sample/' + time_stamp + '/iter-' + str(j) + '/'

        save_folder_path += condition_info + '/'

        os.makedirs(save_folder_path, exist_ok=True)

        if condition_type == 'image':
            copyfile(image_file_path, save_folder_path + 'condition_image.png')
        elif condition_type == 'points':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(save_folder_path + 'condition_points.ply', pcd)
        elif condition_type == 'text':
            with open(save_folder_path + 'condition_text.txt', 'w') as f:
                f.write(text)

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

    return True

def demo(save_folder_path: Union[str, None] = None):
    cfm_model_file_path = './output/24depth_512cond_2000epoch/total_model_last.pth'
    use_ema = True
    sample_num = 100
    device = 'cuda:0'

    ulip_model_file_path = '/home/chli/chLi/Model/ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = '/home/chli/Model/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'

    sampler = Sampler(cfm_model_file_path, use_ema, device)
    detector = Detector(ulip_model_file_path, open_clip_model_file_path, device)

    time_stamp = getCurrentTime()

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
    for categoty_id in range(55):
    # for categoty_id in [0, 2, 6, 18, 22, 23, 24, 26, 30, 46, 47, 49, 53]:
    # for categoty_id in [0, 18]:
        print('start sample for category ' + str(categoty_id) + '...')
        demoCondition(sampler, detector, time_stamp, categoty_id, sample_num, save_folder_path, 'category', str(categoty_id))

    def toRandomIdList(dataset_folder_path: str, valid_category_id_list: Union[list, None]=None, sample_id_num: int=100) -> list:
        random_id_list = []

        if valid_category_id_list is None:
            valid_category_id_list = os.listdir(dataset_folder_path)

        for category_id in valid_category_id_list:
            category_folder_path = dataset_folder_path + category_id + '/'
            if not os.path.exists(category_folder_path):
                continue

            model_id_list = os.listdir(category_folder_path)

            if sample_id_num >= len(model_id_list):
                random_model_id_list = model_id_list
            else:
                random_model_id_list = random.sample(model_id_list, sample_id_num)

            for random_model_id in random_model_id_list:
                random_id_list.append(category_id + '/' + random_model_id.replace('.npy', ''))

        return random_id_list

    image_id_list = [
        '03001627/1a74a83fa6d24b3cacd67ce2c72c02e',
        '03001627/1a38407b3036795d19fb4103277a6b93',
        '03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7',
        '02691156/1a6ad7a24bb89733f412783097373bdc',
        '02691156/1a32f10b20170883663e90eaf6b4ca52',
        '02691156/1abe9524d3d38a54f49a51dc77a0dd59',
        '02691156/1adb40469ec3636c3d64e724106730cf',
    ]
    image_id_list = toRandomIdList('/home/chli/Dataset/MashV4/ShapeNet/', None, sample_num)
    for image_id in image_id_list:
        print('start sample for image ' + image_id + '...')
        image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/' + image_id + '/y_5_x_3.png'
        if not os.path.exists(image_file_path):
            continue
        demoCondition(sampler, detector, time_stamp, image_file_path, sample_num, save_folder_path, 'image', image_id)

    points_id_list = [
        '03001627/1a74a83fa6d24b3cacd67ce2c72c02e',
        '03001627/1a38407b3036795d19fb4103277a6b93',
        '03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7',
        '02691156/1a6ad7a24bb89733f412783097373bdc',
        '02691156/1a32f10b20170883663e90eaf6b4ca52',
        '02691156/1abe9524d3d38a54f49a51dc77a0dd59',
        '02691156/1adb40469ec3636c3d64e724106730cf',
    ]
    points_id_list = toRandomIdList('/home/chli/Dataset/MashV4/ShapeNet/', None, sample_num)
    for points_id in points_id_list:
        print('start sample for points ' + points_id + '...')
        mesh_file_path = '/home/chli/chLi/Dataset/ManifoldMesh/ShapeNet/' + points_id + '.obj'
        if not os.path.exists(mesh_file_path):
            continue
        points = Mesh(mesh_file_path).toSamplePoints(8192)
        demoCondition(sampler, detector, time_stamp, points, sample_num, save_folder_path, 'points', points_id)

    text_list = [
        'a tall chair',
        'a short chair',
        'a circle chair',
        'horizontal slats on top of back',
        'one big hole between back and seat',
        'this chair has wheels',
        'vertical back ribs',
    ]
    for i, text in enumerate(text_list):
        print('start sample for text [' + text + ']...')
        demoCondition(sampler, detector, time_stamp, text, sample_num, save_folder_path, 'text', str(i))

    return True
