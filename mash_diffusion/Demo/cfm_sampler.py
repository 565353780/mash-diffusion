import sys

sys.path.append("../ma-sh/")
sys.path.append("../wn-nc/")
sys.path.append("../ulip-manage/")
sys.path.append('../blender-manage/')
sys.path.append("../dino-v2-detect/")
sys.path.append("../mash-occ-decoder/")
sys.path.append('../distribution-manage/')

import os
import torch
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union

from ma_sh.Config.custom_path import toDatasetRootPath, toModelRootPath
from ma_sh.Data.mesh import Mesh

from mash_diffusion.Config.shapenet import CATEGORY_IDS
from mash_diffusion.Method.time import getCurrentTime
from mash_diffusion.Module.cfm_sampler import CFMSampler


def toRandomRelBasePathList(
    dataset_folder_path: str,
    data_type: str = '.npy',
    valid_category_id_list: Union[list, None]=None,
    per_category_sample_condition_num: int=100,
    filter_tag: Union[str, None]=None,
) -> list:
    random_rel_base_path_list = []

    if valid_category_id_list is None:
        valid_category_id_list = os.listdir(dataset_folder_path)

    print('[INFO][cfm_sampler::toRandomRelBasePathList]')
    print('\t start collect random rel base path from dataset...')
    for category_id in tqdm(valid_category_id_list):
        category_folder_path = dataset_folder_path + category_id + '/'
        if not os.path.exists(category_folder_path):
            continue

        rel_base_path_list = []

        for root, _, files in os.walk(category_folder_path):
            rel_folder_path = os.path.relpath(root, dataset_folder_path) + '/'

            for file in files:
                if not file.endswith(data_type) or file.endswith('_tmp' + data_type):
                    continue

                if filter_tag is not None:
                    if filter_tag not in file:
                        continue

                rel_base_path_list.append(rel_folder_path + file[:-len(data_type)])

        if per_category_sample_condition_num >= len(rel_base_path_list):
            random_rel_base_path_list += rel_base_path_list
        else:
            random_rel_base_path_list += random.sample(rel_base_path_list, per_category_sample_condition_num)

    return random_rel_base_path_list

def demo_dino():
    dataset_root_path = toDatasetRootPath()
    model_root_path = toModelRootPath()
    assert dataset_root_path is not None
    assert model_root_path is not None

    transformer_id = 'Objaverse_82K'

    cfm_model_file_path = model_root_path + 'MashDiffusion/cfm-Objaverse_82K-single_image-0122/model_last.pth'
    occ_model_file_path = model_root_path + 'MashOCCDecoder/noise_1-0118/model_best.pth'
    cfm_use_ema = True
    occ_use_ema = True
    device = 'cuda:0'
    dino_model_file_path = model_root_path + 'DINOv2/dinov2_vitb14_reg4_pretrain.pth'

    occ_batch_size = 1200000 # 24G GPU Memory required
    # occ_batch_size = 500000 # 12G GPU Memory required

    save_folder_path = '/home/chli/chLi/Results/mash-diffusion/output/sample/' + getCurrentTime() + '/'

    objaverse_per_category_sample_condition_num = 100
    objaverse_sample_batch_size = 4

    timestamp_num = 2
    save_results_only = True

    recon_wnnc = False
    recon_occ = True
    render_pcd = False

    smooth_wnnc = True and recon_wnnc
    smooth_occ = True and recon_occ
    render_wnnc = True and recon_wnnc
    render_wnnc_smooth = True and recon_wnnc and smooth_wnnc
    render_occ = False and recon_occ
    render_occ_smooth = False and recon_occ and smooth_occ

    valid_objaverse_category_id_list = [
        '000-' + str(i).zfill(3) for i in range(160)
    ]

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        occ_model_file_path,
        cfm_use_ema,
        occ_use_ema,
        device,
        transformer_id,
        None,
        None,
        dino_model_file_path,
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

    condition_root_folder_path = dataset_root_path + 'Objaverse_82K/render_jpg_v2/'
    data_type = '.jpg'

    rel_base_path_list = toRandomRelBasePathList(condition_root_folder_path,
                                                    data_type,
                                                    valid_objaverse_category_id_list,
                                                    objaverse_per_category_sample_condition_num)
    for rel_base_path in rel_base_path_list:
        print('start sample for condition ' + rel_base_path + '...')
        condition_file_path = condition_root_folder_path + rel_base_path + data_type
        if not os.path.exists(condition_file_path):
            continue
        cfm_sampler.samplePipeline(
            save_folder_path + 'dino/' + rel_base_path + '/',
            'dino',
            condition_file_path,
            objaverse_sample_batch_size,
            timestamp_num,
            save_results_only)

    cfm_sampler.waitRender()

    torch.cuda.empty_cache()
    return True

def demo_multi_modal():
    dataset_root_path = toDatasetRootPath()
    model_root_path = toModelRootPath()
    assert dataset_root_path is not None
    assert model_root_path is not None

    transformer_id = 'ShapeNet'

    cfm_model_file_path = model_root_path + 'MashDiffusion/cfm-ShapeNet-multi_modal-0122/model_last.pth'
    occ_model_file_path = model_root_path + 'MashOCCDecoder/noise_1-0118/model_last.pth'
    cfm_use_ema = True
    occ_use_ema = True
    device = 'cuda:0'
    ulip_model_file_path = model_root_path + 'ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = model_root_path + 'CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'

    occ_batch_size = 1200000 # 24G GPU Memory required
    # occ_batch_size = 500000 # 12G GPU Memory required

    save_folder_path = '/home/chli/chLi/Results/mash-diffusion/output/sample/' + getCurrentTime() + '/'

    shapenet_per_category_sample_multi_modal_condition_num = 30
    shapenet_category_sample_batch_size = 10
    shapenet_multi_modal_sample_batch_size = 10

    timestamp_num = 2
    sample_category = False
    sample_ulip_image = False
    sample_ulip_points = True
    sample_ulip_text = False
    sample_fixed_anchor = False
    sample_combined_anchor = False
    save_results_only = True

    render_pcd = False

    recon_wnnc = False
    smooth_wnnc = True and recon_wnnc
    render_wnnc = False and recon_wnnc
    render_wnnc_smooth = True and recon_wnnc and smooth_wnnc

    recon_occ = True
    smooth_occ = True and recon_occ
    render_occ = False and recon_occ
    render_occ_smooth = True and recon_occ and smooth_occ

    if not sample_ulip_image and not sample_ulip_points and not sample_ulip_text:
        ulip_model_file_path = None

    valid_shapenet_category_id_list = [
        #'02691156', # 0: airplane
        #'02773838', # 2: bag
        #'02828884', # 6: bench
        #'02876657', # 9: bottle
        #'02958343', # 16: car
        '03001627', # 18: chair
        #'03211117', # 22: monitor
        #'03261776', # 23: earphone
        #'03325088', # 24: spigot
        #'03467517', # 26: guitar
        #'03636649', # 30: lamp
        #'03948459', # 40: gun
        #'04090263', # 44: long-gun
        #'04256520', # 47: sofa
        #'04379243', # 49: table
        #'04468005', # 52: train
        #'04530566', # 53: watercraft
    ]

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        occ_model_file_path,
        cfm_use_ema,
        occ_use_ema,
        device,
        transformer_id,
        ulip_model_file_path,
        open_clip_model_file_path,
        None,
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

    if sample_category:
        for categoty_id in valid_shapenet_category_id_list:
            print('start sample for category ' + categoty_id + '...')
            cfm_sampler.samplePipeline(
                save_folder_path + 'category/' + categoty_id + '/',
                'category',
                CATEGORY_IDS[categoty_id],
                shapenet_category_sample_batch_size,
                timestamp_num,
                save_results_only)

    if sample_ulip_image:
        condition_root_folder_path = dataset_root_path + 'CapturedImage/ShapeNet/'
        data_type = '.png'

        rel_base_path_list = toRandomRelBasePathList(
            condition_root_folder_path,
            data_type,
            valid_shapenet_category_id_list,
            shapenet_per_category_sample_multi_modal_condition_num,
            'y_5_x_3',
        )
        for rel_base_path in rel_base_path_list:
            print('start sample for condition ' + rel_base_path + '...')
            condition_file_path = condition_root_folder_path + rel_base_path + data_type
            if not os.path.exists(condition_file_path):
                continue
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-image/' + rel_base_path + '/',
                'ulip-image',
                condition_file_path,
                shapenet_multi_modal_sample_batch_size,
                timestamp_num,
                save_results_only)

    if sample_ulip_points:
        condition_root_folder_path = dataset_root_path + 'ShapeNet/manifold_pcd-512_nonuniform/'
        data_type = '.ply'

        rel_base_path_list = toRandomRelBasePathList(condition_root_folder_path,
                                                     data_type,
                                                     valid_shapenet_category_id_list,
                                                     shapenet_per_category_sample_multi_modal_condition_num)
        for rel_base_path in rel_base_path_list:
            '''
            valid = False
            if '/2c052f' in rel_base_path:
                valid = True
            if '/3d4423' in rel_base_path:
                valid = True
            if '/4c1777' in rel_base_path:
                valid = True
            if '/8be809' in rel_base_path:
                valid = True
            if '/38c67c' in rel_base_path:
                valid = True

            if not valid:
                continue
            '''

            print('start sample for condition ' + rel_base_path + '...')
            condition_file_path = condition_root_folder_path + rel_base_path + data_type
            if not os.path.exists(condition_file_path):
                continue
            #points = Mesh(condition_file_path).toSamplePoints(8192)
            points = np.asarray(o3d.io.read_point_cloud(condition_file_path).points)
            points = np.vstack([points, points])
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-points/' + rel_base_path + '/',
                'ulip-points',
                points,
                shapenet_multi_modal_sample_batch_size,
                timestamp_num,
                save_results_only)

    if sample_ulip_text:
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
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-text/' + str(i) + '/',
                'ulip-text',
                text,
                shapenet_multi_modal_sample_batch_size,
                timestamp_num,
                save_results_only)

    part_mash_folder_path = '/home/chli/chLi/Results/ma-sh/output/part_mash/'
    mash_id_list_1 = [
        '02691156/595556bad291028733de69c9cd670995',
        '02691156/166d333d38897d1513d521050081b441',
        '02691156/cc9b7118034278fcb4cdad9a5bf52dd5',
        '02691156/73f6ccf1468de18d381fd507da445af6',
        '02691156/a75ab6e99a3542eb203936772104a82d',
        '02691156/6f96517661cf1b6799ed03445864bd37',
    ]

    mash_id_list_2 = [
        '02828884/fc3865756db954685896bab37ddebe7',

        #'02828884/bdc3a9776cd0d69b26abe89c4547d5f1',
        #'02828884/d8b87f36fde3f2f3bc5996932c1238cd',
    ]

    mash_id_list_3 = [
        '03001627/d3302b7fa6504cab1a461b43b8f257f',
        '03001627/a75e83a3201cf5ac745004c6a29b0df0',
        '03001627/d3ff300de7ab36bfc8528ab560ff5e59',
        '03001627/d29445f24bbf1b1814c05b481f895c37',
        #'03001627/433c6c88f1a43ab73ebe788797b18766',
        #'03001627/9d9b5f5b3fd41136244d7c2690850fc2',
        #'03001627/f2af2483f9fb980cb237f85c0ae7ac77',
        #'03001627/3c27660aacbcf99886327adaa986dff',
        #'03001627/7ae6518311bf2f66e1a0327ca4c4d5a5',
    ]

    mash_id_list = mash_id_list_1

    mash_rel_path_list = []
    for mash_id in mash_id_list:
        curr_part_mash_folder_path = part_mash_folder_path + mash_id + '/'
        curr_part_mash_filename_list = os.listdir(curr_part_mash_folder_path)
        if len(curr_part_mash_filename_list) == 0:
            continue

        for file in curr_part_mash_filename_list:
            if not file.endswith('.npy') or not file.startswith('part'):
                continue

            mash_rel_path_list.append(mash_id + '/' + file)
            break

    if sample_fixed_anchor:
        for mash_rel_path in mash_rel_path_list:
            print('start sample for fixed anchor for ' + mash_rel_path + '...')
            mash_file_path = part_mash_folder_path + mash_rel_path
            if not os.path.exists(mash_file_path):
                continue
            category_id = mash_rel_path.split('/')[0]
            mash_id = mash_rel_path.split('/')[1]
            cfm_sampler.samplePipeline(
                save_folder_path + 'category-fixed-anchors/' + category_id + '/' + mash_id + '/',
                'category',
                CATEGORY_IDS[category_id],
                shapenet_multi_modal_sample_batch_size,
                timestamp_num,
                save_results_only,
                [mash_file_path])

    if sample_combined_anchor:
        combined_mash_ids_list = [
            #['02691156/5955', '02691156/166d'],
            #['02691156/5955', '02691156/73f6'],
            #['02691156/5955', '02691156/cc96'],
            ['02691156/5955', '02691156/6f96'],
            ['02691156/5955', '02691156/a75a'],
            #['02691156/73f6', '02691156/6f96'],

            ['02828884/bdc3', '02828884/fc38'],
            ['02828884/bdc3', '02828884/d8b8'],
            ['02828884/fc38', '02828884/d8b8'],

            ['03001627/d294', '03001627/a75e'],
            ['03001627/d294', '03001627/9d9b'],
            ['03001627/d294', '03001627/7ae6'],
            ['03001627/d294', '03001627/3c27'],
            ['03001627/433c', '03001627/9d9b'],
            ['03001627/433c', '03001627/a75e'],
            ['03001627/433c', '03001627/3c27'],
            ['03001627/433c', '03001627/f2af'],
        ]
        for i, combined_mash_ids in enumerate(combined_mash_ids_list):
            print('start sample for combined anchor ' + str(i) + '...')
            mash_file_path_list = [
                    part_mash_folder_path + mash_rel_path for mash_rel_path in mash_rel_path_list
                if mash_rel_path.startswith(combined_mash_ids[0]) or mash_rel_path.startswith(combined_mash_ids[1])
            ]
            if len(mash_file_path_list) < 2:
                continue
            category_id = combined_mash_ids[0].split('/')[0]
            mash_id = str(i)
            cfm_sampler.samplePipeline(
                save_folder_path + 'category-combined-anchors/' + category_id + '/' + mash_id + '/',
                'category',
                CATEGORY_IDS[category_id],
                shapenet_multi_modal_sample_batch_size,
                timestamp_num,
                save_results_only,
                mash_file_path_list)

    cfm_sampler.waitRender()

    torch.cuda.empty_cache()
    return True

def demo():
    # demo_dino()
    demo_multi_modal()
