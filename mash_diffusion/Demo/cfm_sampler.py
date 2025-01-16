import sys

sys.path.append("../ma-sh/")
sys.path.append("../wn-nc/")
sys.path.append("../ulip-manage/")
sys.path.append('../blender-manage/')
sys.path.append("../dino-v2-detect/")
sys.path.append("../mash-occ-decoder/")
sys.path.append('../distribution-manage/')

import os
import random
from tqdm import tqdm
from typing import Union

from ma_sh.Config.custom_path import toDatasetRootPath, toModelRootPath
from ma_sh.Data.mesh import Mesh

from mash_diffusion.Config.shapenet import CATEGORY_IDS
from mash_diffusion.Method.time import getCurrentTime
from mash_diffusion.Module.cfm_sampler import CFMSampler


def toRandomRelBasePathList(dataset_folder_path: str,
                            data_type: str = '.npy',
                            valid_category_id_list: Union[list, None]=None,
                            per_category_sample_condition_num: int=100) -> list:
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

                rel_base_path_list.append(rel_folder_path + file[:-len(data_type)])

        if per_category_sample_condition_num >= len(rel_base_path_list):
            random_rel_base_path_list += rel_base_path_list
        else:
            random_rel_base_path_list += random.sample(rel_base_path_list, per_category_sample_condition_num)

    return random_rel_base_path_list

def demo():
    dataset_root_path = toDatasetRootPath()
    model_root_path = toModelRootPath()
    assert dataset_root_path is not None
    assert model_root_path is not None

    # this will decide the sample mode
    # available ids: ['Objaverse_82K', 'ShapeNet']
    # available sample modes:
    # Objaverse_82K: sample_dino
    # ShapeNet: sample_category, sample_ulip_image, sample_ulip_points, sample_ulip_text, sample_fixed_anchor
    transformer_id = 'Objaverse_82K'

    if transformer_id == 'Objaverse_82K':
        cfm_model_file_path = model_root_path + 'MashDiffusion/cfm-Objaverse_82K-single_image-0116/model_last.pth'
    elif transformer_id == 'ShapeNet':
        cfm_model_file_path = model_root_path + 'MashDiffusion/cfm-ShapeNet-multi_modal-0116/model_last.pth'
    else:
        print('transformer id not valid!')
        return False

    occ_model_file_path = model_root_path + 'MashOCCDecoder/noise_1-0116/model_last.pth'
    cfm_use_ema = False
    occ_use_ema = False
    device = 'cuda:0'
    ulip_model_file_path = model_root_path + 'ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = model_root_path + 'CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
    dino_model_file_path = model_root_path + 'DINOv2/dinov2_vitb14_reg4_pretrain.pth'

    save_folder_path = '/home/chli/chLi/Results/mash-diffusion/output/sample/' + getCurrentTime() + '/'
    objaverse_per_category_sample_condition_num = 1
    shapenet_per_category_sample_condition_num = 20
    sample_conditioned_shape_num = 20
    timestamp_num = 2
    sample_dino = transformer_id == 'Objaverse_82K'
    sample_category = False and (transformer_id == 'ShapeNet')
    sample_ulip_image = False and (transformer_id == 'ShapeNet')
    sample_ulip_points = False and (transformer_id == 'ShapeNet')
    sample_ulip_text = False and (transformer_id == 'ShapeNet')
    sample_fixed_anchor = True and (transformer_id == 'ShapeNet')
    save_results_only = True

    if not sample_dino:
        dino_model_file_path = None
    if not sample_ulip_image and not sample_ulip_points and not sample_ulip_text:
        ulip_model_file_path = None

    valid_objaverse_category_id_list = [
        '000-' + str(i).zfill(3) for i in range(160)
    ]
    valid_shapenet_category_id_list = [
        '02691156', # 0: airplane
        '02773838', # 2: bag
        '02828884', # 6: bench
        '02876657', # 9: bottle
        '02958343', # 16: car
        '03001627', # 18: chair
        '03211117', # 22: monitor
        '03261776', # 23: earphone
        '03325088', # 24: spigot
        '03467517', # 26: guitar
        '03513137', # 27: helmet
        '03636649', # 30: lamp
        '03710193', # 33: mailbox
        '03948459', # 40: gun
        '04090263', # 44: long-gun
        '04225987', # 46: skateboard
        '04256520', # 47: sofa
        '04379243', # 49: table
        '04468005', # 52: train
        '04530566', # 53: watercraft
    ]
    valid_shapenet_category_id_list = [
        '02691156', # 0: airplane
        '02958343', # 16: car
        '03001627', # 18: chair
    ]

    tmp_dataset_root_path = '/home/chli/chLi2/Dataset/'

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        occ_model_file_path,
        cfm_use_ema,
        occ_use_ema,
        device,
        transformer_id,
        ulip_model_file_path,
        open_clip_model_file_path,
        dino_model_file_path,
    )

    if sample_dino:
        condition_root_folder_path = dataset_root_path + 'Objaverse_82K/render_jpg/'
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
                sample_conditioned_shape_num,
                timestamp_num,
                save_results_only)

    if sample_category:
        for categoty_id in valid_shapenet_category_id_list:
            print('start sample for category ' + categoty_id + '...')
            cfm_sampler.samplePipeline(
                save_folder_path + 'category/' + categoty_id + '/',
                'category',
                CATEGORY_IDS[categoty_id],
                sample_conditioned_shape_num,
                timestamp_num,
                save_results_only)

    if sample_ulip_image:
        condition_root_folder_path = tmp_dataset_root_path + 'CapturedImage/ShapeNet/'
        data_type = '.png'

        rel_base_path_list = toRandomRelBasePathList(condition_root_folder_path,
                                                     data_type,
                                                     valid_objaverse_category_id_list,
                                                     shapenet_per_category_sample_condition_num)
        for rel_base_path in rel_base_path_list:
            print('start sample for condition ' + rel_base_path + '...')
            condition_file_path = condition_root_folder_path + rel_base_path + data_type
            if not os.path.exists(condition_file_path):
                continue
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-condition/' + rel_base_path + '/',
                'ulip-condition',
                condition_file_path,
                sample_conditioned_shape_num,
                timestamp_num,
                save_results_only)

    if sample_ulip_points:
        condition_root_folder_path = tmp_dataset_root_path + 'ManifoldMesh/ShapeNet/'
        data_type = '.obj'

        rel_base_path_list = toRandomRelBasePathList(condition_root_folder_path,
                                                     data_type,
                                                     valid_objaverse_category_id_list,
                                                     shapenet_per_category_sample_condition_num)
        for rel_base_path in rel_base_path_list:
            print('start sample for condition ' + rel_base_path + '...')
            condition_file_path = condition_root_folder_path + rel_base_path + data_type
            if not os.path.exists(condition_file_path):
                continue
            points = Mesh(condition_file_path).toSamplePoints(8192)
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-points/' + rel_base_path + '/',
                'ulip-points',
                points,
                sample_conditioned_shape_num,
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
                sample_conditioned_shape_num,
                timestamp_num,
                save_results_only)

    if sample_fixed_anchor:
        mash_file_path_list = [
            '/home/chli/chLi/Results/ma-sh/output/combined_mash.npy',
        ]
        categoty_id = '03001627'
        print('start sample for fixed anchor category ' + categoty_id + '...')
        cfm_sampler.samplePipeline(
            save_folder_path + 'category-fixed-anchors/' + categoty_id + '/',
            'category',
            CATEGORY_IDS[categoty_id],
            sample_conditioned_shape_num,
            timestamp_num,
            save_results_only,
            mash_file_path_list)

    cfm_sampler.waitRender()

    return True
