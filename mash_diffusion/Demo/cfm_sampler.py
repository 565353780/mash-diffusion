import sys

sys.path.append("../ma-sh/")
sys.path.append("../wn-nc/")
sys.path.append("../ulip-manage/")
sys.path.append("../dino-v2-detect/")
sys.path.append("../mash-occ-decoder/")
sys.path.append('../distribution-manage/')

import os
import random
from typing import Union

from ma_sh.Config.custom_path import toModelRootPath
from ma_sh.Data.mesh import Mesh

from mash_diffusion.Config.shapenet import CATEGORY_IDS
from mash_diffusion.Method.time import getCurrentTime
from mash_diffusion.Module.cfm_sampler import CFMSampler


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

def demo():
    model_root_path = toModelRootPath()
    assert model_root_path is not None

    cfm_model_file_path = "../../output/cfm-ShapeNet_03001627/model_last.pth".replace('../../', './')
    occ_model_file_path = '../../../mash-occ-decoder/output/512dim-v4/model_best.pth'.replace('../../', './')
    use_ema = True
    device = 'cuda:0'
    transformer_id = 'ShapeNet_03001627'
    ulip_model_file_path = model_root_path + 'ULIP2/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt'
    open_clip_model_file_path = 'CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
    dino_model_file_path = 'DINOv2/dinov2_vitl14_reg4_pretrain.pth'

    save_folder_path = './output/sample/' + getCurrentTime() + '/'
    sample_id_num = 1
    sample_num = 10
    timestamp_num = 2
    sample_category = True
    sample_image = False
    sample_points = False
    sample_text = False
    sample_fixed_anchor = True
    save_results_only = True

    #FIXME: deactivate detectors for fast test only
    ulip_model_file_path = None
    dino_model_file_path = None

    cfm_sampler = CFMSampler(
        cfm_model_file_path,
        occ_model_file_path,
        use_ema,
        device,
        transformer_id,
        ulip_model_file_path,
        open_clip_model_file_path,
        dino_model_file_path,
    )

    valid_category_id_list = [
        '02691156', # 0: airplane
        '02773838', # 2: bag
        '02828884', # 6: bench
        '02876657', # 9: bottle
        '02958343', # 16: bottle
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
    valid_category_id_list = [
        '03001627', # 18: chair
    ]

    if sample_category:
        for categoty_id in valid_category_id_list:
            print('start sample for category ' + categoty_id + '...')
            cfm_sampler.samplePipeline(
                save_folder_path + 'category/' + categoty_id + '/',
                'category',
                CATEGORY_IDS[categoty_id],
                sample_num,
                timestamp_num,
                save_results_only)

    if sample_image:
        image_id_list = [
            '03001627/1a74a83fa6d24b3cacd67ce2c72c02e',
            '03001627/1a38407b3036795d19fb4103277a6b93',
            '03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7',
            '02691156/1a6ad7a24bb89733f412783097373bdc',
            '02691156/1a32f10b20170883663e90eaf6b4ca52',
            '02691156/1abe9524d3d38a54f49a51dc77a0dd59',
            '02691156/1adb40469ec3636c3d64e724106730cf',
        ]
        image_id_list = toRandomIdList('/home/chli/Dataset/MashV4/ShapeNet/', valid_category_id_list, sample_id_num)
        for image_id in image_id_list:
            print('start sample for image ' + image_id + '...')
            image_file_path = '/home/chli/chLi/Dataset/CapturedImage/ShapeNet/' + image_id + '/y_5_x_3.png'
            if not os.path.exists(image_file_path):
                continue
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-image/' + image_id + '/',
                'ulip-image',
                image_file_path,
                sample_num,
                timestamp_num,
                save_results_only)

    if sample_points:
        points_id_list = [
            '03001627/1a74a83fa6d24b3cacd67ce2c72c02e',
            '03001627/1a38407b3036795d19fb4103277a6b93',
            '03001627/1ab8a3b55c14a7b27eaeab1f0c9120b7',
            '02691156/1a6ad7a24bb89733f412783097373bdc',
            '02691156/1a32f10b20170883663e90eaf6b4ca52',
            '02691156/1abe9524d3d38a54f49a51dc77a0dd59',
            '02691156/1adb40469ec3636c3d64e724106730cf',
        ]
        points_id_list = toRandomIdList('/home/chli/Dataset/MashV4/ShapeNet/', valid_category_id_list, sample_id_num)
        for points_id in points_id_list:
            print('start sample for points ' + points_id + '...')
            mesh_file_path = '/home/chli/chLi/Dataset/ManifoldMesh/ShapeNet/' + points_id + '.obj'
            if not os.path.exists(mesh_file_path):
                continue
            points = Mesh(mesh_file_path).toSamplePoints(8192)
            cfm_sampler.samplePipeline(
                save_folder_path + 'ulip-points/' + points_id + '/',
                'ulip-points',
                points,
                sample_num,
                timestamp_num,
                save_results_only)

    if sample_text:
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
                sample_num,
                timestamp_num,
                save_results_only)

    if sample_fixed_anchor:
        mash_file_path_list = [
            '../ma-sh/output/combined_mash.npy',
        ]
        categoty_id = '03001627'
        print('start sample for fixed anchor category ' + categoty_id + '...')
        cfm_sampler.samplePipeline(
            save_folder_path + 'category-fixed-anchors/' + categoty_id + '/',
            'category',
            CATEGORY_IDS[categoty_id],
            sample_num,
            timestamp_num,
            save_results_only,
            mash_file_path_list)
    return True
