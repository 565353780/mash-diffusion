import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ma_sh.Model.mash import Mash


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        embedding_folder_name_dict: dict,
        split: str = "train",
        preload: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.preload = preload

        self.mash_folder_path = self.dataset_root_folder_path + "Objaverse_82K/mash/"
        self.embedding_folder_path_dict = {}
        for key, embedding_folder_name in embedding_folder_name_dict.items():
            self.embedding_folder_path_dict[key] = self.dataset_root_folder_path + embedding_folder_name + "/"

        assert os.path.exists(self.mash_folder_path)
        for embedding_folder_path in self.embedding_folder_path_dict.values():
            assert os.path.exists(embedding_folder_path)

        self.path_dict_list = []

        collection_id_list = os.listdir(self.mash_folder_path)

        print("[INFO][EmbeddingDataset::__init__]")
        print("\t start load dataset collections...")
        for collection_id in tqdm(collection_id_list):
            collection_folder_path = self.mash_folder_path + collection_id + "/"

            mash_filename_list = os.listdir(collection_folder_path)

            for mash_filename in mash_filename_list:
                path_dict = {
                    'embedding': {},
                }
                mash_file_path = collection_folder_path + mash_filename

                if not os.path.exists(mash_file_path):
                    continue

                all_embedding_exist = True

                for key, embedding_folder_path in self.embedding_folder_path_dict.items():
                    embedding_file_path = embedding_folder_path + collection_id + '/' + mash_filename

                    if not os.path.exists(embedding_file_path):
                        all_embedding_exist = False
                        break

                    if self.preload:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        embedding = np.load(embedding_file_path, allow_pickle=True).item()
                        path_dict['mash'] = mash_params
                        path_dict['embedding'][key] = embedding
                    else:
                        path_dict['mash'] = mash_file_path
                        path_dict['embedding'][key] = embedding_file_path

                if not all_embedding_exist:
                    continue

                self.path_dict_list.append(path_dict)
        return

    def __len__(self):
        return len(self.path_dict_list)

    def __getitem__(self, index):
        index = index % len(self.path_dict_list)

        data = {}

        path_dict = self.path_dict_list[index]

        if self.preload:
            mash_params = path_dict['mash']
            embedding_dict = path_dict['embedding']
        else:
            mash_file_path = path_dict['mash']
            embedding_file_path_dict = path_dict['embedding']
            mash_params = np.load(mash_file_path, allow_pickle=True).item()
            embedding_dict = {}
            for key, embedding_file_path in embedding_file_path_dict.items():
                embedding = np.load(embedding_file_path, allow_pickle=True).item()
                embedding_dict[key] = embedding

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        '''
        if self.split == "train":
            scale_range = [0.8, 1.2]
            move_range = [-0.2, 0.2]

            random_scale = (
                scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
            )
            random_translate = move_range[0] + (
                move_range[1] - move_range[0]
            ) * np.random.rand(3)

            positions = positions * random_scale + random_translate
            sh_params = sh_params * random_scale
        '''

        permute_idxs = np.random.permutation(rotate_vectors.shape[0])

        rotate_vectors = rotate_vectors[permute_idxs]
        positions = positions[permute_idxs]
        mask_params = mask_params[permute_idxs]
        sh_params = sh_params[permute_idxs]

        mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')
        mash.loadParams(mask_params, sh_params, rotate_vectors, positions)

        ortho_poses_tensor = mash.toOrtho6DPoses().float()
        positions_tensor = torch.tensor(positions).float()
        mask_params_tesnor = torch.tensor(mask_params).float()
        sh_params_tensor = torch.tensor(sh_params).float()

        cfm_mash_params = torch.cat((ortho_poses_tensor, positions_tensor, mask_params_tesnor, sh_params_tensor), dim=1)

        data['cfm_mash_params'] = cfm_mash_params

        random_embedding_tensor_dict = {}

        for key, embedding in embedding_dict.items():
            embedding_key_idx = np.random.choice(len(embedding.keys()))
            embedding_key = list(embedding.keys())[embedding_key_idx]
            random_embedding_tensor_dict[key] = torch.from_numpy(embedding[embedding_key]).float()

        data['embedding'] = random_embedding_tensor_dict
        return data
