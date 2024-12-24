import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor

from distribution_manage.Module.transformer import Transformer


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        embedding_folder_name_dict: dict,
        split: str = "train",
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split

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

                if not all_embedding_exist:
                    continue

                for key, embedding_folder_path in self.embedding_folder_path_dict.items():
                    embedding_file_path = embedding_folder_path + collection_id + '/' + mash_filename

                    path_dict['mash'] = mash_file_path
                    path_dict['embedding'][key] = embedding_file_path


                self.path_dict_list.append(path_dict)

        self.transformer = Transformer('../ma-sh/output/multi_linear_transformers.pkl')
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return len(self.path_dict_list)

    def __getitem__(self, index):
        index = index % len(self.path_dict_list)

        data = {}

        path_dict = self.path_dict_list[index]

        mash_file_path = path_dict['mash']
        embedding_file_path_dict = path_dict['embedding']
        embedding_dict = {}
        for key, embedding_file_path in embedding_file_path_dict.items():
            embedding = np.load(embedding_file_path, allow_pickle=True).item()
            embedding_dict[key] = embedding

        mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        mash_params = self.normalize(mash_params)

        data['mash_params'] = mash_params

        random_embedding_tensor_dict = {}

        for key, embedding in embedding_dict.items():
            embedding_key_idx = np.random.choice(len(embedding.keys()))
            embedding_key = list(embedding.keys())[embedding_key_idx]
            random_embedding_tensor_dict[key] = torch.from_numpy(embedding[embedding_key]).float()

        data['embedding'] = random_embedding_tensor_dict
        return data
