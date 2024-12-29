import os
import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Method.io import loadMashFileParamsTensor
from ma_sh.Method.transformer import getTransformer


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        embedding_folder_name: str,
        embedding_key: str,
        split: str = "train",
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.embedding_key = embedding_key
        self.split = split

        self.mash_folder_path = self.dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
        self.embedding_root_folder_path = self.dataset_root_folder_path + embedding_folder_name + "/"

        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.embedding_root_folder_path)

        self.paths_list = []

        print("[INFO][EmbeddingDataset::__init__]")
        print("\t start load mash and embedding datasets...")
        for root, _, files in os.walk(self.mash_folder_path):
            rel_folder_path = os.path.relpath(root, self.mash_folder_path)

            for file in files:
                if not file.endswith('.npy'):
                    continue

                mash_file_path = root + '/' + file

                embedding_folder_path = self.embedding_root_folder_path + rel_folder_path + file[:-4] + '/'

                if not os.path.exists(embedding_folder_path):
                    continue

                embedding_filename_list = os.listdir(embedding_folder_path)

                if len(embedding_filename_list) == 0:
                    continue

                embedding_file_path_list = []
                for embedding_filename in embedding_filename_list:
                    if not embedding_filename.endswith('.npy'):
                        continue

                    embedding_file_path = embedding_folder_path + embedding_filename

                    embedding_file_path_list.append(embedding_file_path)

                if len(embedding_file_path_list) == 0:
                    continue

                embedding_file_path_list.sort()

                self.paths_list.append([
                    mash_file_path, embedding_file_path_list
                ])

        self.paths_list.sort(key=lambda x: x[0])

        print(self.paths_list)
        print(len(self.paths_list))
        exit()

        self.transformer = getTransformer('Objaverse_82K')
        assert self.transformer is not None
        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, embedding_file_path_list = self.paths_list[index]

        mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cpu')

        mash_params = self.normalize(mash_params)

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        embedding_file_idx = np.random.choice(len(embedding_file_path_list))

        embedding_file_path = embedding_file_path_list[embedding_file_idx]

        embedding = np.load(embedding_file_path, allow_pickle=True).item()[self.embedding_key]

        embedding = torch.from_numpy(embedding).float()

        data = {
            'mash_params': mash_params,
            'embedding': embedding,
        }

        return data
