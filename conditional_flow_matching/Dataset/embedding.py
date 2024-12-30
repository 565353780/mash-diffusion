import os
import torch
import pickle
import random
import numpy as np
from typing import Union
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
        dataset_json_file_path: Union[str, None] = None,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.embedding_key = embedding_key
        self.split = split
        self.dataset_json_file_path = dataset_json_file_path

        self.mash_folder_path = (
            self.dataset_root_folder_path + "Objaverse_82K/manifold_mash/"
        )
        self.embedding_root_folder_path = (
            self.dataset_root_folder_path + embedding_folder_name + "/"
        )

        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.embedding_root_folder_path)

        self.transformer = getTransformer("Objaverse_82K")
        assert self.transformer is not None

        self.output_error = False

        self.invalid_embedding_file_path_list = []

        self.paths_list = []

        if dataset_json_file_path is not None:
            if os.path.exists(dataset_json_file_path):
                with open(dataset_json_file_path, "rb") as f:
                    paths_list = pickle.load(f)

                    for paths in paths_list:
                        self.paths_list.append(
                            [
                                self.mash_folder_path + paths[0],
                                [
                                    self.embedding_root_folder_path + path
                                    for path in paths[1]
                                ],
                            ]
                        )
                    return

        print("[INFO][EmbeddingDataset::__init__]")
        print("\t start load mash and embedding datasets...")
        for root, _, files in os.walk(self.embedding_root_folder_path):
            if len(files) == 0:
                continue

            rel_folder_path = os.path.relpath(root, self.embedding_root_folder_path)

            mash_file_path = self.mash_folder_path + rel_folder_path + ".npy"

            if not os.path.exists(mash_file_path):
                continue

            embedding_file_path_list = []
            for file in files:
                if not file.endswith(".npy") or file.endswith("_tmp.npy"):
                    continue

                embedding_file_path_list.append(root + "/" + file)

            if len(embedding_file_path_list) == 0:
                continue

            embedding_file_path_list.sort()

            self.paths_list.append([mash_file_path, embedding_file_path_list])

        self.paths_list.sort(key=lambda x: x[0])

        return

    def normalize(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.transform(mash_params, False)

    def normalizeInverse(self, mash_params: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(mash_params, False)

    def __len__(self):
        return len(self.paths_list) * 100

    def __getitem__(self, index):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        mash_file_path, embedding_file_path_list = self.paths_list[index]

        if not os.path.exists(mash_file_path):
            if self.output_error:
                print("[ERROR][EmbeddingDataset::__getitem__]")
                print("\t this npy file is not valid!")
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        embedding_file_idx = np.random.choice(len(embedding_file_path_list))

        embedding_file_path = embedding_file_path_list[embedding_file_idx]

        if embedding_file_path in self.invalid_embedding_file_path_list:
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        try:
            embedding = np.load(embedding_file_path, allow_pickle=True).item()[
                self.embedding_key
            ]
        except KeyboardInterrupt:
            print("[INFO][EmbeddingDataset::__getitem__]")
            print("\t stopped by the user (Ctrl+C).")
            exit()
        except Exception as e:
            if self.output_error:
                print("[ERROR][EmbeddingDataset::__getitem__]")
                print("\t this npy file is not valid!")
                print("\t embedding_file_path:", embedding_file_path)
                print("\t error info:", e)

            self.invalid_embedding_file_path_list.append(embedding_file_path)
            new_idx = random.randint(0, len(self.paths_list) - 1)
            return self.__getitem__(new_idx)

        embedding = torch.from_numpy(embedding).float()

        mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, "cpu")

        mash_params = self.normalize(mash_params)

        permute_idxs = np.random.permutation(mash_params.shape[0])

        mash_params = mash_params[permute_idxs]

        data = {
            "mash_params": mash_params,
            "embedding": embedding,
        }

        return data
