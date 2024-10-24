import os
import numpy as np
from math import ceil
from typing import Tuple


class Convertor(object):
    def __init__(self, dataset_root_folder_path: str) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_folder_path = self.dataset_root_folder_path + "MashV4/"
        self.split_folder_path = self.dataset_root_folder_path + "SplitMashCFM/"

        assert os.path.exists(self.mash_folder_path)

        return

    def toCategoryModelIdList(self, dataset_name: str, category_name: str) -> list:
        mash_category_folder_path = (
            self.mash_folder_path + dataset_name + "/" + category_name + "/"
        )

        modelid_list = []

        print("[INFO][Convertor::toCategoryFilePathList]")
        print("\t start search npy files...")
        print("\t mash_category_folder_path:", mash_category_folder_path)
        file_name_list = os.listdir(mash_category_folder_path)
        for file_name in file_name_list:
            if file_name[-4:] != ".npy":
                continue

            modelid = file_name.split(".npy")[0]

            mash_file_path = mash_category_folder_path + modelid + ".npy"

            if not os.path.exists(mash_file_path):
                continue

            modelid_list.append(modelid)

        modelid_list.sort()
        return modelid_list

    def convertToCategorySplits(
        self,
        dataset_name: str,
        category_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> Tuple[list, list, list]:
        if not os.path.exists(self.dataset_root_folder_path):
            print("[ERROR][Convertor::convertToCategorySplits]")
            print("\t dataset root folder not exist!")
            print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
            return [], [], []

        modelid_list = self.toCategoryModelIdList(dataset_name, category_name)

        permut_modelid_list = np.random.permutation(modelid_list)

        modelid_num = len(modelid_list)

        if modelid_num < 3:
            print("[WARN][Convertor::convertToCategorySplits]")
            print("\t category shape num < 3!")
            print("\t modelid_num:", modelid_num)
            return [], [], []

        train_split_num = ceil(train_scale * modelid_num)
        val_split_num = ceil(val_scale * modelid_num)

        if modelid_num == 3:
            train_split_num = 1
            val_split_num = 1

        if train_split_num + val_split_num == modelid_num:
            train_split_num -= 1

        train_split = permut_modelid_list[:train_split_num]
        val_split = permut_modelid_list[
            train_split_num : train_split_num + val_split_num
        ]
        test_split = permut_modelid_list[train_split_num + val_split_num :]

        return train_split.tolist(), val_split.tolist(), test_split.tolist()

    def convertToCategorySplitFiles(
        self,
        dataset_name: str,
        category_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        train_split, val_split, test_split = self.convertToCategorySplits(
            dataset_name, category_name, train_scale, val_scale
        )

        if len(train_split) + len(val_split) + len(test_split) == 0:
            print("[ERROR][Convertor::convertToCategorySplitFiles]")
            print("\t convertToCategorySplits failed!")
            return False

        save_split_folder_path = (
            self.split_folder_path + dataset_name + "/" + category_name + "/"
        )

        os.makedirs(save_split_folder_path, exist_ok=True)

        with open(save_split_folder_path + "train.txt", "w") as f:
            for train_name in train_split:
                f.write(train_name + "\n")

        with open(save_split_folder_path + "val.txt", "w") as f:
            for val_name in val_split:
                f.write(val_name + "\n")

        with open(save_split_folder_path + "test.txt", "w") as f:
            for test_name in test_split:
                f.write(test_name + "\n")

        return True

    def convertToDatasetSplitFiles(
        self,
        dataset_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        categories = os.listdir(self.mash_folder_path + dataset_name + "/")

        for i, category in enumerate(categories):
            print("[INFO][Convertor::convertToDatasetSplitFiles]")
            print(
                "\t start convert sdf dataset: "
                + dataset_name
                + "["
                + category
                + "], "
                + str(i + 1)
                + "/"
                + str(len(categories))
                + "..."
            )

            self.convertToCategorySplitFiles(
                dataset_name, category, train_scale, val_scale
            )

        return True

    def convertToSplitFiles(
        self,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            self.convertToDatasetSplitFiles(dataset_name, train_scale, val_scale)

        return True
