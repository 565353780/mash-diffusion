import os

from conditional_flow_matching.Module.Convertor.mash_split import Convertor


def demo():
    HOME = os.environ['HOME']
    dataset_root_folder_path = HOME + "/Dataset/"
    train_scale = 0.98
    val_scale = 0.01

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(train_scale, val_scale)

    return True
