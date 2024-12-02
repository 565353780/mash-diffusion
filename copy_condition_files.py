import os
from tqdm import tqdm
from shutil import copyfile

from conditional_flow_matching.Method.path import createFileFolder

def copyConditionFiles(
        root_folder_path: str,
        target_root_folder_path: str,
        overwrite: bool = False
) -> bool:
    if not os.path.exists(root_folder_path):
        return True

    rel_file_path_list = []
    for root, _, files in os.walk(root_folder_path):
        for file in files:
            if 'condition' not in file:
                continue

            rel_folder_path = os.path.relpath(root, root_folder_path) + '/'

            rel_file_path = rel_folder_path + file
            rel_file_path_list.append(rel_file_path)

    print('[INFO][copy_condition_files::copyConditionFiles]')
    print('\t start copy condition files...')
    print('\t', root_folder_path)
    print('\t -->')
    print('\t', target_root_folder_path)
    for rel_file_path in tqdm(rel_file_path_list):
        target_file_path = target_root_folder_path + rel_file_path
        if not overwrite:
            if os.path.exists(target_file_path):
                continue

        createFileFolder(target_file_path)

        copyfile(root_folder_path + '/' + rel_file_path, target_file_path)

    return True

if __name__ == "__main__":
    timestamp = '20241202_18:30:59'
    root_folder_path = './output/sample/' + timestamp + '/'
    recon_root_folder_path = './output/recon_smooth/' + timestamp + '/'
    render_root_folder_path = './output/render_smooth/' + timestamp + '/'
    overwrite = False

    # copyConditionFiles(root_folder_path, recon_root_folder_path, overwrite)
    copyConditionFiles(root_folder_path, render_root_folder_path, overwrite)
