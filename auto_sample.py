import os
import time

from conditional_flow_matching.Demo.sampler import demo as demo_sample

if __name__ == "__main__":
    model_file_path = "./output/20241025_00:13:48/model_last.pth"

    last_time = 0.0

    current_epoch = 1

    while True:
        new_time = os.path.getmtime(model_file_path)

        if new_time == last_time:
            time.sleep(30)
            continue

        last_time = new_time

        time.sleep(30)

        save_folder_path = './output/sample/epoch' + str(current_epoch) + '/'

        demo_sample(save_folder_path)

        print('finish sample for epoch', current_epoch)

        current_epoch += 1
