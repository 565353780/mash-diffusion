import os
import time

from conditional_flow_matching.Demo.sampler import demo as demo_sample

def waitTime(wait_second: int) -> bool:
    for i in range(wait_second, 0, -1):
        print("\r waiting to continue running, time remains:", i, "s ...", end="    ")
        time.sleep(1)
    print()
    return True

if __name__ == "__main__":
    model_file_path = "./output/20241025_00:13:48/model_last.pth"
    start_epoch_idx = 201
    sample_now = False

    new_time = os.path.getmtime(model_file_path)
    if sample_now:
        last_time = 0.0
    else:
        last_time = new_time

    current_epoch = start_epoch_idx

    while True:
        new_time = os.path.getmtime(model_file_path)

        if new_time == last_time:
            print('waiting for updating model to epoch', current_epoch)
            waitTime(30)
            continue

        last_time = new_time

        print('waiting for saving model of epoch', current_epoch)
        waitTime(30)

        save_folder_path = './output/sample/epoch' + str(current_epoch) + '/'

        demo_sample(save_folder_path)

        print('finish sample for epoch', current_epoch)

        current_epoch += 1
