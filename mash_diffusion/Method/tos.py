import tos
from concurrent.futures import ThreadPoolExecutor, as_completed


def listdirTOS(client: tos.TosClientV2, bucket: str, prefix: str) -> list:
    file_name_list = []

    truncated = True
    continuation_token = ""

    print("[INFO][tos::listdirTOS]")
    print("\t start list files in:", bucket + "/" + prefix)
    while truncated:
        result = client.list_objects_type2(
            bucket,
            prefix=prefix,
            continuation_token=continuation_token,
        )
        for item in result.contents:
            file_name_list.append(item.key)

        print(len(file_name_list), "files found")
        truncated = result.is_truncated
        continuation_token = result.next_continuation_token
        # FIXME: for test only
        break

    return file_name_list


def isFileExist(client: tos.TosClientV2, bucket: str, file_key: str) -> bool:
    try:
        client.head_object(bucket=bucket, key=file_key)
        return True
    except tos.exceptions.TosServerError as e:
        if e.status_code == 404:
            return False
        else:
            print(f"[ERROR] {key} 异常：{e}")
            return False


def filterExistFiles(
    client: tos.TosClientV2,
    bucket: str,
    file_key_pairs: list,
    max_workers: int = 64,
) -> list:
    def check_exists(file_key_pair):
        file, key = file_key_pair
        try:
            client.head_object(bucket=bucket, key=key)
            return file, True
        except tos.exceptions.TosServerError as e:
            if e.status_code == 404:
                return file, False
            else:
                print(f"[ERROR] {key} 异常：{e}")
                return file, False

    existing_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(check_exists, file_key_pair): file_key_pair
            for file_key_pair in file_key_pairs
        }
        for future in as_completed(future_to_key):
            file, exists = future.result()
            if exists:
                existing_files.append(file)

    return existing_files
