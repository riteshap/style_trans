
import json
import os


def ori2round1(root_path, data_file_path):
    cur_path = root_path + "/defect_mining"
    data_file = json.load(open(data_file_path, "r", encoding="utf-8"))
    file_name = data_file_path.split("/")[-1].split(".")[0]

    ref_dict = {}
    src_dict = {}
    ref_dict_path = cur_path + f"/{file_name}_ref.json"
    src_dict_path = cur_path + f"/{file_name}_src_level-0.json"
    if os.path.exists(src_dict_path) and os.path.exists(ref_dict_path):
        print(f"ori2round1, {ref_dict_path}，{src_dict_path}")
        return None

    for ind, ele in enumerate(data_file):
        src = ele["src"].replace("\n", "").strip()
        refs = [ee.replace("\n", "").strip() for ee in ele["ref"]]
        src_dict[ind] = [src]
        ref_dict[ind] = refs
    json.dump(ref_dict, open(ref_dict_path, "w", encoding="utf-8"), indent=2)
    json.dump(src_dict, open(src_dict_path, "w", encoding="utf-8"), indent=2)

    return None

def ori2round1_v2(root_path):
    # task2_test: 100条neg  -> task2_test_src, task3_test_ref
    # task2_train: 500条neg  -> task2_train_src, task3_train_ref
    # task3_test: 100条pos  -> task3_test_src, task2_test_ref
    # task3_train: 500条pos  -> task3_train_src, task2_train_ref

    cur_path = root_path + "/defect_mining"

    t2_test_src = "task2_test_src_level-0.json"
    t2_test_ref = "task2_test_ref.json"
    t2_train_src = "task2_train_src_level-0.json"
    t2_train_ref = "task2_train_ref.json"

    t3_test_src = "task3_test_src_level-0.json"
    t3_test_ref = "task3_test_ref.json"
    t3_train_src = "task3_train_src_level-0.json"
    t3_train_ref = "task3_train_ref.json"

    file_name_list = [t2_test_src, t2_test_ref, t2_train_src, t2_train_ref,
                      t3_test_src, t3_test_ref, t3_train_src, t3_train_ref]
    exist_num = 0
    for file_name in file_name_list:
        if os.path.exists(file_name):
            exist_num += 1
    if exist_num == 8:
        return None

    save_dict = {"task2_test": [t2_test_src, t3_test_ref], "task2_train": [t2_train_src, t3_train_ref],
                 "task3_test": [t3_test_src, t2_test_ref], "task3_train": [t3_train_src, t2_train_ref]}

    for data_file_name in save_dict.keys():
        exist_num = 0
        for save_name in save_dict[data_file_name]:
            if os.path.exists(root_path + "/defect_mining/" + save_name):
                exist_num += 1
        if exist_num == len(save_dict[data_file_name]):
            continue

        data_file = json.load(open(root_path + f"/data/{data_file_name}.json", "r", encoding="utf-8"))
        ret_dict = {}
        for ind, ele in enumerate(data_file):
            ret_dict[ind] = [ele["src"].replace("\n", "").strip()]
        for save_name in save_dict[data_file_name]:
            json.dump(ret_dict, open(root_path + "/defect_mining/" + save_name, "w", encoding="utf-8"), indent=2)

    return None


if __name__ == "__main__":
    pass