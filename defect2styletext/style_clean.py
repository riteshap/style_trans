
import os
import json


class StyleCleaner:
    def __init__(self, root_path, llm_name, llm_dict):
        self.root_path = root_path
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.cur_path = self.root_path + "./defect2styletext"
        pass

    def formal_check(self, task_n):
        for file_name in os.listdir(self.cur_path):
            if "rewrite" not in file_name:
                continue
            if task_n not in file_name:
                continue
            if self.llm_dict[self.llm_name] not in file_name:
                continue

            print(file_name)

            file = json.load(open(self.cur_path + "/" + file_name, "r", encoding="utf-8"))

            err_num = 0
            for ele in file:
                out_data = ele[self.llm_name]
                try:
                    ss = json.loads(out_data)
                except Exception as e:
                    err_num += 1
                    print(file_name)
                    print(ele["sentence"])
                    print(out_data)
                    print(e)
                    # break
            if err_num > 0:
                raise Exception(f"error have {err_num}")

            for ele in file:
                try:
                    out_data = ele[self.llm_name]
                    ss = json.loads(out_data)["rewritten_text"]
                    assert type(ss) == str
                except Exception as e:
                    print(ele)
                    raise Exception(f"error have")

    def final_check(self, task_n):
        for file_name in os.listdir(self.cur_path):
            if "rewrite" not in file_name:
                continue
            if task_n not in file_name:
                continue

            if "final" not in file_name:
                continue



            print(file_name)
            file = json.load(open(self.cur_path + "/" + file_name, "r", encoding="utf-8"))
            err_num = 0
            for ele in file:
                out_data = ele[self.llm_name]
                try:
                    ss = json.loads(out_data)
                    rewritten_text = ss["rewritten_text"]
                except Exception as e:
                    err_num += 1
                    print(file_name)
                    print(ele["sentence"])
                    print(out_data)
                    print(e)

            if err_num > 0:
                raise Exception(f"error have {err_num}")



if __name__ == "__main__":
    pass



