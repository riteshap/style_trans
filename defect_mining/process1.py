import json
import os
import time
import copy
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class DefectMining:
    def __init__(self, root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys, sbert, top_k: int = 5):
        self.root_path = root_path
        self.cur_path = self.root_path + "/defect_mining"
        self.target_style = target_style
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.task_n = task_n
        self.level_n = level_n
        self.top_k = top_k

        self.api_keys = api_keys
        self.prefix_temp = open(self.cur_path + "/prefix").read()
        self.prompt_temp = open(self.cur_path + "/prompt").read()
        self.prompt_p2_temp = open(self.cur_path + "/prompt_p2").read()
        ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
        self.prompt_temp = self.prompt_temp.replace("{REF_N}", ss)

        self.model = SentenceTransformer(sbert)

    def _llm_flash(self, prefix, que, api_key):
        if self.llm_name == "deepseek-v4-flash":
            base_url_name = "https://api.deepseek.com"
            model_name = "deepseek-v4-flash"
        elif self.llm_name == "qwen3.5-flash":
            base_url_name = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model_name = "qwen3.5-flash"
        elif self.llm_name == "gemini2.5-flash":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "google/gemini-2.5-flash"
        elif self.llm_name == "llama3.3-70b":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "meta-llama/llama-3.3-70b-instruct"
        elif self.llm_name == "nemotron3-ultra":
            base_url_name = "https://openrouter.ai/api/v1"
            model_name = "nvidia/nemotron-3-ultra-550b-a55b"
        else:
            raise Exception("no such setting")

        client = OpenAI(api_key=api_key, base_url=base_url_name)
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": "{}".format(que)},
                    ],
                    stream=False
                )
                ss = response.choices[0].message
                if len(ss.content) == 0:
                    raise Exception("Defect2Styletext:LLM empty")
                return ss.content
            except Exception as e:
                print(e)
                print(prefix)
                print(que)
                time.sleep(3)
                if i >= 4:
                    raise Exception("times out")


    def select_sim_k(self, src: str, ref_list: list[str], diversity_threshold: float = 0.80):
        src_embeddings = self.model.encode([src], convert_to_tensor=True)
        ref_embeddings = self.model.encode(ref_list, convert_to_tensor=True)

        similarities = self.model.similarity(src_embeddings, ref_embeddings).cpu().numpy()[0]
        sorted_indices = np.argsort(-similarities)

        selected_indices = []

        for idx in sorted_indices:
            if len(selected_indices) >= self.top_k:
                break

            current_emb = ref_embeddings[idx]
            is_diverse = True

            for selected_idx in selected_indices:
                sim = self.model.similarity(current_emb, ref_embeddings[selected_idx]).cpu().numpy()[0]
                if sim > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected_indices.append(idx)

        selected_sentence = [ref_list[idx] for idx in selected_indices]

        return selected_sentence


    def request_llm(self):

        file_llm_name = "_" + self.llm_dict[self.llm_name]
        file_llm_name = "" if self.level_n == "level-0" else file_llm_name

        train_ref = json.load(open(self.cur_path + f"/{self.task_n}_train_ref.json", "r", encoding="utf-8"))

        test_src = f"{self.task_n}_test_src_{self.level_n}{file_llm_name}.json"
        train_src = f"{self.task_n}_train_src_{self.level_n}{file_llm_name}.json"

        def get_llm_output(src_file_name):
            src_file = json.load(open(self.cur_path + "/" + src_file_name, "r", encoding="utf-8"))

            total_num = 0
            for a in src_file.keys():
                total_num += len(src_file[a])

            cur_num = 0
            ret_out_list = []
            for ind in src_file.keys():
                if cur_num % 10 == 0:
                    print(f"{round(100 * cur_num / total_num, 2)}%")

                ref_list = []
                for tr_ind in train_ref.keys():
                    if "test" not in src_file_name and tr_ind == ind:
                        continue
                    ref_list.extend(train_ref[tr_ind])

                for sent in src_file[ind]:
                    cur_num += 1

                    simn_few_shot = self.select_sim_k(sent, ref_list)

                    prefix_dict = {"TARGET_STYLE": self.target_style}
                    prompt_dict = {"TARGET_STYLE": self.target_style, "SOURCE_TEXT": sent}  # , "TARGET_STYLE": prefix_dict["TARGET_STYLE"]
                    for i in range(self.top_k):
                        prefix_dict[f"REF_{i+1}"] = simn_few_shot[i]
                        prompt_dict[f"REF_{i+1}"] = simn_few_shot[i]

                    prefix = self.prefix_temp.format(**prefix_dict)
                    prompt = self.prompt_temp.format(**prompt_dict) + self.prompt_p2_temp

                    llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                    ret_out_list.append({"src_ind": ind, "sentence": sent, self.llm_name: llm_output})

            return ret_out_list

        if file_llm_name == "":
            save_name = self.cur_path + "/{}{}_weakness.json".format(test_src.split(".")[0], "_" + self.llm_dict[self.llm_name])
        else:
            save_name = self.cur_path + "/{}_weakness.json".format(test_src.split(".")[0])
        print(save_name)
        if not os.path.exists(save_name):
            test_out_list = get_llm_output(test_src)
            json.dump(test_out_list, open(save_name, "w", encoding="utf-8"), indent=2)
        else:
            test_out_list = json.load(open(save_name, "r", encoding="utf-8"))
            print(f"DefectMining.request_llm,{save_name}")
            print(f"DefectMining.request_llm, {save_name}, type:dict, len:{len(test_out_list)}")

        if file_llm_name == "":
            save_name = self.cur_path + "/{}{}_weakness.json".format(train_src.split(".")[0], "_" + self.llm_dict[self.llm_name])
        else:
            save_name = self.cur_path + "/{}_weakness.json".format(train_src.split(".")[0])
        print(save_name)
        if not os.path.exists(save_name):
            train_out_list = get_llm_output(train_src)
            json.dump(train_out_list, open(save_name, "w", encoding="utf-8"), indent=2)
        else:
            train_out_list = json.load(open(save_name, "r", encoding="utf-8"))
            print(f"DefectMining.request_llm,{save_name}")
            print(f"DefectMining.request_llm, {save_name}, type:dict, len:{len(test_out_list)}")



if __name__ == "__main__":
    pass


