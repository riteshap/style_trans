import json
import copy
import time
import os
import itertools
import numpy as np
import pickle
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class Defect2Styletext:
    def __init__(self, root_path, target_style, llm_name, llm_dict, task_n, level_n, api_keys, sbert, top_k):
        self.root_path = root_path
        self.cur_path = self.root_path + "/defect2styletext"
        self.target_style = target_style
        self.llm_name = llm_name
        self.task_n, self.level_n = task_n, level_n
        self.top_k = top_k

        self.api_keys = api_keys
        self.llm_dict = llm_dict
        self.create_folder()
        self.model = SentenceTransformer(sbert)

    def create_folder(self):
        os.makedirs(self.cur_path + "/cache/")

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
            # model_name = "meta-llama/llama-3.3-70b-instruct:free"
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
                    raise Exception("Defect2Styletext:LLM empt")
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
        prefix_temp = open(self.cur_path + "/prefix").read()
        prompt_temp = open(self.cur_path + "/prompt").read()
        prompt_p2_temp = open(self.cur_path + "/prompt_p2").read()
        ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
        prompt_temp = prompt_temp.replace("{REF_N}", ss)

        file_llm_name = self.llm_dict[self.llm_name]

        train_ref = json.load(
            open(self.root_path + f"/defect_mining/{self.task_n}_train_ref.json", "r", encoding="utf-8"))

        test_src = self.root_path + f"/defect_mining/{self.task_n}_test_src_{self.level_n}_{file_llm_name}_weakness.json"
        train_src = self.root_path + f"/defect_mining/{self.task_n}_train_src_{self.level_n}_{file_llm_name}_weakness.json"

        def catch_find(catch_obj, prefix, prompt, llm_name):
            for ele in catch_obj:
                if ele["prefix"] == prefix and ele["prompt"] == prompt:
                    return ele[llm_name]
            return None

        def get_llm_output(src_file_path):
            src_file = json.load(open(src_file_path, "r", encoding="utf-8"))
            ret_out_list = []
            test_train_flag = "test" if "test" in src_file_path else "train"
            catch_path = self.cur_path + f"/cache/cache_{self.task_n}_{test_train_flag}_{self.level_n}_{file_llm_name}_rewrite.json"
            if os.path.exists(catch_path):
                cache_ret_out_list = json.load(open(catch_path, "r", encoding="utf-8"))
            else:
                cache_ret_out_list = []
            for ind, ele in enumerate(src_file):
                if int(ind) % 10 == 0:
                    print(f"{round(100 * int(ind) / len(src_file), 2)}%")

                ref_list = []
                for tr_ind in train_ref.keys():
                    if "test" not in src_file_path and tr_ind == ele["src_ind"]:
                        continue
                    ref_list.extend(train_ref[tr_ind])

                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                weaknesses = json.loads(ele[self.llm_name])["weaknesses"]
                if len(weaknesses) == 0:
                    fake_llm_out = json.dumps({"rewritten_text": sentence}, ensure_ascii=False)
                    ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: fake_llm_out})
                    continue

                sim5_few_shot = self.select_sim_k(sentence, ref_list)

                for a_weak in weaknesses:
                    preserved_weakness_list = "\n".join([a for a in weaknesses if a != a_weak])
                    prompt_dict = {"TARGET_STYLE": self.target_style,
                                   "SOURCE_TEXT": sentence,
                                   "target_weakness": a_weak,
                                   "preserved_weakness_list": preserved_weakness_list}
                    for i in range(self.top_k):
                        prompt_dict[f"REF_{i + 1}"] = sim5_few_shot[i]

                    prefix = prefix_temp
                    prompt_p2 = prompt_p2_temp.replace("{target_weakness}", a_weak)
                    prompt = prompt_temp.format(**prompt_dict) + prompt_p2

                    llm_output = catch_find(cache_ret_out_list, prefix, prompt, self.llm_name)
                    if llm_output is not None:
                        ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                    else:
                        llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                        ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                        cache_ret_out_list.append(
                            {"prefix": prefix, "prompt": prompt, "src_ind": src_ind, "sentence": sentence,
                             self.llm_name: llm_output})
                    if len(cache_ret_out_list) % 10 == 0:
                        json.dump(cache_ret_out_list, open(catch_path, "w", encoding="utf-8"))

            return ret_out_list

        test_src_save_name = test_src.split("/")[-1].split(".")[0].replace("weakness", "rewrite")
        test_src_save_path = self.cur_path + "/{}.json".format(test_src_save_name)
        if not os.path.exists(test_src_save_path):
            test_out_list = get_llm_output(test_src)
            json.dump(test_out_list, open(test_src_save_path, "w", encoding="utf-8"), indent=2)
        else:
            test_out_list = json.load(open(test_src_save_path, "r", encoding="utf-8"))
            print(f"Defect2Styletext.request_llm,{test_src_save_path}")
        print(f"Defect2Styletext.request_llm, {test_src_save_path}, type:dict, len:{len(test_out_list)}")

        train_src_save_name = train_src.split("/")[-1].split(".")[0].replace("weakness", "rewrite")
        train_src_save_path = self.cur_path + "/{}.json".format(train_src_save_name)
        if not os.path.exists(train_src_save_path):
            train_out_list = get_llm_output(train_src)
            json.dump(train_out_list, open(train_src_save_path, "w", encoding="utf-8"), indent=2)
        else:
            train_out_list = json.load(open(train_src_save_path, "r", encoding="utf-8"))
            print(f"Defect2Styletext.request_llm,{train_src_save_path}")
        print(f"Defect2Styletext.request_llm, {train_src_save_path}, type:dict, len:{len(train_out_list)}")


    def add_request(self, r_n: int):
        assert r_n > 1
        mis_test_keys = pickle.load(open(self.cur_path + f"/{self.task_n}_{self.level_n}_mis_test_keys.pkl", "rb"))
        mis_train_keys = pickle.load(open(self.cur_path + f"/{self.task_n}_{self.level_n}_mis_train_keys.pkl", "rb"))

        file_llm_name = self.llm_dict[self.llm_name]

        train_ref = json.load(open(self.root_path + f"/defect_mining/{self.task_n}_train_ref.json", "r", encoding="utf-8"))

        test_src = self.root_path + f"/defect_mining/{self.task_n}_test_src_{self.level_n}_{file_llm_name}_weakness.json"
        train_src = self.root_path + f"/defect_mining/{self.task_n}_train_src_{self.level_n}_{file_llm_name}_weakness.json"

        def catch_find(catch_obj, prefix, prompt, llm_name):
            for ele in catch_obj:
                if ele["prefix"] == prefix and ele["prompt"] == prompt:
                    return ele[llm_name]
            return None

        def get_llm_output(src_file_path):
            src_file = json.load(open(src_file_path, "r", encoding="utf-8"))

            ret_out_list = []
            test_train_flag = "test" if "test" in src_file_path else "train"
            catch_path = self.cur_path + f"/cache/cache_{self.task_n}_{test_train_flag}_{self.level_n}_{file_llm_name}_rewrite_r-{r_n}.json"
            if os.path.exists(catch_path):
                cache_ret_out_list = json.load(open(catch_path, "r", encoding="utf-8"))
            else:
                cache_ret_out_list = []
            for ind, ele in enumerate(src_file):
                if int(ind) % 10 == 0:
                    print(f"{round(100 * int(ind) / len(src_file), 2)}%")

                if "test" in src_file_path and ele["src_ind"] not in mis_test_keys:
                    continue
                if "train" in src_file_path and ele["src_ind"] not in mis_train_keys:
                    continue

                ref_list = []
                for tr_ind in train_ref.keys():
                    if "test" not in src_file_path and tr_ind == ele["src_ind"]:
                        continue
                    ref_list.extend(train_ref[tr_ind])

                src_ind = ele["src_ind"]
                sentence = ele["sentence"]
                weaknesses = json.loads(ele[self.llm_name])["weaknesses"]

                sim5_few_shot = self.select_sim_k(sentence, ref_list)

                if r_n > len(weaknesses):
                    continue
                if r_n == len(weaknesses):
                    prefix_temp = open(self.cur_path + "/prefix_v3").read()
                    prompt_temp = open(self.cur_path + "/prompt_v3").read()
                    ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
                    prompt_temp = prompt_temp.replace("{REF_N}", ss)
                    prompt_p2_temp = open(self.cur_path + "/prompt_p2_v3").read()
                    prompt_dict = {"TARGET_STYLE": self.target_style,
                                   "SOURCE_TEXT": sentence,
                                   "target_weakness": "\n".join(weaknesses)}

                    for i in range(self.top_k):
                        prompt_dict[f"REF_{i + 1}"] = sim5_few_shot[i]

                    prefix = prefix_temp
                    prompt = prompt_temp.format(**prompt_dict) + prompt_p2_temp
                    llm_output = catch_find(cache_ret_out_list, prefix, prompt, self.llm_name)
                    if llm_output is not None: # 存在catch
                        ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                    else:
                        llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                        ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                        cache_ret_out_list.append(
                            {"prefix": prefix, "prompt": prompt, "src_ind": src_ind, "sentence": sentence,
                             self.llm_name: llm_output})
                    if len(cache_ret_out_list) % 10 == 0:
                        json.dump(cache_ret_out_list, open(catch_path, "w", encoding="utf-8"))

                else:
                    prefix_temp = open(self.cur_path + "/prefix_v2").read()
                    prompt_temp = open(self.cur_path + "/prompt").read()
                    prompt_p2_temp = open(self.cur_path + "/prompt_p2_v2").read()
                    ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
                    prompt_temp = prompt_temp.replace("{REF_N}", ss)

                    for weak_points_list in itertools.combinations(weaknesses, r_n):
                        preserved_weakness_list = "\n".join([a for a in weaknesses if a not in weak_points_list])
                        weak_points_list = "\n".join(weak_points_list)
                        prompt_dict = {"TARGET_STYLE": self.target_style,
                                       "SOURCE_TEXT": sentence,
                                       "target_weakness": weak_points_list,
                                       "preserved_weakness_list": preserved_weakness_list}

                        for i in range(self.top_k):
                            prompt_dict[f"REF_{i + 1}"] = sim5_few_shot[i]

                        prefix = prefix_temp
                        prompt = prompt_temp.format(**prompt_dict) + prompt_p2_temp
                        llm_output = catch_find(cache_ret_out_list, prefix, prompt, self.llm_name)
                        if llm_output is not None:  # 存在catch
                            ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                        else:
                            llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                            ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})
                            cache_ret_out_list.append({"prefix":prefix, "prompt":prompt, "src_ind": src_ind, "sentence": sentence,
                                                       self.llm_name: llm_output})
                        if len(cache_ret_out_list) % 10 == 0:
                            json.dump(cache_ret_out_list, open(catch_path, "w", encoding="utf-8"))

            return ret_out_list

        test_src_save_name = test_src.split("/")[-1].split(".")[0].replace("weakness", f"rewrite_r-{r_n}")
        test_src_save_path = self.cur_path + "/{}.json".format(test_src_save_name)
        if not os.path.exists(test_src_save_path):
            test_out_list = get_llm_output(test_src)
            json.dump(test_out_list, open(test_src_save_path, "w", encoding="utf-8"), indent=2)
        else:
            test_out_list = json.load(open(test_src_save_path, "r", encoding="utf-8"))
            print(f"Defect2Styletext.add_request,{test_src_save_path}")
        print(f"Defect2Styletext.add_request, {test_src_save_path}, type:dict, len:{len(test_out_list)}")

        train_src_save_name = train_src.split("/")[-1].split(".")[0].replace("weakness", f"rewrite_r-{r_n}")
        train_src_save_path = self.cur_path + "/{}.json".format(train_src_save_name)
        if not os.path.exists(train_src_save_path):
            train_out_list = get_llm_output(train_src)
            json.dump(train_out_list, open(train_src_save_path, "w", encoding="utf-8"), indent=2)
        else:
            train_out_list = json.load(open(train_src_save_path, "r", encoding="utf-8"))
            print(f"Defect2Styletext.add_request,{train_src_save_path}")
        print(f"Defect2Styletext.add_request, {train_src_save_path}, type:dict, len:{len(train_out_list)}")


    def final_request_llm(self):
        prefix_temp = open(self.cur_path + "/prefix_final").read()
        prompt_temp = open(self.cur_path + "/prompt_final").read()
        prompt_p2_temp = open(self.cur_path + "/prompt_p2_final").read()
        ss = "\n".join(["n. {REF_n}".replace("n", str(i + 1)) for i in range(self.top_k)])
        prompt_temp = prompt_temp.replace("{REF_N}", ss)

        file_llm_name = self.llm_dict[self.llm_name]

        train_ref = json.load(open(self.root_path + f"/defect_mining/{self.task_n}_train_ref.json", "r", encoding="utf-8"))

        test_src = self.root_path + f"/defect_mining/{self.task_n}_test_src_level-0.json"

        def get_llm_output(src_file_path):
            src_file = json.load(open(src_file_path, "r", encoding="utf-8"))
            ret_out_list = []
            for ind, src_key in enumerate(src_file.keys()):
                if int(ind) % 10 == 0:
                    print(f"{round(100 * int(ind) / len(src_file), 2)}%")

                ref_list = []
                for tr_ind in train_ref.keys():
                    ref_list.extend(train_ref[tr_ind])

                src_ind = src_key
                sentence = src_file[src_key][0]

                sim5_few_shot = self.select_sim_k(sentence, ref_list)

                prompt_dict = {"TARGET_STYLE": self.target_style,
                               "SOURCE_TEXT": sentence}

                for i in range(self.top_k):
                    prompt_dict[f"REF_{i+1}"] = sim5_few_shot[i]

                prefix = prefix_temp
                prompt_p2 = prompt_p2_temp
                prompt = prompt_temp.format(**prompt_dict) + prompt_p2

                llm_output = self._llm_flash(prefix, prompt, self.api_keys)
                ret_out_list.append({"src_ind": src_ind, "sentence": sentence, self.llm_name: llm_output})

            return ret_out_list

        test_src_save_name = test_src.split("/")[-1].split(".")[0].replace("level-0", f"level-final_{file_llm_name}")
        test_src_save_path = self.cur_path + "/{}.json".format(test_src_save_name)
        if not os.path.exists(test_src_save_path):
            test_out_list = get_llm_output(test_src)
            json.dump(test_out_list, open(test_src_save_path, "w", encoding="utf-8"), indent=2)
        else:
            test_out_list = json.load(open(test_src_save_path, "r", encoding="utf-8"))
            print(f"Defect2Styletext.final_request_llm,{test_src_save_path}")
        print(f"Defect2Styletext.final_request_llm, {test_src_save_path}, type:dict, len:{len(test_out_list)}")


if __name__ == "__main__":
    pass

