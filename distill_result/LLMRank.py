
import os
import json
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from scipy.stats import kendalltau, spearmanr

class LLMRank:
    def __init__(self, root_path, task_n, llm_name, style, llm_dict, api_key):
        self.root_path = root_path
        self.cur_path = root_path + "/distill_result"
        self.task_n = task_n
        self.llm_name = llm_name
        self.style = style
        self.api_key = api_key
        self.llm_dict = llm_dict

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
            raise Exception("config does not exist")

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
                    raise Exception("Defect2Styletext:LLM return empty message")
                return ss.content
            except Exception as e:
                print(e)
                print(prefix)
                print(que)
                time.sleep(3)
                if i >= 4:
                    raise Exception("out of range, end request")

    def _select_representative_sentence(self, sentences, model):

        if len(sentences) == 1:
            return sentences[0]

        embeddings = model.encode(
            sentences,
            normalize_embeddings=True
        )

        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        similarities = embeddings @ centroid
        best_idx = int(np.argmax(similarities))

        return sentences[best_idx]

    def _get_ref_src(self):
        test_ref_path = f"{self.task_n}_test_ref.json"
        test_src_path = f"{self.task_n}_test_src_level-0.json"
        train_ref_path = f"{self.task_n}_train_ref.json"
        train_src_path = f"{self.task_n}_train_src_level-0.json"
        test_ref = json.load(open(self.root_path + "/defect_mining/" + test_ref_path, "r", encoding="utf-8"))
        test_src = json.load(open(self.root_path + "/defect_mining/" + test_src_path, "r", encoding="utf-8"))
        train_ref = json.load(open(self.root_path + "/defect_mining/" + train_ref_path, "r", encoding="utf-8"))
        train_src = json.load(open(self.root_path + "/defect_mining/" + train_src_path, "r", encoding="utf-8"))

        return test_ref, test_src, train_ref, train_src

    def _get_leveled_record(self, record_path, ret_dict=None):
        if ret_dict is None:
            ret_dict = {}
        low_path = self.root_path + "/classifier/record_low"
        up_path = self.root_path + "/classifier/record_up"
        assert record_path in [low_path, up_path]
        flag = "low" if record_path == low_path else "up"
        for file_name in os.listdir(record_path):
            if f"{self.task_n}_test" not in file_name:
                continue
            if self.llm_dict[self.llm_name] not in file_name:
                continue
            level_n = ""
            for i in range(10):
                if f"level-{i}" in file_name:
                    level_n = f"level-{i}"
                    break
            assert level_n != ""

            file = pickle.load(open(record_path + "/" + file_name, "rb"))
            for ele in file:
                sent = ele[2]
                src_ind = ele[3]
                if src_ind not in ret_dict.keys():
                    ret_dict[src_ind] = {level_n + flag: [sent]}
                else:
                    if level_n + flag not in ret_dict[src_ind].keys():
                        ret_dict[src_ind][level_n + flag] = [sent]
                    else:
                        ret_dict[src_ind][level_n + flag].append(sent)
        return ret_dict

    def _prepare_llmrank_record(self, model, data_dict):
        ret_dict = {}
        for src_ind in data_dict:
            src_record = data_dict[src_ind]
            leveled_record = {"0": src_record["src"][0]}
            leveled_num = 1
            for i in range(10):
                if f"level-{i}low" in src_record:
                    leveled_record[str(leveled_num)] = self._select_representative_sentence(src_record[f"level-{i}low"], model)
                    leveled_num += 1
                if f"level-{i}up" in src_record:
                    leveled_record[str(leveled_num)] = self._select_representative_sentence(src_record[f"level-{i}up"], model)
                    leveled_num += 1
                if f"level-{i+1}low" not in src_record and f"level-{i+1}up" not in src_record:
                    break
            ret_dict[src_ind] = leveled_record

        return ret_dict

    def _prepare_ori_rank_data(self):
        low_path = self.root_path + "/classifier/record_low"
        up_path = self.root_path + "/classifier/record_up"
        ori_rank_data_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_ori_rank.json"
        if os.path.exists(ori_rank_data_path):
            record_dict = json.load(open(ori_rank_data_path, "r", encoding="utf-8"))
        else:
            record_dict = self._get_leveled_record(low_path)
            record_dict = self._get_leveled_record(up_path, record_dict)
            test_ref, test_src, _, _ = self._get_ref_src()
            pass
            for src_ind in test_src:
                src = test_src[src_ind]
                ref = test_ref[src_ind]
                record_dict[src_ind]["ref"] = ref
                record_dict[src_ind]["src"] = src

            json.dump(record_dict, open(ori_rank_data_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return record_dict

    def _prepare_pre_rank_data(self, record_dict):
        pre_rank_data_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_pre_rank.json"
        if not os.path.exists(pre_rank_data_path):
            model = SentenceTransformer(r"F:\python_code\pre_model\all-mpnet-base-v2")
            pre_rank_data = self._prepare_llmrank_record(model, record_dict)
            json.dump(pre_rank_data, open(pre_rank_data_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        else:
            pre_rank_data = json.load(open(pre_rank_data_path, "r", encoding="utf-8"))

        return pre_rank_data

    def _org_src_rank_data(self, text_dict):
        c_list = "ABCDEFGHIJKLMNOP"
        prompt_lines = ""
        ret_dict = {}
        for i in range(10):
            if str(i) in text_dict.keys():
                prompt_lines += f"{c_list[i]}: {text_dict[str(i)]}\n"
                ret_dict[c_list[i]] = text_dict[str(i)]
            elif str(i + 1) in text_dict.keys():
                raise Exception("ranking error")
            else:
                break
        return prompt_lines, ret_dict

    def request_llm_rank(self):
        rank_save_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_rank_result.json"
        if os.path.exists(rank_save_path):
            return json.load(open(rank_save_path, "r", encoding="utf-8"))

        record_dict = self._prepare_ori_rank_data()

        pre_rank_data = self._prepare_pre_rank_data(record_dict)

        prefix_temp = open(self.cur_path + "/prefix", "r", encoding="utf-8").read()
        prompt_temp = open(self.cur_path + "/prompt", "r", encoding="utf-8").read()

        llm_rank_result = {}
        for ind, src_ind in enumerate(pre_rank_data.keys()):
            if ind % 10 == 0:
                print(f"{ind/len(pre_rank_data.keys())*100:.2f}%")
            src_dict = pre_rank_data[src_ind]
            text_lines, text_dict = self._org_src_rank_data(src_dict)
            prefix = prefix_temp.format(**{"STYLE": self.style})
            prompt = prompt_temp.format(**{"STYLE": self.style, "src_texts": text_lines})

            rank_result = self._llm_flash(prefix, prompt, self.api_key)
            text_dict[self.llm_dict[self.llm_name]] = rank_result
            llm_rank_result[src_ind] = text_dict

        json.dump(llm_rank_result, open(rank_save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return llm_rank_result

    def correlation(self):
        c_list = [ele for ele in "ABCDEFGHIJKLMNOP"]
        file_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_rank_result.json"
        file = json.load(open(file_path, "r", encoding="utf-8"))
        kendall = []
        spearman = []
        for key_ind in file.keys():
            rank_result = file[key_ind][self.llm_dict[self.llm_name]].split(",")
            rank_result = [c_list.index(ele.strip()) for ele in rank_result]

            stand_result = [i for i in range(len(rank_result))]

            # Kendall Tau
            kendall_score, _ = kendalltau(rank_result, stand_result)
            # Spearman Rho
            spearman_score, _ = spearmanr(rank_result, stand_result)


            kendall.append(kendall_score)
            spearman.append(spearman_score)
        ken_mean = sum(kendall) / len(kendall)
        ken_var = np.var(kendall)
        spe_mean = sum(spearman) / len(spearman)
        spe_var = np.var(spearman)
        print(f"Kendall-mean:{ken_mean}, Kendall-var:{ken_var}")
        print(f"Spearman-mean:{spe_mean}, Spearman-var:{spe_var}")

    def rouge(self):
        c_list = [ele for ele in "ABCDEFGHIJKLMNOP"]
        file_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_rank_result.json"
        file = json.load(open(file_path, "r", encoding="utf-8"))
        rouge_1 = {}
        rouge_2 = {}
        rouge_n = {}
        for key_ind in file.keys():
            src_sent = file[key_ind]["A"]
            for lev in c_list[1:]:
                if lev not in file[key_ind].keys():
                    break
                lev_ind = c_list.index(lev)
                target_sent = file[key_ind][lev]

                scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )

                scores = scorer.score(src_sent, target_sent)

                for name_ind, rouge_dict in enumerate([rouge_1, rouge_2, rouge_n]):
                    rouge_name = ["rouge1", "rouge2", "rougeL"][name_ind]
                    if lev_ind not in rouge_dict.keys():
                        rouge_dict[lev_ind] = [scores[rouge_name].fmeasure]
                    else:
                        rouge_dict[lev_ind].append(scores[rouge_name].fmeasure)

        for rou_ind, rouge_dict in enumerate([rouge_1, rouge_2, rouge_n]):
            print(["rouge1", "rouge2", "rougeL"][rou_ind])
            for lev_num in rouge_dict.keys():
                rouge_mean = np.mean(rouge_dict[lev_num])
                rouge_var = np.var(rouge_dict[lev_num])
                print(f"level:{lev_num}, num_per:{len(rouge_dict[lev_num]) / len(rouge_dict[1])},"
                      f" mean:{round(rouge_mean, 4)}, var:{round(rouge_var, 4)}")

        pass

    def bert_score(self):
        model = SentenceTransformer(r"F:\python_code\pre_model\all-MiniLM-L6-v2")
        scorer = BERTScorer(
            model_type=r"F:\python_code\pre_model\deberta-v3-base",
            num_layers=12,
            lang="en",
            rescale_with_baseline=False
        )
        c_list = [ele for ele in "ABCDEFGHIJKLMNOP"]
        file_path = self.cur_path + f"/test_result/{self.task_n}_{self.llm_dict[self.llm_name]}_rank_result.json"
        file = json.load(open(file_path, "r", encoding="utf-8"))
        bert_score = {}
        s_bert = {}
        for key_ind in file.keys():
            src_sent = file[key_ind]["A"]
            for lev in c_list[1:]:
                if lev not in file[key_ind].keys():
                    break
                lev_ind = c_list.index(lev)
                target_sent = file[key_ind][lev]

                P, R, bert_score_result = scorer.score([target_sent],[src_sent])
                bert_score_result = bert_score_result.item()

                embeddings = model.encode(
                    [src_sent, target_sent],
                    normalize_embeddings=True
                )

                s_bert_result = cosine_similarity(
                    [embeddings[0]],
                    [embeddings[1]]
                )[0][0]

                if lev_ind not in bert_score.keys():
                    bert_score[lev_ind] = [bert_score_result]
                else:
                    bert_score[lev_ind].append(bert_score_result)

                if lev_ind not in s_bert.keys():
                    s_bert[lev_ind] = [s_bert_result]
                else:
                    s_bert[lev_ind].append(s_bert_result)


        print("BERTScore")
        for lev_num in bert_score.keys():
            rouge_mean = np.mean(bert_score[lev_num])
            rouge_var = np.var(bert_score[lev_num])
            print(
                f"berts: level:{lev_num}, num_per:{len(bert_score[lev_num]) / len(bert_score[1])},"
                f" mean:{round(rouge_mean, 4)}, var:{round(rouge_var, 4)}")
        print("SBERT score")
        for lev_num in s_bert.keys():
            rouge_mean = np.mean(s_bert[lev_num])
            rouge_var = np.var(s_bert[lev_num])
            print(
                f"sbert: level:{lev_num}, num_per:{len(s_bert[lev_num]) / len(s_bert[1])},"
                f" mean:{rouge_mean:.4f}, var:{rouge_var:.4f}")

        pass

if __name__ == "__main__":
    root_path = r"F:\python_code\StyleDistill"
    task_n, llm_name = "task6", "nemotron3-ultra"

    llm_dict = {"deepseek-v4-flash": "dsv4f", "qwen3.5-flash": "qw35f", "gemini2.5-flash": "gm25f",
                "llama3.3-70b": "la3370", "nemotron3-ultra": "ne3u"}
    api_keys_dict = {"deepseek-v4-flash": "sk-600",
                     "qwen3.5-flash": "sk-f5de",
                     "gemini2.5-flash": "sk-or-v1-e05b73",
                     "llama3.3-70b": "sk-or-v1-e05b731",
                     "nemotron3-ultra": "sk-or-v1-e05b"}
    api_keys = api_keys_dict[llm_name]
    target_style = ["formal English style", "positive English style",
                    "negative English style", "formal English style", "informal English style",
                    "neutral English style", "toxic English style"][int(task_n[-1]) - 1]
    llr = LLMRank(root_path, task_n, llm_name, target_style, llm_dict, api_keys)
    llr.request_llm_rank()
    llr.correlation()
    llr.rouge()
    llr.bert_score()

