import os
import json
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

class Turn2Emb:
    def __init__(self, root_path, llm_name, llm_dict, emb_model_path):
        self.root_path = root_path
        self.llm_name = llm_name
        self.llm_dict = llm_dict
        self.create_folder()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = emb_model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        pass
    
    def create_folder(self):
        folder_list = ["pkl", "record_low", "record_up", "temp_pkl"]
        for folder in folder_list:
            path = self.root_path + "/classifier/" + folder
            if not os.path.exists(path):
                os.makedirs(path)

    def encode_text_list_with_prefix(
            self,
            tokenizer,
            model,
            texts: list[str],
            src_ind: int,
            max_length: int = 32,
            prefix_token_id: int = 128000,
    ):

        device = next(model.parameters()).device

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        result_list = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length - 1,
                add_special_tokens=False,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            batch_size = input_ids.size(0)

            prefix_ids = torch.full(
                (batch_size, 1),
                prefix_token_id,
                dtype=input_ids.dtype,
            )

            prefix_mask = torch.ones(
                (batch_size, 1),
                dtype=attention_mask.dtype,
            )

            input_ids = torch.cat([prefix_ids, input_ids], dim=1).to(device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

            last_hidden = outputs.last_hidden_state.detach().cpu()
            attention_mask_cpu = attention_mask.detach().cpu()

            for i in range(batch_size):
                result_list.append((last_hidden[i], attention_mask_cpu[i], batch_texts[i], src_ind))

        return result_list

    def src_ref_turn2emb(self, task_n:str, level_n:str):
        file_llm_name = "_" + self.llm_dict[self.llm_name]
        file_llm_name = "" if level_n == "level-0" else file_llm_name

        test_ref = f"{task_n}_test_ref.json"
        test_src = f"{task_n}_test_src_{level_n}{file_llm_name}.json"
        train_ref = f"{task_n}_train_ref.json"
        train_src = f"{task_n}_train_src_{level_n}{file_llm_name}.json"

        for file_name in [test_ref, test_src, train_ref, train_src]:
            print(file_name)

            src_pickle_emb = os.path.join(self.root_path + "/classifier", "pkl/{}_emb.pkl".format(file_name.split(".")[0]))
            if os.path.exists(src_pickle_emb):
                obj = pickle.load(open(src_pickle_emb, "rb"))
                print(f"Turn2Emb.src_ref_turn2emb, {src_pickle_emb}已存在")
                print(f"Turn2Emb.src_ref_turn2emb, save info:{src_pickle_emb}, type:{type(obj)}, len:{len(obj)}")
                continue
            file_path = os.path.join(self.root_path + "/defect_mining", file_name)

            file = json.load(open(file_path, "r", encoding="utf-8"))

            emb_list = []
            for ind in file.keys():
                ind_list = self.encode_text_list_with_prefix(self.tokenizer, self.model, file[ind], ind)
                emb_list.extend(ind_list)

            print(f"Turn2Emb.src_ref_turn2emb, save info:{src_pickle_emb}, type:{type(emb_list)}, len:{len(emb_list)}")
            pickle.dump(emb_list, open(src_pickle_emb, "wb"))

    def style_text_turn2emb(self, task_n:str, level_n:str, r_n:int=0):

        file_llm_name = self.llm_dict[self.llm_name]
        file_r_n = "" if r_n < 2 else f"_r-{r_n}"

        test_src = f"{task_n}_test_src_{level_n}_{file_llm_name}_rewrite{file_r_n}.json"
        train_src = f"{task_n}_train_src_{level_n}_{file_llm_name}_rewrite{file_r_n}.json"

        for file_name in [test_src, train_src]:
            print(file_name)
            src_pickle_emb = os.path.join(self.root_path + "/classifier", "temp_pkl/{}_emb.pkl".format(file_name.split(".")[0]))

            if os.path.exists(src_pickle_emb):
                obj = pickle.load(open(src_pickle_emb, "rb"))
                print(f"Turn2Emb.style_text_turn2emb, {src_pickle_emb}已存在")
                print(f"Turn2Emb.style_text_turn2emb, save info:{src_pickle_emb}, type:{type(obj)}, len:{len(obj)}")
                continue

            file_path = os.path.join(self.root_path + "/defect2styletext", file_name)

            file = json.load(open(file_path, "r", encoding="utf-8"))

            emb_list = []
            for ele in file:
                llm_out = json.loads(ele[self.llm_name])
                rewritten_text = [llm_out["rewritten_text"]]
                ind = ele["src_ind"]

                ind_list = self.encode_text_list_with_prefix(self.tokenizer, self.model, rewritten_text, ind)
                emb_list.extend(ind_list)

            print(f"Turn2Emb.style_text_turn2emb, save info:{src_pickle_emb}, type:{type(emb_list)}, len:{len(emb_list)}")
            pickle.dump(emb_list, open(src_pickle_emb, "wb"))

class StyleBinaryDataset(Dataset):

    def __init__(
            self,
            pos_vec_list: list[tuple[torch.Tensor, torch.Tensor, str, int]],
            neg_vec_list: list[tuple[torch.Tensor, torch.Tensor, str, int]],
            hidden_dtype: torch.dtype = torch.float32,
    ):
        self.samples = []
        self.hidden_dtype = hidden_dtype

        for value in pos_vec_list:
            hidden_states, attention_mask, src_text, sample_id = value
            self.samples.append({
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "label": 1,
                "sample_id": sample_id,
                "text": src_text,
                "source": "pos",
            })

        for value in neg_vec_list:
            hidden_states, attention_mask, src_text, sample_id = value
            self.samples.append({
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "label": 0,
                "sample_id": sample_id,
                "text": src_text,
                "source": "neg",
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        hidden_states = item["hidden_states"]
        attention_mask = item["attention_mask"]

        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.tensor(hidden_states)

        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)

        return {
            "hidden_states": hidden_states.to(self.hidden_dtype),      # [seq_len, hidden_size]
            "attention_mask": attention_mask.long(),                   # [seq_len]
            "label": torch.tensor(item["label"], dtype=torch.long),    # []
            "sample_id": item["sample_id"],
            "text": item["text"],
            "source": item["source"],
        }

class StyleBinaryClassifier(nn.Module):

    def __init__(
        self,
        hidden_size: int = 3072,
        max_len: int = 32,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        labels: Optional[torch.Tensor] = None,
    ):
        device = next(self.parameters()).device

        padding_mask = attention_mask == 0

        encoded = self.transformer(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )

        cls_vec = encoded[:, 0, :]

        logits = self.classifier(cls_vec)

        if labels is not None:
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            return {
                "loss": loss,
                "logits": logits,
                "probs": torch.softmax(logits, dim=-1),
            }

        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
        }

class TrainTest:
    def __init__(self, root_path, llm_b_name):
        self.root_path = root_path
        self.llm_b_name = llm_b_name
        self.cur_path = self.root_path + "/classifier"

    def train(self, train_dict, test_dict, task_n:str, level_n:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_vec_dict, neg_vec_dict = test_dict
        test_dataset = StyleBinaryDataset(pos_vec_dict, neg_vec_dict)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        pos_vec_dict, neg_vec_dict = train_dict
        dataset = StyleBinaryDataset(pos_vec_dict, neg_vec_dict)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = StyleBinaryClassifier(hidden_size=3072, max_len=32, num_heads=8).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

        num_epochs = 20
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs["loss"]
                logits = outputs["logits"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.size(0)

            avg_loss = total_loss / len(loader)
            acc = total_correct / total_count

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"loss={avg_loss:.4f} | acc={acc:.4f}"
            )

            model.eval()
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            for batch in test_loader:
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.size(0)

            avg_loss = total_loss / len(test_loader)
            acc = total_correct / total_count

            print(
                f"loss={avg_loss:.4f} | acc={acc:.4f}"
            )
            if acc >= best_acc:
                #
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), self.cur_path + f"/{task_n}_{level_n}_{self.llm_b_name}_best_classifier.pth")
        print(f"best_acc{best_acc:.4f} | best_epoch：{best_epoch}")

        return model, best_epoch, best_acc

    def evaluate(self, test_dict, task_n:str, level_n:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_vec_dict, neg_vec_dict = test_dict
        test_dataset = StyleBinaryDataset(pos_vec_dict, neg_vec_dict)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        model = StyleBinaryClassifier(hidden_size=3072, max_len=32, num_heads=8).to(device)
        model.load_state_dict(torch.load(self.cur_path + "/{}_{}_{}_best_classifier.pth".format(task_n, level_n, self.llm_b_name)))
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0
        with torch.no_grad():
            for batch in test_loader:
                hidden_states = batch["hidden_states"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_count += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        acc = total_correct / total_count

        print(
            f"evaluate: loss={avg_loss:.4f} | acc={acc:.4f}"
        )

        return acc

    def train_main(self, task_n:str, level_n:str):

        file_llm_name = "_" + self.llm_b_name
        file_llm_name = "" if level_n == "level-0" else file_llm_name

        if os.path.exists(self.cur_path + f"/{task_n}_{level_n}_{self.llm_b_name}_best_classifier.pth"):
            print(f"TrainTest.train_main: {task_n}_{level_n}_{self.llm_b_name}_best_classifier.pth 已存在")
            test_ref = self.cur_path + f"/pkl/{task_n}_test_ref_emb.pkl"
            test_src = self.cur_path + f"/pkl/{task_n}_test_src_{level_n}{file_llm_name}_emb.pkl"
            pos_vec_dict = pickle.load(open(test_ref, "rb"))
            neg_vec_dict = pickle.load(open(test_src, "rb"))
            test_set = [pos_vec_dict, neg_vec_dict]
            best_acc = self.evaluate(test_set, task_n, level_n)
            return best_acc

        test_ref = self.cur_path + f"/pkl/{task_n}_test_ref_emb.pkl"
        test_src = self.cur_path + f"/pkl/{task_n}_test_src_{level_n}{file_llm_name}_emb.pkl"
        train_ref = self.cur_path + f"/pkl/{task_n}_train_ref_emb.pkl"
        train_src = self.cur_path + f"/pkl/{task_n}_train_src_{level_n}{file_llm_name}_emb.pkl"

        pos_vec_dict = pickle.load(open(train_ref, "rb"))
        neg_vec_dict = pickle.load(open(train_src, "rb"))
        train_set = [pos_vec_dict, neg_vec_dict]

        pos_vec_dict = pickle.load(open(test_ref, "rb"))
        neg_vec_dict = pickle.load(open(test_src, "rb"))
        test_set = [pos_vec_dict, neg_vec_dict]

        trained_model, best_epoch, best_acc = self.train(train_set, test_set, task_n, level_n)

        print(f"best_epoch={best_epoch}")

        return best_acc

class StyleTextClassifier:

    def __init__(self, root_path, task_n, level_n, llm_b_name):
        self.root_path = root_path
        self.cur_path = self.root_path + "/classifier"
        self.llm_b_name = llm_b_name
        self.task_n = task_n
        self.level_n = level_n

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StyleBinaryClassifier(hidden_size=3072, max_len=32, num_heads=8).to(self.device)
        self.model.load_state_dict(torch.load(self.cur_path + "/{}_{}_{}_best_classifier.pth".format(task_n, level_n, self.llm_b_name)))
        self.model.eval()

    def _separate(self, pkl_list:list):
        low_level_text_list = []
        up_level_text_list = []
        with torch.no_grad():

            for value in pkl_list:
                hidden_states, attention_mask, text, src_ind = value
                hidden_states = hidden_states.unsqueeze(0).to(torch.float32).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).long().to(self.device)
                outputs = self.model(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1).to("cpu").numpy()[0]  # pos: 1, neg: 0
                if preds == 1:
                    up_level_text_list.append(value)
                else:
                    low_level_text_list.append(value)

        return low_level_text_list, up_level_text_list

    def round_sep(self, r_n:int=0):
        file_r_n = "" if r_n < 2 else f"_r-{r_n}"

        test_name = f"{self.task_n}_test_{self.level_n}_{self.llm_b_name}{file_r_n}.pkl"
        train_name = f"{self.task_n}_train_{self.level_n}_{self.llm_b_name}{file_r_n}.pkl"
        exists_record = []
        for name_tt in [test_name, train_name]:
            if os.path.exists(os.path.join(self.cur_path + "/record_low", name_tt)):
                exists_record.append(1)
            if os.path.exists(os.path.join(self.cur_path + "/record_up", name_tt)):
                exists_record.append(1)
        if sum(exists_record) != 0 and sum(exists_record) !=4:
            raise Exception("record_up,record_low error")
        elif sum(exists_record) == 4:
            return None

        test_low_list, test_up_list = [], []
        train_low_list, train_up_list = [], []

        for file_name in os.listdir(self.cur_path + "/temp_pkl"):
            sp_name_part = f"{self.level_n}_{self.llm_b_name}_rewrite{file_r_n}"
            if sp_name_part not in file_name:
                continue
            if self.task_n not in file_name:
                continue

            vec_dict = pickle.load(open(os.path.join(self.cur_path + "/temp_pkl", file_name), "rb"))

            low_list, up_list = self._separate(vec_dict)

            if "test" in file_name:
                test_low_list.extend(low_list)
                test_up_list.extend(up_list)
            else:
                train_low_list.extend(low_list)
                train_up_list.extend(up_list)

        low_list = [test_low_list, train_low_list]
        for ind, name_tt in enumerate([test_name, train_name]):
            saved_list = low_list[ind]
            pickle.dump(saved_list, open(os.path.join(self.cur_path + "/record_low", name_tt), "wb"))
            print(f"record_low:{name_tt}:len:{len(saved_list)}")

        up_list = [test_up_list, train_up_list]
        for ind, name_tt in enumerate([test_name, train_name]):
            saved_list = up_list[ind]
            pickle.dump(saved_list, open(os.path.join(self.cur_path + "/record_up", name_tt), "wb"))
            print(f"record_up:{name_tt}:len:{len(saved_list)}")

        return None

    def check_mis_ind(self):
        ori_test_data = json.load(open(self.root_path + f"/defect_mining/{self.task_n}_test_src_level-0.json", "r", encoding="utf-8"))
        ori_train_data = json.load(open(self.root_path + f"/defect_mining/{self.task_n}_train_src_level-0.json", "r", encoding="utf-8"))
        test_keys = [a for a in ori_test_data.keys()]
        train_keys = [a for a in ori_train_data.keys()]

        test_name_key = f"{self.task_n}_test_{self.level_n}_{self.llm_b_name}"
        train_name_key = f"{self.task_n}_train_{self.level_n}_{self.llm_b_name}"
        # 合并
        test_pkl, train_pkl = [], []
        for file_name in os.listdir(self.cur_path + "/record_up"):
            if test_name_key in file_name:
                test_part_pkl = pickle.load(open(os.path.join(self.cur_path + "/record_up", file_name), "rb"))
                test_pkl.extend(test_part_pkl)
            if train_name_key in file_name:
                train_part_pkl = pickle.load(open(os.path.join(self.cur_path + "/record_up", file_name), "rb"))
                train_pkl.extend(train_part_pkl)

        curent_test_keys = [a[3] for a in test_pkl]
        curent_train_keys = [a[3] for a in train_pkl]
        mis_test_keys = [a for a in test_keys if a not in curent_test_keys]
        mis_train_keys = [a for a in train_keys if a not in curent_train_keys]
        percent_mis_test = len(mis_test_keys) / len(test_keys)
        percent_mis_train = len(mis_train_keys) / len(train_keys)
        print(f"mis_test_keys: {len(mis_test_keys)}")
        print(f"mis_train_keys: {len(mis_train_keys)}")

        pickle.dump(mis_test_keys, open(self.root_path + f"/defect2styletext/{self.task_n}_{self.level_n}_mis_test_keys.pkl", "wb"))
        pickle.dump(mis_train_keys, open(self.root_path + f"/defect2styletext/{self.task_n}_{self.level_n}_mis_train_keys.pkl", "wb"))

        return mis_test_keys, mis_train_keys, percent_mis_test, percent_mis_train

    def main(self):
        self.round_sep()
        self.check_mis_ind()


if __name__ == "__main__":
    pass


