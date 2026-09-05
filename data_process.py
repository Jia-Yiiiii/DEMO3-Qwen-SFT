import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

#参考Firefly框架
class MyDataset(Dataset):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = config["max_len"]
        self.data = self.read_data(config["data_path"])

#这里直接在读取数据时构造Prompt格式,以及COMPLETION格式
    @staticmethod
    def read_data(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        processed = []
        for item in all_data:
            sentence = item["sentence"]
            ents = item["entities"]
            entities = [{"entity_text": e["name"], "entity_type": e["type"]} for e in ents]
            completion_text = json.dumps({"entities": entities}, ensure_ascii=False)
            prompt_text = f"Extract entities from the following sentences:\n{sentence}\n"
            processed.append({"prompt": prompt_text, "completion": completion_text})
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        completion = sample["completion"]

        # 分开编码，避免在后续 Loss 时出错
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        comp_ids.append(self.tokenizer.eos_token_id)
        all_ids = prompt_ids + comp_ids
        num_prompt = len(prompt_ids)
        # 如果总长度超过最大限制，触发截断机制，证明只有截断PROMPT，此时对应批次单个样本设置成-100了，因为没有目标输出
        # 如果只是截掉部分的COMPLETION，则未被截掉的COMPLETION，可以参与LOSS
        all_ids = all_ids[:self.max_len]
        if num_prompt >= len(all_ids):
            num_prompt = len(all_ids)
        labels = all_ids.copy()
        labels[:num_prompt] = [-100] * num_prompt
        attention_mask = [1] * len(all_ids)

        # 验证截断后长度一致，参考Firefly
        assert len(all_ids) == len(labels) == len(attention_mask)

        return {
            "input_ids": all_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_ids,
        }

    def collate_fn(self, batch):
        input_ids_batch = [x["input_ids"] for x in batch]
        labels_batch = [x["labels"] for x in batch]
        prompt_ids_batch = [x["prompt_ids"] for x in batch]
        max_batch_len = max(len(seq) for seq in input_ids_batch)
        max_prompt_len = max(len(seq) for seq in prompt_ids_batch)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_prompt_ids = []
        batch_prompt_attention_mask = []

        for ids, lab, pids in zip(input_ids_batch, labels_batch, prompt_ids_batch):

            pad_len = max_batch_len - len(ids)
            batch_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
            batch_labels.append(lab + [-100] * pad_len)
            pad_prompt_len = max_prompt_len - len(pids)
            batch_prompt_ids.append(pids + [self.tokenizer.pad_token_id] * pad_prompt_len)
            batch_prompt_attention_mask.append([1] * len(pids) + [0] * pad_prompt_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "prompt_ids": torch.tensor(batch_prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(batch_prompt_attention_mask, dtype=torch.long),
        }


    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
