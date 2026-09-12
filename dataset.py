import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loguru import logger

#参考Firefly框架
class MyDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            r_data = json.load(f)
        processed = []
        for data in  r_data:
            sentence = data["sentence"]
            ents = data["entities"]
            entities = [{"name": e["name"], "type": e["type"]} for e in ents]
            completion_text = json.dumps({"entities": entities}, ensure_ascii=False)
            prompt_text = f"Extract entities from the following sentences:\n{sentence}\n"
            processed.append({"prompt": prompt_text, "completion": completion_text})

        logger.info("There are {} data in dataset".format(len(processed)))
        self.data_list = processed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        prompt = data["prompt"]
        completion = data["completion"]
        #数据格式不需要拼接多轮对话
        # 分开编码，避免在后续 Loss 时出错
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        comp_ids.append(self.tokenizer.eos_token_id)
        all_ids = prompt_ids + comp_ids
        num_prompt = len(prompt_ids)
        # 如果总长度超过最大限制，触发截断机制，证明只有截断PROMPT，此时对应批次单个样本设置成-100了，因为没有目标输出
        # 如果只是截掉部分的COMPLETION，则未被截掉的COMPLETION，可以参与LOSS
        all_ids = all_ids[:self.max_seq_length]
        if num_prompt >= len(all_ids):
            num_prompt = len(all_ids)
        labels = all_ids.copy()
        labels[:num_prompt] = [-100] * num_prompt
        attention_mask = [1] * len(all_ids)


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
