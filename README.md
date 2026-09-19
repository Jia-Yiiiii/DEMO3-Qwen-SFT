# QwenNER — 基于 Qwen 大模型的命名实体识别
本项目基于Qwen2.5-7B 模型，实现对bc2gm数据集中的实体抽取任务，理解指令微调的过程与原理
## 项目简介

- **基础模型**：[Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)

- **实现模块**：
  - 使用 PyTorch 和 Hugging Face 生态，独立完成以下模块：
    - 数据预处理，参考 [Firefly 开源框架](https://github.com/YeungNLP/firefly-train-1.1M)
    - 模型设计与训练流程，参考 [Hugging Face Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) 的源码
    - 模型评估（手写）与结果可视化（[SwanLab]）

## 项目结构

```
DEMO3-Qwen-SFT/
├── configs/
│   ├── Lora-Linear.json
│   └── Qlora-Linear.json
├── data/
│   ├── dev.json
│   ├── labels.json
│   ├── test.json
│   └── train.json
├── README.md
├── config.py
├── dataset.py
├── modeldownloead.py
├── trian.py
└── utils.py
```

## 环境安装

```bash
pip install requirements.txt
```

## 下载基础模型

```bash
python modeldownloead.py
```

---

## 实验结果

### 1. configs/Lora-Linear.json

**完整参数**：

```json
{
   {
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 2,
    "patience": 3,
    "device": "cuda:0",
    "dropout": 0.1,
    "weight_decay": 0.01,
    "epochs": 4,
    "learning_rate": 2e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/lora_weights/all_linear",
    "trained_save_root_path": "./logs",
    "lora_r": 16,
    "prompt": "Extract gene/protein entities. Output JSON:\n{\"entities\": [{\"name\": \"...\", \"type\": \"GENE\"}]}\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 1,
    "max_new_tokens": 512,
    "train_mode": "lora",
    "lora_alpha": 16,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 400,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
}
```

**训练**：

```bash
python trian.py --mode train --config_path ./configs/Lora-Linear.json
```

**注入模块**：LoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**实验结果**：

<img width="1606" height="316" alt="image" src="https://github.com/user-attachments/assets/d555c13c-e9e8-412e-ba45-468c9cc0f81c" />
<img width="1591" height="663" alt="image" src="https://github.com/user-attachments/assets/9b3a1a9c-ff31-47b0-b84c-ce5a6989519a" />

| 实验 | Precision (查准率) | Recall (召回率) | F1-Score (F1分数) |
| :--- | :---: | :---: | :---: |
| Dev (LoRA) | 0.8303 | 0.8369 | 0.8386 |
| Test（LoRA） | 0.8189 | 0.8215 | 0.8202 |


**训练时显存**：
<img width="943" height="515" alt="ac26a535fb728531fc665bbfddcb265b" src="https://github.com/user-attachments/assets/026b0cd1-002b-4185-8a29-8cdf28360852" />


---
---

### 2. configs/Qlora-Linear.json

**完整参数**：

```json
{
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 4,
    "patience": 5,
    "device": "cuda:0",
    "weight_decay": 0.01,
    "epochs": 5,
    "learning_rate": 1e-4,
    "cache_dir": "./model",
    "max_length": 512,
    "output_dir": "/root/autodl-tmp/qlora_weights/all_linear",
    "trained_save_root_path": "./logs",
    "lora_r": 16,
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Output strictly as JSON: {\"entities\": [{\"name\": \"entity name\", \"type\": \"entity type\"}]}. No extra text.\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 1,
    "max_new_tokens": 128,
    "train_mode": "qlora",
    "lora_alpha": 32,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

**运行指令**：

```bash
python trian.py --mode train --config_path ./configs/Qlora-Linear.json
```

**注入模块**：QLoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**实验结果**：
<img width="1578" height="643" alt="image" src="https://github.com/user-attachments/assets/11d73f9d-d1ee-4752-8b77-67ccb773122f" />
<img width="1587" height="327" alt="image" src="https://github.com/user-attachments/assets/58ec58ce-edc6-41f6-a4cd-e0d698f6b959" />



| 实验 | Precision (查准率) | Recall (召回率) | F1-Score (F1分数) |
| :--- | :---: | :---: | :---: |
| Dev (QLoRA) | 0.8440 | 0.8192 | 0.8314 |
| Test（QLoRA） | 0.8346 | 0.7950 | 0.8143 |



**训练时显存**：
<img width="952" height="491" alt="c7b70ca93d0898375799c2def15e3af0" src="https://github.com/user-attachments/assets/5ff730d6-0eb4-4f2f-8980-c7a68fd74a35" />


