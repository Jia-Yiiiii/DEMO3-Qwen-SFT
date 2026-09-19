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
pip install -r requirements.txt
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

#### Dev 最佳（Epoch 2）

| 指标 | 值 |
|------|-----|
| **Dev Precision** | **0.8387** |
| **Dev Recall** | **0.8441** |
| **Dev F1** | **0.8414** |

#### Test 结果

| 指标 | 值 |
|------|-----|
| **Test Precision** | **0.8203** |
| **Test Recall** | **0.8312** |
| **Test F1** | **0.8257** |

**训练时显存**：




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
    "epochs": 8,
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
<img width="1576" height="637" alt="image" src="https://github.com/user-attachments/assets/bfd5bf23-5899-4fc2-88e1-c94b3c021602" />
<img width="1583" height="292" alt="image" src="https://github.com/user-attachments/assets/5b42c358-43d9-4ce7-a778-465654c5cf4e" />


### | 测试指标 | Precision (查准率) | Recall (召回率) | F1-Score (F1分数) |
| :--- | :---: | :---: | :---: |
| Test | 0.8465 | 0.8055 | 0.8255 |
| Dev (Epoch 2) | 0.8487 | 0.8092 | 0.8285 |


**显存**：
<img width="777" height="315" alt="image" src="https://github.com/user-attachments/assets/7a594a28-909b-470b-8587-284cf02f12c2" />

