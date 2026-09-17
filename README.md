# QwenNER — 基于 Qwen 大模型的命名实体识别

## 项目简介

- **基础模型**：[Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)

- **实现模块**：
  - 使用 PyTorch 和 Hugging Face 生态，独立完成以下模块：
    - 数据预处理，参考 [Firefly 开源框架](https://github.com/YeungNLP/firefly-train-1.1M)
    - 模型设计与训练流程，参考 [Hugging Face Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py) 的源码
    - 模型评估（手写）与结果可视化（[SwanLab](https://swanlab.cn/)）

## 项目结构

```
DEMO3-Qwen-SFT/
├── configs/
│   ├── Lora-Attention.json
│   ├── Lora-Linear.json
│   ├── Qlora-Attention.json
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

### 1. configs/Lora-Attention.json

**完整参数**：

```json
{
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 2,
    "patience": 10,
    "device": "cuda:0",
    "dropout": 0.1,
    "weight_decay": 0.01,
    "epochs": 5,
    "learning_rate": 5e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/lora_weights/attention_only",
    "trained_save_root_path": "./logs",
    "lora_r": 8,
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Please strictly output the results in the following JSON format: {{\"entities\": [{{\"name\": \"entity name\", \"type\": \"entity type\"}}]}}. You must output only a valid JSON string with no additional content. Only use the predefined entity type \"GENE\"; do not recognize or include any other entity types. Only extract entities that clearly belong to the GENE category as defined, and do not include any reasoning or explanations—just the JSON output.\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 0.5,
    "max_new_tokens": 256,
    "train_mode": "lora",
    "lora_alpha": 16,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 200,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

**训练**：

```bash
python trian.py --mode train --config_path ./configs/Lora-Attention.json
```

**评估**：

```bash
python trian.py --mode eval --config_path ./configs/Lora-Attention.json
```

**注入模块**：LoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`

**实验结果**：

![image](https://github.com/user-attachments/assets/df95fec3-96fc-4636-bd4e-5435786a1e63)

![image](https://github.com/user-attachments/assets/e574031f-fc83-4504-83b9-019264cdb7d6)

#### Dev 最佳（Epoch 3）

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8399** |
| **Recall** | **0.8261** |
| **F1** | **0.8329** |

#### Test 结果

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8192** |
| **Recall** | **0.8179** |
| **F1** | **0.8155** |

**显存**：

![image](https://github.com/user-attachments/assets/bce5028d-7281-4001-9675-5969024971d8)

---

### 2. configs/Lora-Linear.json

**完整参数**：

```json
{
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 4,
    "patience": 10,
    "device": "cuda:0",
    "dropout": 0.1,
    "weight_decay": 0.01,
    "epochs": 5,
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
    "gradient_accumulation_steps": 1,
    "warmup_steps": 400,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

**训练**：

```bash
python trian.py --mode train --config_path ./configs/Lora-Linear.json
```

**评估**：

```bash
python trian.py --mode eval --config_path ./configs/Lora-Linear.json
```

**注入模块**：LoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**实验结果**：

![image](https://github.com/user-attachments/assets/5c321e0f-e1f3-4d80-bea4-7c1d86cfc799)

![image](https://github.com/user-attachments/assets/52093d74-968e-45e8-ae91-e68a4aeb9e46)

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

**显存**：

![image](https://github.com/user-attachments/assets/bcbb5940-1cdf-4f3e-bfe6-191efa34d44b)

---

### 3. configs/Qlora-Attention.json

**完整参数**：

```json
{
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 4,
    "patience": 10,
    "device": "cuda:0",
    "weight_decay": 0.01,
    "epochs": 5,
    "learning_rate": 2e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/qlora_weights/attention_only",
    "trained_save_root_path": "./logs",
    "lora_r": 32,
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Output strictly as JSON: {\"entities\": [{\"name\": \"entity name\", \"type\": \"entity type\"}]}. No extra text.\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 1,
    "max_new_tokens": 128,
    "train_mode": "qlora",
    "lora_alpha": 64,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 500,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

**训练**：

```bash
python trian.py --mode train --config_path ./configs/Qlora-Attention.json
```

**评估**：

```bash
python trian.py --mode eval --config_path ./configs/Qlora-Attention.json
```

**注入模块**：QLoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`

**实验结果**：

![image](https://github.com/user-attachments/assets/7ce8b152-4730-4fbb-94f5-cec5f9cab9fa)

![image](https://github.com/user-attachments/assets/f10bfabc-0802-4dc7-b5ef-fdb76d894dd1)

#### Dev 最佳（Epoch 2）

| 指标 | 值 |
|------|-----|
| **Dev Precision** | **0.8523** |
| **Dev Recall** | **0.7975** |
| **Dev F1** | **0.824** |

#### Test 结果

| 指标 | 值 |
|------|-----|
| **Test Precision** | **0.8288** |
| **Test Recall** | **0.7822** |
| **Test F1** | **0.8048** |

**显存**：

![image](https://github.com/user-attachments/assets/d19a30e7-e788-4558-b5cd-ba432bfe5b77)

---

### 4. configs/Qlora-Linear.json

**完整参数**：

```json
{
    "train_path": "./data/train.json",
    "dev_path": "./data/dev.json",
    "test_path": "./data/test.json",
    "model_name": "Qwen2.5-7B",
    "model_path": "./model/Qwen2.5-7B",
    "batch_size": 4,
    "patience": 10,
    "device": "cuda:0",
    "weight_decay": 0.01,
    "epochs": 5,
    "learning_rate": 2e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/qlora_weights/all_linear",
    "trained_save_root_path": "./logs",
    "lora_r": 32,
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Output strictly as JSON: {\"entities\": [{\"name\": \"entity name\", \"type\": \"entity type\"}]}. No extra text.\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 1,
    "max_new_tokens": 128,
    "train_mode": "qlora",
    "lora_alpha": 64,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 500,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

**训练**：

```bash
python trian.py --mode train --config_path ./configs/Qlora-Linear.json
```

**评估**：

```bash
python trian.py --mode eval --config_path ./configs/Qlora-Linear.json
```

**注入模块**：QLoRA，`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**实验结果**：

<img width="1600" height="302" alt="image" src="https://github.com/user-attachments/assets/212a52be-400b-4c27-bbc2-64b2bac2f264" />
<img width="1607" height="737" alt="image" src="https://github.com/user-attachments/assets/71c22495-cea1-4316-bca7-401576f6a137" />


### Dev 最佳（Step 5）

| 指标 | 值 |
|------|-----|
| **Dev Precision** | **0.8482** |
| **Dev Recall** | **0.8148** |
| **Dev F1** | **0.8312** |

### Test 结果

| 指标 | 值 |
|------|-----|
| **Test Precision** | **0.8391** |
| **Test Recall** | **0.7993** |
| **Test F1** | **0.8187** |

**显存**：

![image](https://github.com/user-attachments/assets/61ebb195-c43d-4e44-99d7-6432e000e4e4)
