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



### 环境安装

```bash
pip install -r requirements.txt
```

### 下载基础模型

```bash
# 下载 模型
python modeldownloead.py
```

### 实验结果

```bash
1.configs/Lora-Attention.json完整参数

```

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
    "learning_rate":  5e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/lora_weights/attention_only",
    "trained_save_root_path": "./logs",
    "lora_r": 8,
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Please strictly output the results in the following JSON format: {{\"entities\": [{{\"name\": \"entity name\", \"type\": \"entity type\"}}]}}. You must output only a valid JSON string with no additional content. Only use the predefined entity type \"GENE\"; do not recognize or include any other entity types. Only extract entities that clearly belong to the GENE category as defined, and do not include any reasoning or explanations—just the JSON output.\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 0.5,
    "max_new_tokens":  256,
    "train_mode":"lora",
    "lora_alpha": 16,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps":8,
    "warmup_steps": 200,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
}

```

### 训练

```bash
python train.py --mode train --config_path ./configs/Lora-Attention.json
```


### 验证评估

```bash
python train.py --mode eval --config_path ./configs/Lora-Attention.json
```
###使用了LoRA，注入 "q_proj","k_proj","v_proj","o_proj"
## 实验结果

<img width="1593" height="303" alt="image" src="https://github.com/user-attachments/assets/df95fec3-96fc-4636-bd4e-5435786a1e63" />

<img width="1542" height="622" alt="image" src="https://github.com/user-attachments/assets/e574031f-fc83-4504-83b9-019264cdb7d6" />

### Dev 最佳（Epoch 3）

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8399** |
| **Recall** | **0.8261** |
| **F1** | **0.8329** |

### Test 结果

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8192** |
| **Recall** | **0.8179** |
| **F1** | **0.8155** |

显存：
<img width="982" height="476" alt="1879f6d73a69f0bd5ecb4c366713b6e3" src="https://github.com/user-attachments/assets/bce5028d-7281-4001-9675-5969024971d8" />

2.configs/Lora-Linear.json完整参数
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
    "learning_rate":  2e-5,
    "cache_dir": "./model",
    "max_length": 400,
    "output_dir": "/root/autodl-tmp/lora_weights/all_linear",
    "trained_save_root_path": "./logs",
    "lora_r": 16,
    "prompt": "Extract gene/protein entities. Output JSON:\n{\"entities\": [{\"name\": \"...\", \"type\": \"GENE\"}]}\nSentence: {sentence}\nOutput:",
    "max_grad_norm": 1,
    "max_new_tokens": 512,
    "train_mode":"lora",
    "lora_alpha": 16,
    "gradient_checkpointing": true,
    "gradient_accumulation_steps":1,
    "warmup_steps": 400,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
}

### 训练

```bash
python train.py --mode train --config_path ./configs/Lora-Linear.json
```


### 验证评估

```bash
python train.py --mode eval --config_path ./configs/Lora-Linear.json
```
使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
## 实验结果
<img width="1562" height="631" alt="image" src="https://github.com/user-attachments/assets/52093d74-968e-45e8-ae91-e68a4aeb9e46" />


### Dev 最佳（Epoch 3）

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8399** |
| **Recall** | **0.8261** |
| **F1** | **0.8329** |

### Test 结果

| 指标 | 值 |
|------|-----|
| **Precision** | **0.8192** |
| **Recall** | **0.8179** |
| **F1** | **0.8155** |

显存：
<img width="982" height="476" alt="1879f6d73a69f0bd5ecb4c366713b6e3" src="https://github.com/user-attachments/assets/bce5028d-7281-4001-9675-5969024971d8" />

## 数据集


数据格式：

```json
{
    "sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase",
    "entities": [
        {"name": "alkaline phosphatases", "type": "GENE", "pos": [16, 37]}
    ]
}
```


## 📈 评估指标

在实体级别进行精确匹配评估：

- **Precision**：预测正确的实体数 / 预测实体总数
- **Recall**：预测正确的实体数 / 真实实体总数
- **F1**：精确率和召回率的调和平均
- 支持按实体类型单独计算和 micro/macro 平均

### 实验结果
#### 使用了LoRA，注入 "q_proj","k_proj","v_proj","o_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8254 | 0.8203 | 0.8229 | 6323 |
| macro_avg | 0.8254 | 0.8203 | 0.8229 | - |
| micro_avg | 0.8254 | 0.8203 | 0.8229 | - |
**Qwen2.5-7B-Instruct**
| | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.821344 | 0.821604 | 0.821474 | 6323.0 |
| macro_avg | 0.821344 | 0.821604 | 0.821474 | NaN |
| micro_avg | 0.821344 | 0.821604 | 0.821474 | NaN |

<figure>
  <img src="img/lora2.png" alt="LoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj"，显存占用</figcaption>
</figure>

#### 使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8365 | 0.8331 | 0.8348 | 6323 |
| macro_avg | 0.8365 | 0.8331 | 0.8348 | - |
| micro_avg | 0.8365 | 0.8331 | 0.8348 | - |
**Qwen2.5-7B-Instruct**
|               | Precision | Recall | F1 | Support |
|---------------|-----------|---------|----------|---------|
| GENE | 0.837331 | 0.833623 | 0.835473 | 6323.0 |
| macro_avg | 0.837331 | 0.833623 | 0.835473 | NaN |
| micro_avg | 0.837331 | 0.833623 | 0.835473 | NaN |

<figure>
  <img src="img/lora1.png" alt="LoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了LoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"，显存占用</figcaption>
</figure>

#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj" 
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8251 | 0.8172 | 0.8211 | 6323 |
| macro_avg | 0.8251 | 0.8172 | 0.8211 | - |
| micro_avg | 0.8251 | 0.8172 | 0.8211 | - |
**Qwen2.5-7B-Instruct**
|  | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.825468 | 0.816068 | 0.820741 | 6323.0 |
| macro_avg | 0.825468 | 0.816068 | 0.820741 | NaN |
| micro_avg | 0.825468 | 0.816068 | 0.820741 | NaN |

<figure>
  <img src="img/qlora1.png" alt="QLoRA 训练结果">
  <figcaption>Qwen2.5-7B-Instruct使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj"，显存占用</figcaption>
</figure>

#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" r=8，α=32
**Qwen2.5-7B**
|  | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.8333 | 0.8238 | 0.8285 | 6323 |
| macro_avg | 0.8333 | 0.8238 | 0.8285 | - |
| micro_avg | 0.8333 | 0.8238 | 0.8285 | - |
**Qwen2.5-7B-Instruct**
|  | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| GENE | 0.832987 | 0.824292 | 0.828617 | 6323.0 |
| macro_avg | 0.832987 | 0.824292 | 0.828617 | NaN |
| micro_avg | 0.832987 | 0.824292 | 0.828617 | NaN |


<figure>
  <img src="img/qlora2.png" alt="alt text">
  <figcaption>Qwen2.5-7B-Instruct使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"，显存占用</figcaption>
</figure>
