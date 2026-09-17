# QwenNER — 基于 Qwen 大模型的命名实体识别

使用 Qwen2.5-7B-Instruct和Qwen2.5-7B模型，通过 QLoRA / LoRA 在 BC2GM数据集上进行命名实体识别（NER）。

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



### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 下载基础模型

```bash
# 下载 模型
hf download Qwen/Qwen2.5-7B --local-dir ./Qwen2.5-7B
```

### 3. 配置训练参数

编辑 `args/arg1.json`，完整参数：

```json
{
    "num_epochs": 5,
    "batch_size": 4,
    "lr":2e-05,
    "weight_decay": 0.01,
    "device": "cuda:0",
    "model_name": "Qwen2.5-7B",
    "model_dir": "../autodl-tmp/model/Qwen2.5-7B",
    "dropout_rate": 0.2,
    "data_path": "./data/",
    "max_length":400 ,
    "max_new_tokens": 300,
    "patience":10,
    "monitor": "val_f1",
    "delta":0.0001,
    "save_dir": "../autodl-tmp/checkpoint/",
    "warmup_steps": 100,
    "eps":1e-8,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "method": "lora",
    "lora_target_modules": [ 
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"],
    "prompt": "You are an expert in biomedical named entity recognition. Your task is to identify gene and protein entities from the given English biomedical text. The entity type is defined as follows: GENE includes gene or protein names, such as gene products, enzymes, receptors, antibodies, cytokines, and similar molecules. Please strictly output the results in the following JSON format: {\"entities\": [{\"name\": \"entity name\", \"type\": \"entity type\"}]}. You must output only a valid JSON string with no additional content. Only use the predefined entity type \"GENE\"; do not recognize or include any other entity types. Only extract entities that clearly belong to the GENE category as defined, and do not include any reasoning or explanations—just the JSON output. The input sentence is provided below.",
    "template_name": "qwen"
}
```

### 4. 训练

```bash
python trainer.py --arg ./args/arg1.json
```


### 5. 推理

```bash
python predict.py \
    --arg ./args/arg1.json \
    --weight ./checkpoint/exp1/ \
    --text "Using the same approach we have shown that hFIRE binds the stimulatory proteins Sp1 and Sp3 in addition to CBF"
```

输出示例：

```json
[{"entities": [{"name": "hFIRE", "type": "GENE"}, {"name": "Sp1", "type": "GENE"}, {"name": "Sp3", "type": "GENE"}, {"name": "CBF", "type": "GENE"}]}]
```

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
