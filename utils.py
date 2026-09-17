import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import json


def load_model(config):
    if config.train_mode == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map=config.device,
        trust_remote_code=True
    )

    if config.train_mode == 'qlora':
        model = prepare_model_for_kbit_training(
            model,

        )

    peft_config = LoraConfig(target_modules=config.lora_target_modules, task_type=TaskType.CAUSAL_LM,
                             inference_mode=False, r=config.lora_r, lora_alpha=config.lora_alpha,
                             lora_dropout=config.lora_dropout)
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    footprint = model.get_memory_footprint()
    print(f"Memory footprint:")
    print(f"- Bytes: {footprint:,} B")
    print(f"- GB: {footprint / 1024 ** 3:.2f} GB")

    return model


def get_trained_model(config):
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16, device_map=config.device,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, config.output_dir, device_map=config.device)

    return model


def get_pred_entities(text):
    if text is None:
        return []
    if not isinstance(text, str):
        return []
    if text.strip() == "":
        return []

    text = text.strip()
    if text.startswith("entities"):
        text = "{ " + text + " }"
    try:
        data = json.loads(text)
    except Exception :
        return []
    entities = []
    if "entities" not in data:
        return []
    for e in data["entities"]:
        if not isinstance(e, dict):
            continue
        entity_text = e.get("name", "")
        entity_type = e.get("type", "")
        if entity_text and entity_type:
            entities.append((entity_text, entity_type))
    return entities


def get_label_entities(text):
    try:
        data = json.loads(text)
        entities = data.get("entities", [])
        return [(e.get("name", ""), e.get("type", "")) for e in entities]
    except:
        return []


def get_metrics(pred_entities_list, gold_entities_list):
    pred_set = set()
    for sentence_entities in pred_entities_list:
        for entity in sentence_entities:
            pred_set.add(entity)
    gold_set = set()
    for sentence_entities in gold_entities_list:
        for entity in sentence_entities:
            gold_set.add(entity)
    correct_count = 0
    for entity in pred_set:
        if entity in gold_set:
            correct_count += 1
    if len(pred_set) > 0:
        precision = correct_count / len(pred_set)
    else:
        precision = 0.0

    if len(gold_set) > 0:
        recall = correct_count / len(gold_set)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
