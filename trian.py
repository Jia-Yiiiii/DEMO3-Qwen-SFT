import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
from torch.utils.data import DataLoader
from transformers import get_scheduler
from dataset import MyDataset
from config import load_config
from utils import load_model, get_trained_model
from tqdm import tqdm
import swanlab
from utils import get_label_entities, get_pred_entities, get_metrics
from accelerate import Accelerator
from torch.optim import AdamW
import os
import argparse
import bitsandbytes as bnb


class Trainer:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_dataset = MyDataset(
            file=config.train_path,
            tokenizer=self.tokenizer,
            max_seq_length=config.max_length,
            prompt_template=config.prompt,
            is_train=True,
        )
        self.dev_dataset = MyDataset(
            file=config.dev_path,
            tokenizer=self.tokenizer,
            max_seq_length=config.max_length,
            prompt_template=config.prompt,
            is_train=False,
        )
        self.test_dataset = MyDataset(
            file=config.test_path,
            tokenizer=self.tokenizer,
            max_seq_length=config.max_length,
            prompt_template=config.prompt,
            is_train=False,
        )

        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            batch_size=config.batch_size*4,
            shuffle=False,
            collate_fn=self.dev_dataset.collate_fn
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size ,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size*4,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )

        os.makedirs(config.output_dir, exist_ok=True)
        self.best_f1 = 0.0
        self.patience = getattr(config, 'patience')
        self.counts = 0
        if getattr(config, 'gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps)

    def create_optimizer(self, model=None):
        if self.config.train_mode == "qlora":
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,

            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        warmup_steps = self.config.warmup_steps  
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.scheduler

    def compute_loss(self, model, input_ids, attention_mask, labels):
        loss = model(input_ids, attention_mask, labels=labels,return_dict=False)[0]
        
        return loss

    def train(self):
        config = self.config
        device = self.device
        model = self.model

        swanlab.init(
            project="Qwen-NER-SFT",
            config=vars(config)
        )

        train_dataloader = self.train_dataloader
        dev_dataloader = self.dev_dataloader
        total_steps = len(self.train_dataloader) * self.config.epochs // self.config.gradient_accumulation_steps
        self.create_optimizer_and_scheduler(total_steps)

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            model.config.use_cache = False
            torch.cuda.empty_cache()
            progress_bar = tqdm(
                train_dataloader,
                disable=not self.accelerator.is_main_process,
                desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]",
            )
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            for step, batch in enumerate(progress_bar):
                model_inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }
                # 梯度累加
                per_loss = self.compute_loss(model, model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["labels"])
                del batch["input_ids"]
                del batch["attention_mask"]
                del batch["labels"]


                loss = per_loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(loss)
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    
                    self.accelerator.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                total_loss += per_loss.item()
                progress_bar.set_postfix({"loss": per_loss.item()})

                if (step + 1) % 50 == 0:
                    swanlab.log({
                        "train/loss_step": per_loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                    })

                del per_loss, loss
            
            if len(train_dataloader) % self.config.gradient_accumulation_steps != 0:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                self.accelerator.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_dataloader)

            swanlab.log({"train/loss_epoch": avg_loss})
            print(f"[Train] Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

            dev_loss, dev_f1, dev_precision, dev_recall = self.evaluate(
                epoch=epoch,
                model=model,
                dataLoader=dev_dataloader,
                is_test=False
            )

            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                self.counts = 0
                print(f"[Dev] New best F1: {self.best_f1:.4f}, saving model to {config.output_dir}")
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(config.output_dir)
                self.accelerator.wait_for_everyone()
            else:
                self.counts += 1
                if self.counts >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        print("Training Finished. Best F1:", self.best_f1)
        self.model = None
        torch.cuda.empty_cache()
        # Use the model with the best dev F1 (saved to config.output_dir) for testing
        print("Testing with the best dev model...")
        self.evaluate_test()
        swanlab.finish()

    def evaluate(self, epoch, model, dataLoader=None, is_test=False):
        config = self.config
        device = self.device

        if dataLoader is None:
            dataLoader = self.test_dataloader if is_test else self.dev_dataloader

        desc = "Test" if is_test else "Dev"

        model.eval()
        model.config.use_cache = False

        all_preds = []
        all_labels = []
        with torch.no_grad():
            gen_bar = tqdm(
                dataLoader,
                desc=f"Epoch {epoch + 1}/{config.epochs} [{desc} Gen]",
                position=0,
                leave=True
            )
            for batch in gen_bar:
                generated = model.generate(
                    input_ids=batch["prompt_ids"].to(device),
                    attention_mask=batch["prompt_attention_mask"].to(device),
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                input_len = batch["prompt_ids"].shape[1]
                generated = generated[:, input_len:]
                pred_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

                labels_ids = batch["labels"].clone()
                labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
                gold_texts = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
                del labels_ids
                
                del batch["prompt_ids"]
                del batch["prompt_attention_mask"]
                for pred, gold in zip(pred_texts, gold_texts):
                    all_preds.append(get_pred_entities(pred))
                    all_labels.append(get_label_entities(gold))

        metrics = get_metrics(all_preds, all_labels)
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        if not is_test:
            print(f"[{desc}] Epoch {epoch + 1} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")
        else:
            print(f"[{desc}] F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")

        log_dict = {
                f"{desc}/precision": precision,
                f"{desc}/recall": recall,
                f"{desc}/f1": f1,
            }

        swanlab.log(log_dict, step=epoch)

        return None, f1, precision, recall

    def evaluate_test(self, epoch=None):
        if epoch is None:
            epoch = self.config.epochs
        model = get_trained_model(self.config)
        model = model.to(self.device)
        _, _, _, _ = self.evaluate(
            epoch=epoch,
            model=model,
            dataLoader=self.test_dataloader,
            is_test=True
    )


    def predict_sentence(self, sentence):
        max_new_tokens = self.config.max_new_tokens
        model = get_trained_model(self.config)
        model = model.to(self.device)
        model.eval()
        model.config.use_cache = True
        prompt = self.config.prompt.replace("{sentence}", sentence)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = generated[:, input_ids.shape[1]:]
        pred_text = self.tokenizer.decode(
            gen_ids[0],
            skip_special_tokens=True
        )

        return pred_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEMO3")
    parser.add_argument(
        "--config_path",
        type=str,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "predict"],
        required=True,
    )
    args = parser.parse_args()
    config =load_config(args.config_path)
    model = load_model(config)
    trainer = Trainer(config, model)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        swanlab.init(
            project="Qwen-NER-SFT",
            config=vars(config)
        )
        trainer.evaluate_test()
        swanlab.finish()
