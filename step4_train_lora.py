import argparse
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

def process_func(example):
    MAX_LENGTH = 384
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def load_dataset(data_path):
    ext = os.path.splitext(data_path)[-1]
    if ext == '.jsonl':
        df = pd.read_json(data_path, lines=True)
    elif ext == '.json':
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"不支持的数据集格式: {ext}")
    return Dataset.from_pandas(df)

def main():
    parser = argparse.ArgumentParser(description="LLaMA3 LoRA 微调训练脚本")
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', help='基础模型路径')
    parser.add_argument('--data_path', type=str, default='./dataset/huanhuan.json', help='训练数据集路径')
    parser.add_argument('--output_dir', type=str, default='./output/llama3_1_instruct_lora', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4, help='每卡batch size')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--save_steps', type=int, default=100, help='保存步数')
    args = parser.parse_args()

    global tokenizer
    print(f"加载模型和分词器: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"加载数据集: {args.data_path}")
    ds = load_dataset(args.data_path)
    print(f"数据集样本数: {len(ds)}")
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("开始训练...")
    trainer.train()
    print("训练完成！")

if __name__ == "__main__":
    main() 