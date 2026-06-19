import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

def parse_arguments():
    p = argparse.ArgumentParser(description="Generalized SFT training script using LoRA.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model Hugging Face ID")
    p.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="HF Dataset path or local JSON path")
    p.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    p.add_argument("--output_dir", type=str, default="/data/axeai_lora_output", help="Local save path for adapter")
    p.add_argument("--hub_repo", type=str, default=None, help="Optional HF hub repository to push adapter to")
    
    p.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (LoRA baseline)")
    p.add_argument("--r", type=int, default=64, help="LoRA rank")
    p.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    p.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    
    p.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA quantization")
    p.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    return p.parse_args()

def main():
    args = parse_arguments()
    print(f"Loading tokenizer and model: {args.model}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Precision: {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": compute_dtype,
    }
    
    if args.use_qlora:
        print("Enabling 4-bit QLoRA...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset.endswith(".json") or os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split=args.dataset_split)
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)

    # Automatic formatting to ChatML format if dataset uses alpaca structure
    sample_entry = dataset[0]
    if "messages" not in sample_entry and "instruction" in sample_entry and "output" in sample_entry:
        print("Converting dataset from Alpaca instruction-output format to chat conversation format...")
        def format_to_chat(example):
            user_content = example["instruction"]
            if example.get("input") and example["input"].strip() != "" and example["input"].strip() != "<noinput>":
                user_content += f"\n\nInput:\n{example['input']}"
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": example["output"]},
                ]
            }
        dataset = dataset.map(format_to_chat, remove_columns=dataset.column_names)

    # Setup LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Setup training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=args.max_length,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    print("Saved locally successfully!")
    
    if args.hub_repo:
        print(f"Pushing model to Hugging Face Model Hub: {args.hub_repo}")
        try:
            trainer.push_to_hub(args.hub_repo)
            print("Successfully pushed to Hugging Face Hub!")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    main()
