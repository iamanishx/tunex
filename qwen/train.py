import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--use_qlora", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_arguments()
    print(f"Starting fine-tuning script with model: {args.model}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Compute precision: {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": compute_dtype,
    }
    
    if args.use_qlora:
        print("Loading model in 4-bit (QLoRA) mode...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        print("Loading model in standard full-precision/half-precision mode...")

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    print("Loading datasets...")
    
    local_dataset_path = "data/hinglish_dataset.json"
    if not os.path.exists(local_dataset_path):
        local_dataset_path = "/data/hinglish_dataset.json"

    if os.path.exists(local_dataset_path):
        print(f"Loading local dataset from {local_dataset_path}...")
        try:
            from datasets import Dataset
            import json
            with open(local_dataset_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
            hinglish_dataset = Dataset.from_list(local_data)
            print(f"Successfully loaded {len(hinglish_dataset)} local samples.")
        except Exception as e:
            print(f"Failed to load local dataset: {e}. Falling back to HF...")
            hinglish_dataset = None
    else:
        hinglish_dataset = None

    if hinglish_dataset is None:
        try:
            full_hinglish = load_dataset("Sujalvc/hinglish-instruct-dataset", split="train")
            hinglish_dataset = full_hinglish.shuffle(seed=42).select(range(4000))
        except Exception as e:
            print(f"Failed to load hinglish-instruct-dataset directly. Using fallbacks: {e}")
            hinglish_dataset = None

    try:
        english_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
    except Exception as e:
        print(f"Failed to load alpaca-cleaned: {e}")
        english_dataset = None

    SYSTEM_PROMPT = "Tum ek helpful AI assistant ho. Tum Hinglish mein jawab dete ho, yaani Hindi ko English letters mein likhte ho. Agar user English mein puchhe toh bhi Hinglish mein jawab do."

    def format_alpaca_to_chat(example):
        user_content = example["instruction"]
        if example.get("input") and example["input"].strip() != "" and example["input"].strip() != "<noinput>":
            user_content += f"\n\nInput:\n{example['input']}"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    dataset_list = []
    if hinglish_dataset:
        formatted_hinglish = hinglish_dataset.map(format_alpaca_to_chat, remove_columns=hinglish_dataset.column_names)
        dataset_list.append(formatted_hinglish)
    if english_dataset:
        formatted_english = english_dataset.map(format_alpaca_to_chat, remove_columns=english_dataset.column_names)
        dataset_list.append(formatted_english)

    if len(dataset_list) > 0:
        mixed_dataset = concatenate_datasets(dataset_list).shuffle(seed=42)
        print(f"Total mixed training samples: {len(mixed_dataset)}")
    else:
        print("Using dummy Hinglish dataset for local test...")
        from datasets import Dataset
        dummy_data = {
            "messages": [
                [
                    {"role": "user", "content": "Hello, aap kaise hain?"},
                    {"role": "assistant", "content": "Hello! Main bilkul theek hoon. Aap batayein, aapki kya madad kar sakta hoon?"}
                ],
                [
                    {"role": "user", "content": "Chai banane ka recipe batayein."},
                    {"role": "assistant", "content": "Chai banane ke liye, sabse pehle ek cup paani ko garam karein. Usme thodi chai patti, cheeni, aur thoda adrak (ginger) daalein. Jab paani ubal jaye, toh usme thoda doodh (milk) milayein aur 2-3 minute ke liye dheemi aanch par ubaalein. Aapki garam chai taiyaar hai!"}
                ]
            ] * 20
        }
        mixed_dataset = Dataset.from_dict(dummy_data)
        print(f"Total mixed training samples (dummy): {len(mixed_dataset)}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir="./qwen_hindi_lora",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=512,
        eos_token="<|im_end|>",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mixed_dataset,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving fine-tuned adapter...")
    save_dir = "/data/qwen_hinglish_lora" if os.path.exists("/data") else "./qwen_hindi_lora_final"
    trainer.save_model(save_dir)
    print(f"Model saved to {save_dir}.")
    
    try:
        print("Pushing model to Hugging Face Model Hub...")
        trainer.push_to_hub("iamanishx/qwen-hinglish-lora")
        print("Model pushed to Hugging Face Hub successfully!")
    except Exception as e:
        print(f"Failed to push to Hugging Face Hub: {e}")
        print("Ensure your HF_TOKEN is configured with write permissions in your Space.")

if __name__ == "__main__":
    main()
