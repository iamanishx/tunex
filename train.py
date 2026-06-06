import argparse
import os
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

class PyTorchProfilerCallback(TrainerCallback):
    def __init__(self, warmup_steps, active_steps, trace_dir):
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trace_dir = trace_dir
        self.prof = None

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"--> Starting PyTorch Profiler (warmup={self.warmup_steps}, active={self.active_steps})...")
        os.makedirs(self.trace_dir, exist_ok=True)
        
        import torch.profiler
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=1
        )
        
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof:
            print(f"--> Stopping PyTorch Profiler. Saving trace to {self.trace_dir}...")
            self.prof.stop()
            self.prof = None

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--active_steps", type=int, default=3)
    p.add_argument("--trace_dir", default="./traces/qwen_profile")
    p.add_argument("--use_qlora", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
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
        tokenizer.pad_token = tokenizer.eos_token

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
    
    try:
        hinglish_dataset = load_dataset("smangrul/hinglish_self_instruct_v0", split="train")
    except Exception as e:
        print(f"Failed to load hinglish_self_instruct_v0 directly. Using fallbacks: {e}")
        hinglish_dataset = None

    try:
        english_dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
    except Exception as e:
        print(f"Failed to load alpaca-cleaned: {e}")
        english_dataset = None

    def format_alpaca_to_chat(example):
        user_content = example["instruction"]
        if example.get("input") and example["input"].strip() != "" and example["input"].strip() != "<noinput>":
            user_content += f"\n\nInput:\n{example['input']}"
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    dataset_list = []
    if hinglish_dataset:
        formatted_hinglish = hinglish_dataset.select_columns(["messages"])
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir="./qwen_hindi_lora",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=10,
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
        callbacks=[PyTorchProfilerCallback(args.warmup_steps, args.active_steps, args.trace_dir)]
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving fine-tuned adapter...")
    trainer.save_model("./qwen_hindi_lora_final")
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
