# Fine-Tuning LLMs with TRL: Key Learnings and Best Practices

This guide documents the technical configuration, code patterns, and training strategies developed during the fine-tuning of Qwen 2.5 0.5B on Romanized Hindi (Hinglish) instructions using Hugging Face's TRL (Transformer Reinforcement Learning) library.

---

## 1. Dataset Schema Normalization in SFTTrainer

### The Mistake
Assuming the dataset used flat `instruction`, `input`, and `output` columns. When the training script loaded `smangrul/hinglish_self_instruct_v0`, it threw key errors because the dataset uses the modern conversational `messages` format.

### The Learning
TRL's `SFTTrainer` natively supports standard OpenAI-style chat schemas. A chat schema uses a single `messages` column containing a list of dictionaries with `role` and `content` keys.

### The Solution
When combining multiple datasets, map traditional formats to the unified conversational schema:
```python
def format_alpaca_to_chat(example):
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip() not in ["", "<noinput>"]:
        user_content += f"\n\nInput:\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }
```
Select only the `messages` column from all datasets before concatenating them together.

---

## 2. Preventing Catastrophic Forgetting

### The Strategy
Fine-tuning a model solely on a narrow, language-specific dataset (like Romanized Hindi instructions) can degrade its reasoning capabilities in its native languages (such as English).

### The Implementation
Mix the target dataset with a general instruction-following dataset (e.g., Alpaca Cleaned) at a controlled ratio:
* **Target Data**: `smangrul/hinglish_self_instruct_v0` (approx. 1,000 samples)
* **General Data**: `yahma/alpaca-cleaned` (first 500 samples)
* **Result**: The model retains English comprehension and formatting skills while adapting its linguistic output to Hinglish.

---

## 3. LoRA Configuration and Target Modules

### The Strategy
Using LoRA (Low-Rank Adaptation) reduces trainable parameters to less than 2% of the original model size, allowing fine-tuning to fit on consumer-grade GPUs.

### Optimal Hyperparameters
* **Rank (`r=16`) and Alpha (`lora_alpha=32`)**: A scaling ratio of 2.0 (Alpha / Rank) is standard and stable.
* **Target All Linear Modules**: Instead of targeting only `q_proj` and `v_proj`, specify all linear projections to capture deeper linguistic structures:
  ```python
  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  ```

---

## 4. Optimization and Precision

### Memory Management
* **QLoRA (4-bit Mode)**: Crucial for local machines with limited VRAM (e.g., RTX 4050 6GB). It uses a NormalFloat4 (`nf4`) quantization format with double quantization enabled.
* **Full Precision (16-bit Mode)**: Faster on cloud GPUs with larger VRAM (e.g., Tesla T4 16GB) where quantization overhead can be avoided.

### Precision Selectors
Always check hardware compatibility to choose between `bfloat16` and `float16`. Modern GPUs support `bfloat16` which has a larger dynamic range and prevents underflow:
```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```
Use `bf16` or `fp16` parameters in the training arguments accordingly.
