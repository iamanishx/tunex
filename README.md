### general lora fine tuning setup

this project provides scripts and notebooks to fine-tune qwen and llama models using low-rank adaptation (lora) with optimal hyperparameters.

### project structure

*   `train.py`: a generalized sft training script that accepts command line arguments to fine-tune any hugging face model.
*   `lora_finetuning.ipynb`: a generalized jupyter notebook for running the training loop in cloud environments (colab, jupyterlab).
*   `qwen/train.py`: our specific training script designed for the qwen hinglish task.
*   `chat.py`: a cli interface to chat with the base model and your trained lora adapter.
*   `compare.py`: a tool to test the base model and your fine-tuned model side by side on prompts.
*   `distill.py`: a dataset cleaning and rewriting pipeline.
*   `check_dataset.py`: a dataset validation utility.

### baseten sft best practices and hyperparameters

we have configured the training parameters in this repository to use the optimal defaults derived from baseten's post-training research. the full research paper can be accessed here:
https://www.datocms-assets.com/104802/1781805778-baseten-research-sft.pdf

here is why these specific hyperparameter values are selected:

### learning rate set to 1e-3

baseten's sweeps showed that the optimal lora learning rate is scale-invariant and sits flat at 1e-3 (0.001) for all model sizes from 0.6b up to 235b. this is 10 times higher than previous common defaults and allows the model to learn the target style and vocabulary much faster.

### lora rank set to 64 and alpha set to 32

their experiments show that rank 8 and 16 are under-capacity, while rank 64 provides the optimal capacity for the adapter. rank 128 does not yield any meaningful validation loss improvement but doubles the parameter overhead. keeping alpha at 32 at the 1e-3 learning rate sets the correct scaling factor.

### training epochs set to 2

training beyond 2 epochs on the same dataset causes the validation loss to overfit and degrades the model's general instruction-following capabilities (ifeval). to get better results, it is better to add fresh data rather than repeating passes beyond 2 epochs.

### global batch size set to 16

the training configuration uses 16 gradient accumulation steps to reach a global batch size of 16, which is the optimal throughput and loss trade-off point found in the sweeps.

### getting started

first, install all required packages:
```bash
pip install torch transformers peft datasets accelerate bitsandbytes trl huggingface_hub
```

### running the generalized training script

to run the training script from the root of this project:
```bash
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset yahma/alpaca-cleaned --epochs 2
```

### running the qwen hinglish training script

to run the specific qwen hinglish training script:
```bash
python qwen/train.py --epochs 2
```
