## Qwen Hinglish Fine-Tuning

This project is a simple setup for fine-tuning the Qwen 2.5 0.5B model to speak Hinglish (Hindi written using English words). It runs locally and is designed to fit on consumer GPUs like the laptop RTX 4050 with 6GB of VRAM.

To prevent the model from forgetting English, the training mixes Hinglish data with English instruction data.

### Getting Started

First, install the necessary libraries:
```bash
pip install trl transformers datasets peft bitsandbytes accelerate torch tensorboard torch_tb_profiler
```

### Running the Training Locally

Run the training script with PyTorch profiling enabled:
```bash
python train.py --warmup_steps 3 --active_steps 3 --trace_dir ./traces/qwen_profile
```

If your GPU runs out of VRAM, you can load the model in 4-bit mode by adding the QLoRA flag:
```bash
python train.py --use_qlora
```

### Running on Google Colab

If you want to train on a cloud GPU (like a free T4 GPU in Google Colab), open and run the notebook:
*   [Qwen_Hinglish_Finetuning.ipynb](file:///c:/Users/manis/trl/Qwen_Hinglish_Finetuning.ipynb)

This notebook mounts your Google Drive and automatically saves the final fine-tuned adapter weights to a folder named `qwen_hinglish_lora` directly on your Google Drive.

### Viewing the Profile

To check memory usage and step execution speeds, launch TensorBoard:
```bash
tensorboard --logdir ./traces/qwen_profile
```
