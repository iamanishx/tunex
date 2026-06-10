# Hugging Face Spaces: Key Learnings and Best Practices

This document outlines the operational lessons, common pitfalls, and architectural insights gained during the setup and execution of fine-tuning jobs on Hugging Face Space containers.

---

## 1. Ephemeral vs. Persistent Storage

### The Mistake
Saving code files (like `train.py`) and installing packages (via `pip`) in root directories or under `/home/user/` led to data loss. Every time the Space restarted due to inactivity or a crash, all custom scripts and installed dependencies were wiped out.

### The Learning
Hugging Face Spaces run inside Docker containers. Only designated directories (specifically `/data` in persistent storage spaces) remain unaffected by container restarts.

### The Solution
* **Store Everything in `/data`**: Ensure all training scripts, logs, adapters, and configuration files reside in `/data/` (e.g., `/data/train.py`).
* **Automate Environment Setup**: Use a persistent shell script (`run_training.sh`) placed in `/data` to check for and reinstall necessary packages automatically upon container boot:
  ```bash
  python -c "import trl" 2>/dev/null || pip install trl transformers datasets peft bitsandbytes accelerate torch
  ```

---

## 2. Resource Starvation from PyTorch Profiler

### The Mistake
Enabling the PyTorch Profiler with high-overhead settings (`with_stack=True`, `profile_memory=True`, `record_shapes=True`) caused JupyterLab to freeze and throw a connection failure:
`Launcher Error: Invalid response: 206`

### The Learning
The PyTorch Profiler is a very heavy tool. During its serialization phase (`profiler_stop`), it attempts to resolve system call stacks, memory maps, and tensor shapes. This process:
1. Locks the CPU at 100% capacity.
2. Swaps massive amounts of trace data in system RAM.
3. Overloads disk write bandwidth.

Because Hugging Face Space containers have strict resource allocations, this activity starves the Jupyter server, causing the container backend to crash.

### The Solution
* **Profile Locally**: Run heavy trace operations on a local machine (e.g., a laptop GPU) where system RAM and CPU are more flexible.
* **Use Lightweight Profiling in the Cloud**: If profiling in the cloud is required, disable the memory, stack, and shape details:
  ```python
  self.prof = torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
      on_trace_ready=torch.profiler.tensorboard_trace_handler(self.trace_dir),
      record_shapes=False,
      profile_memory=False,
      with_stack=False
  )
  ```

---

## 3. Background Job Management

### The Mistake
Launching training directly in the foreground blocked the interactive terminal session, making the terminal prone to disconnects due to network drops or browser sleep modes.

### The Learning
Foreground training is highly fragile in cloud interfaces. Running training in the background allows the job to continue even if the web browser window is closed.

### The Solution
* **Run with `nohup`**: Use `nohup` combined with the unbuffered output flag (`-u`) to push the job to the background and redirect output to a file:
  ```bash
  nohup python -u /data/train.py --epochs 3 > /data/train.log 2>&1 &
  ```
* **Verify GPU and System Process**:
  * Check running Python processes using: `ps aux | grep python`
  * Check active GPU memory occupation (and verify that ~3GB VRAM is occupied for Qwen-0.5B) using: `nvidia-smi`
