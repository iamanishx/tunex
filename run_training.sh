#!/bin/bash
# Install dependencies if not already present
python -c "import trl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install trl transformers datasets peft bitsandbytes accelerate torch
fi

# Ensure train.py is executed from /data to persist across restarts
if [ -f "/data/train.py" ]; then
    echo "Starting training from /data/train.py..."
    nohup python -u /data/train.py --epochs 3 > /data/train.log 2>&1 &
    echo "Training started in the background. PID: $!"
    echo "Monitor the training logs with: tail -f /data/train.log"
else
    echo "Error: /data/train.py not found. Please move your train.py to /data/train.py first."
fi
