# Running Hybrid-GCS on NVIDIA Brev GPU Cloud

**Complete Setup Guide for Cloud GPU Training**

---

## 📋 Table of Contents

1. [Brev Setup](#brev-setup)
2. [Instance Configuration](#instance-configuration)
3. [Environment Setup](#environment-setup)
4. [Running Hybrid-GCS](#running-hybrid-gcs)
5. [Monitoring & Deployment](#monitoring--deployment)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Brev Setup

### Step 1: Create Brev Account

```bash
# Go to https://console.brev.dev
# Sign up with email or Google/GitHub account
# Free tier includes 2 hours of GPU compute
```

### Step 2: Install Brev CLI

**Linux/macOS:**
```bash
brew install brevdev/homebrew-brev/brev
```

**Windows (WSL2):**
```bash
# Install Homebrew in WSL
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Brev
brew install brevdev/homebrew-brev/brev
```

### Step 3: Authenticate with Brev

```bash
brev login

# This opens browser for authentication
# Follow the prompts to authenticate
```

### Step 4: Verify Installation

```bash
brev ls
# Should show your existing instances (if any)

brev version
# Should show CLI version
```

---

## 💾 Instance Configuration

### Option 1: Create Instance via CLI (Recommended)

```bash
# Create instance with A100 GPU (high performance)
brev create hybrid-gcs-gpu \
  --gpu a100:1 \
  --machine-type high-performance \
  --disk 100

# Create instance with T4 GPU (cost-effective)
brev create hybrid-gcs-gpu \
  --gpu t4:1 \
  --machine-type standard \
  --disk 50

# Create instance with RTX 4090 (consumer-grade)
brev create hybrid-gcs-gpu \
  --gpu rtx-4090:1 \
  --machine-type standard \
  --disk 50
```

### Option 2: Create Instance via Web Console

1. Go to `https://console.brev.dev`
2. Click **"Create New Instance"**
3. Select **"GPU Instances"**
4. Choose GPU type:
   - **A100** - Best for production (40GB VRAM)
   - **V100** - Excellent for training (16GB VRAM)
   - **T4** - Cost-effective (16GB VRAM)
   - **RTX 4090** - Consumer-grade (24GB VRAM)
5. Click **"Deploy"**
6. Wait for deployment (2-5 minutes)

### GPU Selection Guide

| GPU | VRAM | Cost/Hour | Best For |
|-----|------|-----------|----------|
| A100 | 40GB | $3-5 | Production training |
| V100 | 16GB | $2-3 | Large models |
| T4 | 16GB | $0.35 | Budget training |
| RTX 4090 | 24GB | $0.50 | Research |

---

## 🔧 Environment Setup

### Step 1: Connect to Instance

```bash
# Open in VS Code (recommended)
brev open hybrid-gcs-gpu

# Or SSH into instance
brev shell hybrid-gcs-gpu

# Or use direct SSH
ssh ubuntu@<instance-ip>
```

### Step 2: Clone Hybrid-GCS Repository

```bash
# SSH into instance first
brev shell hybrid-gcs-gpu

# Clone repository
git clone https://github.com/hybrid-gcs/hybrid-gcs.git
cd hybrid-gcs

# Verify CUDA
nvidia-smi
# Should show your GPU details
```

### Step 3: Setup Python Environment

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Verify Python
python --version
# Should show Python 3.10.x
```

### Step 4: Install Hybrid-GCS

```bash
# Install in development mode
pip install -e ".[dev,jupyter]"

# Verify installation
python -c "from hybrid_gcs import GCSDecomposer; print('✅ Installation successful!')"
```

### Step 5: Install Additional Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Jupyter for interactive development
pip install jupyter jupyterlab ipywidgets

# Verify GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name()}')"
```

---

## 🎯 Running Hybrid-GCS

### Option 1: Command Line Training

```bash
# Basic training
python scripts/train.py --env manipulation

# With configuration
python scripts/train.py \
  --config configs/training/standard.yaml \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --total-steps 100000

# With specific preset
python scripts/train.py \
  --preset high_quality \
  --output results/

# Monitor with TensorBoard
tensorboard --logdir results/logs/
# Access at: http://localhost:6006
```

### Option 2: Jupyter Notebook (Recommended for Brev)

```bash
# Start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Copy the token URL from output
# Open in browser or via Brev console

# Or use Brev tunnel
brev jupyter hybrid-gcs-gpu
# Automatically opens in browser
```

**Jupyter Notebook Example:**

```python
# Cell 1: Setup
from hybrid_gcs.training.configs import ConfigLoader
from hybrid_gcs.training import OptimizedTrainer
from hybrid_gcs.environments import ManipulationEnvironment
import torch

# Cell 2: Check GPU
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# Cell 3: Load Configuration
loader = ConfigLoader()
config = loader.load_config(
    preset='high_quality',
    overrides={'batch_size': 256}
)

# Cell 4: Create Environment
env = ManipulationEnvironment(state_dim=20, action_dim=6)

# Cell 5: Initialize Trainer
trainer = OptimizedTrainer(
    config=config,
    actor=actor_network,
    critic=critic_network,
    env=env
)

# Cell 6: Train
metrics = trainer.train()
print(f"Training complete: {metrics}")
```

### Option 3: Docker Container (Advanced)

```bash
# Build Docker image
docker build -t hybrid-gcs:latest -f docker/Dockerfile .

# Run training in container
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  hybrid-gcs:latest \
  python scripts/train.py --env manipulation
```

---

## 📊 Monitoring & Deployment

### Real-Time Monitoring

```bash
# Option 1: TensorBoard
tensorboard --logdir results/logs/

# Option 2: Built-in Dashboard
python scripts/dashboard.py --port 5000

# Option 3: Shell monitoring
watch -n 1 nvidia-smi
```

### Access via Brev Tunnel

```bash
# Forward ports through Brev tunnel
brev open hybrid-gcs-gpu

# Or use direct tunnel
brev shell hybrid-gcs-gpu
# Then access internal services via port forwarding
```

### Save Results Locally

```bash
# From your local machine
brev scp hybrid-gcs-gpu:/home/ubuntu/hybrid-gcs/results ./results

# Or via SFTP
# Use VS Code remote to sync files automatically
```

### Model Checkpointing

```bash
# Trained models are saved to:
# results/checkpoints/

# Download best model
brev scp hybrid-gcs-gpu:/home/ubuntu/hybrid-gcs/results/models/best.pt ./models/

# Resume training from checkpoint
python scripts/train.py --resume results/checkpoints/checkpoint_latest.pt
```

---

## 💰 Cost Optimization

### Stop Instance When Not in Use

```bash
# Stop instance (keeps data, saves costs)
brev stop hybrid-gcs-gpu

# List instances
brev ls

# Resume instance
brev create hybrid-gcs-gpu --resume

# Delete instance permanently
brev delete hybrid-gcs-gpu
```

### Spot Instances (70% Discount)

```bash
# Create spot instance (can be interrupted)
brev create hybrid-gcs-gpu \
  --gpu a100:1 \
  --spot

# Spot instances are cheaper but may be interrupted
# Use checkpointing to resume training
```

### Budget Management

1. Go to https://console.brev.dev/settings
2. Set **Monthly Budget** to limit spending
3. Enable **Spot Instances** for discounts
4. Set **Auto-Stop** after idle time

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solution:**

```bash
# Reduce batch size
python scripts/train.py --batch-size 128

# Or reduce model size
python scripts/train.py \
  --config configs/training/fast_train.yaml

# Check GPU memory
nvidia-smi
```

### Issue 2: Connection Timeout

**Symptoms:** `Connection refused` or `timeout`

**Solution:**

```bash
# Verify instance is running
brev ls

# Check instance status
brev status hybrid-gcs-gpu

# Reconnect
brev shell hybrid-gcs-gpu

# Or restart instance
brev stop hybrid-gcs-gpu
brev create hybrid-gcs-gpu --resume
```

### Issue 3: Low Disk Space

**Symptoms:** `No space left on device`

**Solution:**

```bash
# Check disk usage
df -h

# Clean up old checkpoints
rm -rf results/checkpoints/checkpoint_*.pt

# Or increase disk size via Brev console
```

### Issue 4: Slow Training

**Symptoms:** Low GPU utilization, slow progress

**Solution:**

```bash
# Check GPU utilization
nvidia-smi dmon

# Increase batch size
python scripts/train.py --batch-size 512

# Use faster preset
python scripts/train.py --preset fast_train

# Check network connectivity
ping github.com
```

### Issue 5: Authentication Issues

**Symptoms:** `401 Unauthorized` or expired token

**Solution:**

```bash
# Re-authenticate
brev logout
brev login

# Verify authentication
brev ls
```

---

## 📚 Advanced Usage

### Multi-GPU Training

```bash
# With Horovod
horovodrun -np 2 -H localhost:2 \
  python scripts/train.py --env manipulation

# Or use PyTorch DDP
torchrun --nproc_per_node=2 scripts/train.py
```

### Custom Training Script

```python
# custom_train.py
from hybrid_gcs.training.configs import ConfigLoader
from hybrid_gcs.training import OptimizedTrainer

# Load config
loader = ConfigLoader()
config = loader.load_config(
    preset='high_quality',
    overrides={
        'total_timesteps': 10_000_000,
        'batch_size': 512,
        'learning_rate': 1e-4
    }
)

# Train (your custom setup here)
trainer = OptimizedTrainer(...)
metrics = trainer.train()
```

### Integration with Weights & Biases

```bash
# Install W&B
pip install wandb

# Login
wandb login

# Modify training to use W&B
# See docs/training_guide.md for integration
```

---

## 🎓 Quick Reference

| Task | Command |
|------|---------|
| Create instance | `brev create hybrid-gcs-gpu --gpu a100:1` |
| SSH into instance | `brev shell hybrid-gcs-gpu` |
| Open in VS Code | `brev open hybrid-gcs-gpu` |
| Stop instance | `brev stop hybrid-gcs-gpu` |
| Delete instance | `brev delete hybrid-gcs-gpu` |
| List instances | `brev ls` |
| View logs | `brev logs hybrid-gcs-gpu` |
| Check GPU | `nvidia-smi` |
| Train | `python scripts/train.py --preset standard` |
| Jupyter | `jupyter lab --ip=0.0.0.0` |
| TensorBoard | `tensorboard --logdir results/logs/` |

---

## 📞 Support

- **Brev Docs**: https://docs.nvidia.com/brev/
- **Brev Console**: https://console.brev.dev
- **Hybrid-GCS Issues**: https://github.com/hybrid-gcs/hybrid-gcs/issues
- **Community**: https://brev.dev/community

---

**Happy training on Brev GPU Cloud!** 🚀🤖✨
