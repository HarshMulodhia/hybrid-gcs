# Troubleshooting Guide

Solutions to common issues.

## Installation Issues

### Import Error

```
ModuleNotFoundError: No module named 'hybrid_gcs'
```

**Solution:**
```bash
# Make sure you're in virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall
pip install -e .
```

### CUDA Not Available

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python scripts/train.py

# Or reduce batch size
python scripts/train.py --batch-size 16
```

## Training Issues

### Training Unstable

**Symptoms:** Reward jumps around, doesn't converge

**Solution:**
```yaml
# In config file
training:
  learning_rate: 1e-4  # Lower
  batch_size: 64       # Increase
  entropy_coefficient: 0.001  # Lower
```

### Slow Convergence

**Symptoms:** Training takes very long

**Solution:**
```python
# Use curriculum learning
curriculum = CurriculumLearning(
    schedule_type="exponential",  # Faster
    total_steps=100000
)

# Or increase learning rate
trainer = OptimizedTrainer(
    learning_rate=1e-3  # Higher
)
```

### Out of Memory

**Symptoms:** OOM during training

**Solution:**
```bash
# Reduce batch size
python scripts/train.py --batch-size 16

# Reduce episode length
python scripts/train.py --max-steps 100

# Reduce model size
# Edit configs/training/default.yaml
```

## Environment Issues

### Environment Not Rendering

**Symptoms:** No visualization

**Solution:**
```python
env = ManipulationEnvironment(
    render_mode='human'  # Enable rendering
)

# Or check GUI
export DISPLAY=:0  # Linux
```

### Reward Always Zero

**Symptoms:** No learning signal

**Solution:**
```python
# Check reward function
from hybrid_gcs.training import RewardShaper
shaper = RewardShaper()
print(shaper.compute_reward(state, action))

# Verify observation
obs = env.reset()
print(obs.shape, obs.dtype)
```

## Evaluation Issues

### Model Not Loading

```
Error loading model: 'model.pt' not found
```

**Solution:**
```bash
# Check path
ls -la results/models/

# Use full path
python scripts/evaluate.py \
    --model /full/path/to/model.pt
```

### Inconsistent Results

**Symptoms:** Different results on reruns

**Solution:**
```python
import numpy as np
import torch

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Run evaluation
python scripts/evaluate.py --seed 42
```

## Docker Issues

### Build Fails

**Solution:**
```bash
# Clean build
docker build --no-cache -t hybrid-gcs:2.0.0 .

# Check Dockerfile
docker build -f docker/Dockerfile -t hybrid-gcs .
```

### Container Exits

**Solution:**
```bash
# Check logs
docker-compose logs hybrid-gcs

# Run interactively
docker-compose run hybrid-gcs bash
```

## Deployment Issues

### Robot Connection Failed

**Solution:**
```python
# Test connection
from hybrid_gcs.integration import RobotController
controller = RobotController("192.168.1.100")
controller.ping()

# Check network
ping 192.168.1.100
```

### Safety Mechanism Triggered

**Solution:**
```python
# Check limits
robot.check_joint_limits()
robot.check_velocity_limits()
robot.check_torque_limits()

# Reset
robot.reset()
robot.enable_safety_mode()
```

## Getting Help

1. **Check logs:**
   ```bash
   tail -f results/logs/training.log
   ```

2. **Run diagnostics:**
   ```bash
   python -m hybrid_gcs.diagnostics
   ```

3. **Open issue:**
   ```
   https://github.com/hybrid-gcs/hybrid-gcs/issues
   ```

4. **Contact support:**
   ```
   team@hybrid-gcs.dev
   ```

---

Still stuck? See [Documentation](index.md) or open an [issue](https://github.com/hybrid-gcs/hybrid-gcs/issues).
