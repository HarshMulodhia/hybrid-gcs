# Training Guide

Complete guide to training models with Hybrid-GCS.

## Overview

This guide covers training policies using PPO with optional curriculum learning and reward shaping.

## Basic Training

```python
from hybrid_gcs.environments import ManipulationEnvironment
from hybrid_gcs.training import OptimizedTrainer, CurriculumLearning
import numpy as np

# Setup
env = ManipulationEnvironment(state_dim=20, action_dim=6)
trainer = OptimizedTrainer(
    policy_dim=20,
    action_dim=6,
    learning_rate=3e-4,
    batch_size=32,
    gamma=0.99,
    gae_lambda=0.95
)

# Training loop
for episode in range(1000):
    obs = env.reset()
    episode_reward = 0
    
    for step in range(200):
        action = trainer.select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    # Update policy
    if (episode + 1) % 10 == 0:
        trainer.update()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
```

## Advanced Training with Curriculum

```python
# Add curriculum learning
curriculum = CurriculumLearning(
    schedule_type="linear",
    total_steps=100000
)

for episode in range(1000):
    # Get difficulty
    difficulty = curriculum.get_difficulty(episode * 200)
    env.set_difficulty(difficulty)
    
    obs = env.reset()
    
    for step in range(200):
        action = trainer.select_action(obs)
        obs, reward, done, info = env.step(action)
        
        if done:
            break
```

## Configuration Files

Use YAML configs for reproducible training:

```bash
python scripts/train.py --config configs/training/default.yaml
```

### Config Structure

```yaml
environment:
  name: manipulation
  state_dim: 20
  action_dim: 6
  max_steps: 200

training:
  num_episodes: 1000
  batch_size: 32
  learning_rate: 3e-4

curriculum:
  enabled: true
  type: linear
```

## Hyperparameter Tuning

Key hyperparameters:

| Parameter | Range | Default | Impact |
|-----------|-------|---------|--------|
| Learning Rate | 1e-5 to 1e-3 | 3e-4 | Convergence speed |
| Batch Size | 16 to 128 | 32 | Stability |
| Gamma | 0.9 to 0.999 | 0.99 | Horizon |
| GAE Lambda | 0.90 to 0.99 | 0.95 | Variance |

## Monitoring Training

```bash
# TensorBoard
tensorboard --logdir results/logs/

# Watch training
python scripts/train.py --visualize
```

## Best Practices

1. **Start with default config** - Use fast_train for quick testing
2. **Monitor rewards** - Use TensorBoard to visualize
3. **Use curriculum** - Gradually increase difficulty
4. **Validate regularly** - Run evaluation every 100 episodes
5. **Save checkpoints** - Keep best models

## Common Issues

### Training Unstable
- Reduce learning rate
- Increase batch size
- Lower entropy coefficient

### Slow Convergence
- Increase learning rate
- Use curriculum learning
- Try different reward shaping

### Memory Issues
- Reduce batch size
- Reduce episode length
- Clear old checkpoints

---

See [Evaluation Guide](evaluation.md) for testing trained models.
