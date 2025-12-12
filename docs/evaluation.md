# Evaluation Guide

Evaluate trained models and analyze performance.

## Basic Evaluation

```python
from hybrid_gcs.evaluation import Evaluator
from hybrid_gcs.environments import ManipulationEnvironment
import numpy as np

# Create environment
env = ManipulationEnvironment(state_dim=20, action_dim=6)

# Create evaluator
evaluator = Evaluator(env)

# Evaluate
results = evaluator.evaluate(
    num_episodes=100,
    max_steps=200,
    render=False
)

print(f"Mean Reward: {results['mean_reward']:.2f}")
print(f"Std Reward: {results['std_reward']:.2f}")
print(f"Min Reward: {results['min_reward']:.2f}")
print(f"Max Reward: {results['max_reward']:.2f}")
print(f"Success Rate: {results['success_rate']:.1%}")
```

## Metrics

### Performance Metrics
- **Mean Reward** - Average episode reward
- **Success Rate** - % of episodes succeeding
- **Episode Length** - Average steps to completion
- **Return Distribution** - Reward histogram

### Statistical Tests

```python
from hybrid_gcs.evaluation import Comparator

# Compare two policies
comparator = Comparator(env)

stats = comparator.compare(
    policy1="results/model_v1.pt",
    policy2="results/model_v2.pt",
    num_episodes=100
)

print(f"Win Rate: {stats['win_rate']:.1%}")
print(f"Statistical Significance: {stats['p_value']:.4f}")
```

## Analysis

```python
from hybrid_gcs.evaluation import Analyzer

analyzer = Analyzer()

# Analyze training logs
analysis = analyzer.analyze("results/logs/training.log")

print(f"Learning Curve: {analysis['learning_curve']}")
print(f"Convergence Step: {analysis['convergence_step']}")
print(f"Final Performance: {analysis['final_reward']:.2f}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Reward distribution
axes[0, 0].hist(results['rewards'], bins=30)
axes[0, 0].set_title('Reward Distribution')

# Learning curve
axes[0, 1].plot(results['episode_rewards'])
axes[0, 1].set_title('Learning Curve')

# Episode lengths
axes[1, 0].plot(results['episode_lengths'])
axes[1, 0].set_title('Episode Lengths')

# Success over time
axes[1, 1].plot(results['success_rate_rolling'])
axes[1, 1].set_title('Rolling Success Rate')

plt.tight_layout()
plt.savefig('evaluation.png')
```

## CLI Evaluation

```bash
# Evaluate model
python scripts/evaluate.py \
    --model results/models/best.pt \
    --num-episodes 100 \
    --env manipulation

# Generate report
python scripts/generate_report.py \
    --results-dir results/ \
    --output report.html
```

## Benchmarking

```bash
# Run benchmarks
python scripts/benchmark.py \
    --output benchmark_results.json \
    --log-level INFO
```

---

See [Deployment Guide](deployment.md) for deploying evaluated models.
