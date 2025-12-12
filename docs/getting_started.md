# Getting Started Guide

Get started with Hybrid-GCS in 5 minutes!

## 1. Installation

```bash
# Install package
pip install hybrid-gcs

# Or for development
git clone https://github.com/hybrid-gcs/hybrid-gcs.git
cd hybrid-gcs
pip install -e .
```

## 2. Your First GCS Decomposition

```python
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace

# Create configuration space
config_space = ConfigSpace(
    name="robot_config",
    dim=6,  # 6D configuration space
    bounds=[(-1, 1)] * 6  # Each dimension from -1 to 1
)

# Create decomposer
decomposer = GCSDecomposer(config_space=config_space)

# Define start and goal
start = np.array([-0.5, -0.5, -0.5, 0, 0, 0])
goal = np.array([0.5, 0.5, 0.5, 0, 0, 0])
obstacles = []  # No obstacles for now

# Compute decomposition
result = decomposer.decompose(start, goal, obstacles)

print(f"Feasible: {result['feasible']}")
print(f"Number of regions: {len(result['regions'])}")
```

## 3. Training with RL

```python
from hybrid_gcs.environments import ManipulationEnvironment
from hybrid_gcs.training import OptimizedTrainer
import numpy as np

# Create environment
env = ManipulationEnvironment(
    state_dim=20,
    action_dim=6,
    task="reach"
)

# Create trainer
trainer = OptimizedTrainer(
    policy_dim=20,
    action_dim=6,
    learning_rate=3e-4,
    batch_size=32
)

# Training loop
for episode in range(100):
    obs = env.reset()
    episode_reward = 0
    
    for step in range(200):
        # Random action for demo (use policy in practice)
        action = np.random.randn(6) * 0.1
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
```

## 4. Using Hybrid Policy

```python
from hybrid_gcs.integration import HybridPolicy, HybridPolicyConfig

# Configure hybrid policy
config = HybridPolicyConfig(
    blend_method="weighted",
    gcs_weight=0.5,
    rl_weight=0.5
)

# Create hybrid policy
# (Would use actual decomposer and policy in practice)

state = np.random.randn(20)
goal = np.random.randn(20)

# Compute action (blending GCS plan with RL policy)
# action = hybrid_policy.compute_action(state, goal)
print(f"State: {state.shape}, Goal: {goal.shape}")
```

## 5. Evaluation

```python
from hybrid_gcs.evaluation import Evaluator

# Create evaluator
evaluator = Evaluator(env)

# Evaluate policy
num_episodes = 100
results = evaluator.evaluate(num_episodes=num_episodes)

print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Std reward: {results['std_reward']:.2f}")
print(f"Success rate: {results['success_rate']:.1%}")
```

## Next Steps

- [Read the full training guide](training_guide.md)
- [Explore examples](examples/)
- [Check API reference](api_reference.md)
- [Deploy to robot](deployment.md)

## Common Tasks

### Custom Environment

See [Creating Custom Environments](examples/custom_environment.md)

### Real Robot Integration

See [Real-World Deployment](examples/real_world_deployment.md)

### Docker Deployment

```bash
docker-compose up -d
docker-compose exec hybrid-gcs python scripts/train.py --env manipulation
```

## Troubleshooting

- **Import error?** → See [Installation Guide](installation.md)
- **GPU issues?** → Check [Troubleshooting](troubleshooting.md)
- **Want examples?** → Browse [Examples](examples/)

---

Ready to dive deeper? Check the [full documentation](index.md)!
