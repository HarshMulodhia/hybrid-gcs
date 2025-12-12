# Basic Usage Examples

Simple examples to get started with Hybrid-GCS.

## Example 1: GCS Decomposition

```python
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace

# Create configuration space
cs = ConfigSpace(
    name="robot",
    dim=6,
    bounds=[(-1, 1)] * 6
)

# Create decomposer
decomposer = GCSDecomposer(cs)

# Define problem
start = np.array([-0.8, -0.8, -0.8, 0, 0, 0])
goal = np.array([0.8, 0.8, 0.8, 0, 0, 0])
obstacles = [(np.array([0, 0, 0, 0, 0, 0]), 0.2)]

# Solve
result = decomposer.decompose(start, goal, obstacles)

print(f"Solution feasible: {result['feasible']}")
print(f"Regions: {len(result['regions'])}")
```

## Example 2: Training with PPO

```python
from hybrid_gcs.environments import ManipulationEnvironment
from hybrid_gcs.training import OptimizedTrainer

# Environment
env = ManipulationEnvironment(state_dim=20, action_dim=6)

# Trainer
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
        action = trainer.select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: {episode_reward:.2f}")
        trainer.update()
```

## Example 3: Hybrid Policy

```python
from hybrid_gcs.integration import HybridPolicy, HybridPolicyConfig
from hybrid_gcs.core import GCSDecomposer, ConfigSpace

# Setup
cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
gcs_planner = GCSDecomposer(cs)

# Hybrid policy config
config = HybridPolicyConfig(
    blend_method="weighted",
    gcs_weight=0.5,
    rl_weight=0.5
)

# Create policy
policy = HybridPolicy(config)

# Compute action
state = np.random.randn(20)
goal = np.random.randn(20)
action = policy.compute_action(state, goal)

print(f"Action: {action}")
```

## Example 4: Evaluation

```python
from hybrid_gcs.evaluation import Evaluator
from hybrid_gcs.environments import ManipulationEnvironment

# Environment
env = ManipulationEnvironment(state_dim=20, action_dim=6)

# Evaluator
evaluator = Evaluator(env)

# Run evaluation
results = evaluator.evaluate(
    num_episodes=100,
    max_steps=200
)

print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Success rate: {results['success_rate']:.1%}")
```

## Example 5: Configuration Files

Train using configuration:

```bash
# Create config
cat > my_config.yaml <<EOF
environment:
  name: manipulation
  state_dim: 20
  action_dim: 6
  max_steps: 200

training:
  num_episodes: 1000
  batch_size: 32
  learning_rate: 3e-4
EOF

# Train
python scripts/train.py --config my_config.yaml
```

---

See [Getting Started](../getting_started.md) for more.
