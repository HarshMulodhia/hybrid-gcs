# Custom Environment Guide

Create custom environments for Hybrid-GCS.

## Creating a Custom Environment

```python
from gym import Env
from gym.spaces import Box
import numpy as np

class CustomEnvironment(Env):
    """Custom environment template."""
    
    def __init__(self, state_dim=20, action_dim=6, max_steps=200):
        """Initialize environment."""
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define spaces
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.goal = np.random.randn(self.state_dim).astype(np.float32)
        return self.state
    
    def step(self, action):
        """Execute action."""
        self.current_step += 1
        
        # Update state
        self.state = self.state + action[:self.state_dim] * 0.1
        
        # Compute reward
        distance = np.linalg.norm(self.state - self.goal)
        reward = -distance  # Negative distance as reward
        
        # Check if done
        done = (distance < 0.1) or (self.current_step >= self.max_steps)
        
        # Info
        info = {'distance': distance}
        
        return self.state.astype(np.float32), reward, done, info
    
    def render(self, mode='human'):
        """Render environment."""
        print(f"State: {self.state}, Goal: {self.goal}")
```

## Using Custom Environment with Training

```python
from hybrid_gcs.training import OptimizedTrainer

# Create custom environment
env = CustomEnvironment(state_dim=20, action_dim=6)

# Train
trainer = OptimizedTrainer(policy_dim=20, action_dim=6)

for episode in range(100):
    obs = env.reset()
    
    for step in range(200):
        action = trainer.select_action(obs)
        obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    if (episode + 1) % 10 == 0:
        trainer.update()
        print(f"Episode {episode + 1}")
```

## Physics-Based Environment

```python
import pybullet as p
import pybullet_data

class PhysicsEnvironment(Env):
    """Physics-based environment."""
    
    def __init__(self):
        """Initialize with PyBullet."""
        super().__init__()
        
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load robot
        self.robot_id = p.loadURDF("r2d2.urdf")
        
        # Space definitions
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,))
        self.action_space = Box(low=-1, high=1, shape=(6,))
    
    def reset(self):
        """Reset physics simulation."""
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 1],
            [0, 0, 0, 1]
        )
        
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        """Physics step."""
        # Apply action
        p.setJointMotorControlArray(
            self.robot_id,
            list(range(6)),
            p.POSITION_CONTROL,
            targetPositions=action
        )
        
        # Simulation step
        p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        return obs, reward, False, {}
    
    def _get_observation(self):
        """Get observation from simulation."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        joint_states = p.getJointStates(
            self.robot_id,
            list(range(6))
        )
        
        obs = np.concatenate([
            np.array(pos),
            np.array(orn),
            np.array([state[0] for state in joint_states]),
            np.array([state[1] for state in joint_states])
        ])
        
        return obs[:20]  # Return first 20 dims
    
    def _compute_reward(self):
        """Compute reward from simulation state."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        goal = np.array([1, 0, 1])
        
        distance = np.linalg.norm(np.array(pos) - goal)
        return -distance
    
    def close(self):
        """Close connection."""
        p.disconnect(self.physics_client)
```

---

See [Training Guide](../training_guide.md) for training examples.
