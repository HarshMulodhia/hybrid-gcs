# Sim-to-Real Transfer Guide

Transfer policies from simulation to real robots.

## Domain Randomization

```python
from hybrid_gcs.environments import ManipulationEnvironment
import numpy as np

class SimulationWithDomainRandomization(ManipulationEnvironment):
    """Simulation with domain randomization."""
    
    def __init__(self, **kwargs):
        """Initialize with randomization."""
        super().__init__(**kwargs)
        self.randomize_parameters()
    
    def randomize_parameters(self):
        """Randomize simulation parameters."""
        self.friction = np.random.uniform(0.4, 0.8)
        self.restitution = np.random.uniform(0.1, 0.5)
        self.mass_scale = np.random.uniform(0.95, 1.05)
        self.gravity = np.random.uniform(9.6, 9.9)
    
    def reset(self):
        """Reset with randomization."""
        self.randomize_parameters()
        return super().reset()
```

## Calibration

```python
class RealRobotCalibration:
    """Calibrate real robot to match simulation."""
    
    def __init__(self, robot_controller):
        """Initialize calibration."""
        self.robot = robot_controller
        self.calibration_params = {}
    
    def calibrate_sensors(self):
        """Calibrate sensors."""
        # Force/torque calibration
        self.robot.calibrate_ft_sensor()
        
        # Joint encoder calibration
        self.robot.calibrate_joint_encoders()
        
        # Camera calibration
        self.robot.calibrate_camera()
    
    def estimate_dynamics(self):
        """Estimate robot dynamics."""
        # Apply known actions and measure results
        actions = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            # ... more actions
        ]
        
        for action in actions:
            initial_state = self.robot.get_state()
            self.robot.apply_action(action)
            final_state = self.robot.get_state()
            
            # Estimate dynamics
            delta = final_state - initial_state
            print(f"Action: {action}, Delta: {delta}")
```

## Transfer Validation

```python
def validate_transfer(sim_policy, real_robot):
    """Validate policy transfer to real robot."""
    
    # Test 1: Same actions produce similar results
    print("Test 1: Action consistency...")
    test_actions = [
        np.zeros(6),
        np.ones(6) * 0.1,
        np.random.randn(6) * 0.1,
    ]
    
    for action in test_actions:
        # Simulate
        sim_result = simulate_action(action)
        
        # Execute on real robot
        real_result = real_robot.execute_action(action)
        
        # Compare
        error = np.linalg.norm(sim_result - real_result)
        print(f"Error: {error:.4f}")
        
        if error > 0.5:
            print("WARNING: Large discrepancy detected!")
    
    # Test 2: Policy performance on real robot
    print("\nTest 2: Policy performance...")
    
    results = []
    for episode in range(10):
        obs = real_robot.reset()
        episode_reward = 0
        
        for step in range(100):
            action = sim_policy.compute_action(obs)
            obs, reward, done, info = real_robot.step(action)
            episode_reward += reward
            
            if done:
                break
        
        results.append(episode_reward)
    
    print(f"Mean reward: {np.mean(results):.2f}")
    print(f"Std reward: {np.std(results):.2f}")
```

## Adaptation

```python
class OnlineAdaptation:
    """Adapt policy online for real robot."""
    
    def __init__(self, initial_policy):
        """Initialize adaptation."""
        self.policy = initial_policy
        self.buffer = []
    
    def adapt(self, state, action, reward, next_state):
        """Update policy based on real experience."""
        self.buffer.append((state, action, reward, next_state))
        
        # Update policy when buffer is full
        if len(self.buffer) > 32:
            self.update_policy()
            self.buffer = []
    
    def update_policy(self):
        """Update policy with real robot data."""
        # Fine-tune policy
        for state, action, reward, next_state in self.buffer:
            loss = self.compute_loss(state, action)
            self.policy.update(loss)
```

## Multi-Robot Transfer

```python
def transfer_to_fleet(sim_policy, robots):
    """Transfer to multiple robots."""
    
    results = {}
    
    for robot_id, robot in robots.items():
        print(f"Testing robot {robot_id}...")
        
        # Calibrate robot
        robot.calibrate()
        
        # Validate transfer
        validate_transfer(sim_policy, robot)
        
        # Run on robot
        episode_rewards = []
        for episode in range(10):
            obs = robot.reset()
            episode_reward = 0
            
            for step in range(100):
                action = sim_policy.compute_action(obs)
                obs, reward, done, info = robot.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        results[robot_id] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
        }
    
    return results
```

## Best Practices

1. **Start Simple** - Test on simple tasks first
2. **Incremental** - Gradually increase task complexity
3. **Monitor** - Continuously monitor performance
4. **Adapt** - Adapt policy if needed
5. **Document** - Record all results

---

See [Real-World Deployment](real_world_deployment.md) for deployment details.
