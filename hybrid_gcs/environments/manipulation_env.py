"""
Manipulation Environment
File: hybrid_gcs/environments/manipulation_env.py

RL environment for robotic manipulation tasks.
"""

import logging
from typing import Dict, Tuple, Any
import numpy as np
from hybrid_gcs.environments.base_env import BaseEnvironment, EnvironmentConfig

logger = logging.getLogger(__name__)


class ManipulationEnvironment(BaseEnvironment):
    """
    Robotic manipulation environment.
    
    Agent learns to manipulate objects and reach target positions.
    """
    
    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 7,
        max_episode_steps: int = 500,
        task: str = "reach",
    ):
        """
        Initialize manipulation environment.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (7D: 6D position + 1D gripper)
            max_episode_steps: Max steps per episode
            task: Task type ('reach', 'pick', 'push', 'stack')
        """
        config = EnvironmentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_episode_steps=max_episode_steps,
            action_bounds=(-1.0, 1.0),
        )
        
        super().__init__(config)
        
        self.task = task
        
        # Robot state
        self.ee_pos = np.array([0.0, 0.0, 0.5])
        self.ee_rot = np.eye(3)
        self.gripper_state = 0.0
        
        # Object states
        self.object_pos = np.array([0.3, 0.0, 0.0])
        self.object_rot = np.eye(3)
        
        # Target
        self.target_pos = np.array([0.5, 0.5, 0.0])
        
        # Metrics
        self.task_progress = 0.0
        self.grasped = False
        
        logger.info(f"Initialized ManipulationEnvironment (task={task})")
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.step_count = 0
        self.ee_pos = np.array([0.0, 0.0, 0.5])
        self.ee_rot = np.eye(3)
        self.gripper_state = 0.0
        
        self.object_pos = np.random.uniform([-0.3, -0.3, 0.0], [0.3, 0.3, 0.1])
        self.object_rot = np.eye(3)
        
        self.target_pos = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 0.1])
        
        self.task_progress = 0.0
        self.grasped = False
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        action = self._clip_action(action)
        self.step_count += 1
        
        # Update end-effector position
        delta_pos = action[:3] * 0.01
        self.ee_pos += delta_pos
        
        # Clamp position
        self.ee_pos = np.clip(self.ee_pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])
        
        # Update gripper
        self.gripper_state = action[6]
        
        # Check grasp
        distance_to_object = np.linalg.norm(self.ee_pos - self.object_pos)
        if distance_to_object < 0.05 and self.gripper_state > 0.5:
            self.grasped = True
        
        # Update object if grasped
        if self.grasped:
            self.object_pos = self.ee_pos + np.array([0, 0, -0.05])
        
        # Compute reward based on task
        reward = self._compute_reward_for_task()
        
        # Check done
        done = self.step_count >= self.config.max_episode_steps
        
        if self.task == "reach":
            if distance_to_object < 0.05:
                done = True
        
        elif self.task == "pick":
            if self.grasped and self.ee_pos[2] > 0.3:
                done = True
        
        elif self.task == "push":
            object_to_target = np.linalg.norm(self.object_pos - self.target_pos)
            if object_to_target < 0.1:
                done = True
        
        info = {
            'task_progress': self.task_progress,
            'grasped': self.grasped,
            'ee_pos': self.ee_pos.copy(),
            'object_pos': self.object_pos.copy(),
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.config.state_dim)
        
        # End-effector state
        obs[:3] = self.ee_pos
        obs[3:9] = self.ee_rot.flatten()[:6]
        obs[9] = self.gripper_state
        
        # Object state
        obs[10:13] = self.object_pos
        obs[13:16] = self.object_rot.flatten()[:3]
        
        # Target
        obs[16:19] = self.target_pos
        
        # Relative positions
        obs[19:22] = self.object_pos - self.ee_pos
        obs[22:25] = self.target_pos - self.object_pos
        
        # Distance metrics
        obs[25] = np.linalg.norm(self.ee_pos - self.object_pos)
        obs[26] = np.linalg.norm(self.object_pos - self.target_pos)
        
        return obs
    
    def _compute_reward_for_task(self) -> float:
        """Compute reward based on task."""
        reward = 0.0
        
        if self.task == "reach":
            distance = np.linalg.norm(self.ee_pos - self.object_pos)
            reward = -distance
            if distance < 0.05:
                reward += 10.0
        
        elif self.task == "pick":
            if not self.grasped:
                distance = np.linalg.norm(self.ee_pos - self.object_pos)
                reward = -distance * 2.0
            else:
                height_reward = self.ee_pos[2] - self.object_pos[2]
                reward = height_reward + 5.0
                if self.ee_pos[2] > 0.3:
                    reward += 10.0
        
        elif self.task == "push":
            if self.grasped:
                distance = np.linalg.norm(self.object_pos - self.target_pos)
                reward = -distance
                if distance < 0.1:
                    reward += 10.0
            else:
                contact_distance = np.linalg.norm(self.ee_pos - self.object_pos)
                reward = -contact_distance * 0.1
        
        elif self.task == "stack":
            # Stacking two objects
            reward = -np.linalg.norm(self.ee_pos - self.object_pos)
            if self.grasped:
                reward += 5.0
        
        return float(reward)


class PickAndPlaceEnvironment(ManipulationEnvironment):
    """Pick and place task environment."""
    
    def __init__(self, state_dim: int = 30, action_dim: int = 7):
        """Initialize pick and place environment."""
        super().__init__(state_dim, action_dim, task="pick")
        self.placed = False
        
        logger.info("Initialized PickAndPlaceEnvironment")
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        super().reset()
        self.placed = False
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step."""
        obs, reward, done, info = super().step(action)
        
        # Check if placed at target
        if self.grasped and np.linalg.norm(self.object_pos - self.target_pos) < 0.05:
            self.placed = True
            done = True
            reward += 20.0
        
        info['placed'] = self.placed
        
        return obs, reward, done, info
