"""
YCB Grasping Environment
File: hybrid_gcs/environments/ycb_grasp_env.py

RL environment for YCB object grasping.
"""

import logging
from typing import Dict, Tuple, Any
import numpy as np
from hybrid_gcs.environments.base_env import BaseEnvironment, EnvironmentConfig

logger = logging.getLogger(__name__)


class YCBGraspEnvironment(BaseEnvironment):
    """
    YCB grasping environment.
    
    Agent learns to grasp objects from the YCB dataset.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 6,
        max_episode_steps: int = 500,
        object_id: int = 0,
        num_objects: int = 10,
    ):
        """
        Initialize YCB grasp environment.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (6D: position + rotation)
            max_episode_steps: Max steps per episode
            object_id: Initial object ID
            num_objects: Number of objects in dataset
        """
        config = EnvironmentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_episode_steps=max_episode_steps,
            action_bounds=(-1.0, 1.0),
        )
        
        super().__init__(config)
        
        self.object_id = object_id
        self.num_objects = num_objects
        self.gripper_pos = np.zeros(3)
        self.gripper_rot = np.eye(3)
        self.grasp_success = False
        
        logger.info(f"Initialized YCBGraspEnvironment with {num_objects} objects")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment.
        
        Returns:
            Initial observation
        """
        self.step_count = 0
        self.gripper_pos = np.random.uniform(-0.5, 0.5, 3)
        self.gripper_rot = np.eye(3)
        self.grasp_success = False
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 6D action (dx, dy, dz, rx, ry, rz)
            
        Returns:
            (observation, reward, done, info)
        """
        action = self._clip_action(action)
        self.step_count += 1
        
        # Update gripper position
        delta_pos = action[:3] * 0.01
        self.gripper_pos += delta_pos
        
        # Check boundaries
        self.gripper_pos = np.clip(self.gripper_pos, -1.0, 1.0)
        
        # Compute reward
        reward = self._compute_reward()
        
        done = self.step_count >= self.config.max_episode_steps or self.grasp_success
        
        info = {
            'grasp_success': self.grasp_success,
            'gripper_pos': self.gripper_pos.copy(),
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.config.state_dim)
        
        # Gripper position and rotation
        obs[:3] = self.gripper_pos
        obs[3:9] = self.gripper_rot.flatten()[:6]
        
        # Random object features (simulation)
        obs[9:] = np.random.randn(self.config.state_dim - 9) * 0.1
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward."""
        # Distance-based reward
        object_pos = np.array([0.0, 0.0, 0.0])  # Simulated object position
        distance = np.linalg.norm(self.gripper_pos - object_pos)
        
        reward = -distance
        
        # Bonus for successful grasp
        if distance < 0.05:
            self.grasp_success = True
            reward += 10.0
        
        return float(reward)
    
    def change_object(self, object_id: int) -> None:
        """
        Change object to grasp.
        
        Args:
            object_id: New object ID
        """
        if 0 <= object_id < self.num_objects:
            self.object_id = object_id
            logger.info(f"Changed object to ID {object_id}")


class DualArmGraspEnvironment(YCBGraspEnvironment):
    """Dual-arm grasping environment."""
    
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 12,
        max_episode_steps: int = 500,
    ):
        """Initialize dual-arm environment."""
        super().__init__(state_dim, action_dim, max_episode_steps)
        
        self.gripper_pos_left = np.zeros(3)
        self.gripper_pos_right = np.zeros(3)
        
        logger.info("Initialized DualArmGraspEnvironment")
    
    def reset(self) -> np.ndarray:
        """Reset dual-arm environment."""
        self.step_count = 0
        self.gripper_pos_left = np.random.uniform(-0.5, 0.0, 3)
        self.gripper_pos_right = np.random.uniform(0.0, 0.5, 3)
        self.grasp_success = False
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation from both arms."""
        obs = np.zeros(self.config.state_dim)
        
        obs[:3] = self.gripper_pos_left
        obs[3:6] = self.gripper_pos_right
        obs[6:] = np.random.randn(self.config.state_dim - 6) * 0.1
        
        return obs
