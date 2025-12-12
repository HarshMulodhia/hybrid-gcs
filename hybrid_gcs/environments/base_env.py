"""
Base Environment Class
File: hybrid_gcs/environments/base_env.py

Defines the base environment interface for RL tasks.
"""

import logging
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for environment."""
    
    state_dim: int
    action_dim: int
    max_episode_steps: int = 1000
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    observation_bounds: Optional[Tuple[float, float]] = None
    reward_scale: float = 1.0
    seed: Optional[int] = None


class BaseEnvironment(ABC):
    """
    Base class for all environments.
    
    Defines the interface for RL environments.
    """
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(state_dim={config.state_dim}, action_dim={config.action_dim})"
        )
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, info)
        """
        pass
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array')
        """
        logger.debug(f"Rendering in {mode} mode")
    
    def close(self) -> None:
        """Close environment and cleanup resources."""
        logger.debug("Closed environment")
    
    def seed(self, seed: int) -> None:
        """
        Set random seed.
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        self.config.seed = seed
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.
        
        Args:
            action: Raw action
            
        Returns:
            Clipped action
        """
        lower, upper = self.config.action_bounds
        return np.clip(action, lower, upper)
    
    def _clip_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Clip observation to valid bounds.
        
        Args:
            observation: Raw observation
            
        Returns:
            Clipped observation
        """
        if self.config.observation_bounds is not None:
            lower, upper = self.config.observation_bounds
            return np.clip(observation, lower, upper)
        return observation
    
    def get_state(self) -> Optional[np.ndarray]:
        """Get current state."""
        return self.state
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.config.state_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.config.action_dim
    
    def is_done(self) -> bool:
        """Check if episode is done."""
        return self.step_count >= self.config.max_episode_steps
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.config.state_dim}, "
            f"action_dim={self.config.action_dim})"
        )
