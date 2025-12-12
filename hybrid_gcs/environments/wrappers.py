"""
Environment Wrappers
File: hybrid_gcs/environments/wrappers.py

Utility wrappers for environments.
"""

import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
from hybrid_gcs.environments.base_env import BaseEnvironment

logger = logging.getLogger(__name__)


class EnvironmentWrapper:
    """Base class for environment wrappers."""
    
    def __init__(self, env: BaseEnvironment):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
        """
        self.env = env
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        return self.env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step."""
        return self.env.step(action)
    
    def render(self, mode: str = 'human') -> None:
        """Render environment."""
        self.env.render(mode)
    
    def close(self) -> None:
        """Close environment."""
        self.env.close()


class NormalizationWrapper(EnvironmentWrapper):
    """
    Normalizes observations and actions.
    """
    
    def __init__(
        self,
        env: BaseEnvironment,
        norm_obs: bool = True,
        norm_action: bool = False,
    ):
        """
        Initialize normalization wrapper.
        
        Args:
            env: Environment to wrap
            norm_obs: Whether to normalize observations
            norm_action: Whether to normalize actions
        """
        super().__init__(env)
        
        self.norm_obs = norm_obs
        self.norm_action = norm_action
        
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None
        
        logger.info(f"Initialized NormalizationWrapper (obs={norm_obs}, action={norm_action})")
    
    def reset(self) -> np.ndarray:
        """Reset and return normalized observation."""
        obs = self.env.reset()
        
        if self.norm_obs:
            if self.obs_mean is None:
                self.obs_mean = np.mean(obs)
                self.obs_std = np.std(obs) + 1e-8
            obs = (obs - self.obs_mean) / self.obs_std
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take step with optional action normalization."""
        if self.norm_action:
            action = action * 2.0  # Denormalize from [-1, 1]
        
        obs, reward, done, info = self.env.step(action)
        
        if self.norm_obs:
            obs = (obs - self.obs_mean) / self.obs_std
        
        return obs, reward, done, info


class RecordingWrapper(EnvironmentWrapper):
    """
    Records trajectories.
    """
    
    def __init__(self, env: BaseEnvironment):
        """
        Initialize recording wrapper.
        
        Args:
            env: Environment to wrap
        """
        super().__init__(env)
        
        self.episode_data = []
        self.current_episode = []
        
        logger.info("Initialized RecordingWrapper")
    
    def reset(self) -> np.ndarray:
        """Reset and start new episode."""
        if self.current_episode:
            self.episode_data.append(self.current_episode)
        
        self.current_episode = []
        obs = self.env.reset()
        
        self.current_episode.append({
            'observation': obs.copy(),
            'action': None,
            'reward': None,
            'done': False,
        })
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take step and record."""
        obs, reward, done, info = self.env.step(action)
        
        self.current_episode.append({
            'observation': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'done': done,
        })
        
        return obs, reward, done, info
    
    def get_episodes(self) -> list:
        """Get recorded episodes."""
        return self.episode_data.copy()
    
    def save_episode(self, filepath: str) -> None:
        """Save episode to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        data = []
        for ep in self.episode_data:
            ep_data = []
            for step in ep:
                ep_data.append({
                    'observation': step['observation'].tolist() if isinstance(step['observation'], np.ndarray) else step['observation'],
                    'action': step['action'].tolist() if isinstance(step['action'], np.ndarray) else step['action'],
                    'reward': float(step['reward']) if step['reward'] is not None else None,
                    'done': bool(step['done']),
                })
            data.append(ep_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved {len(self.episode_data)} episodes to {filepath}")


class TimeoutWrapper(EnvironmentWrapper):
    """
    Adds timeout limit to episodes.
    """
    
    def __init__(self, env: BaseEnvironment, max_steps: int = 1000):
        """
        Initialize timeout wrapper.
        
        Args:
            env: Environment to wrap
            max_steps: Maximum steps per episode
        """
        super().__init__(env)
        
        self.max_steps = max_steps
        self.step_count = 0
        
        logger.info(f"Initialized TimeoutWrapper (max_steps={max_steps})")
    
    def reset(self) -> np.ndarray:
        """Reset and reset step counter."""
        self.step_count = 0
        return self.env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take step with timeout check."""
        obs, reward, done, info = self.env.step(action)
        
        self.step_count += 1
        
        if self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
        
        return obs, reward, done, info


class StackingWrapper(EnvironmentWrapper):
    """
    Stacks observations.
    """
    
    def __init__(self, env: BaseEnvironment, num_stack: int = 4):
        """
        Initialize stacking wrapper.
        
        Args:
            env: Environment to wrap
            num_stack: Number of observations to stack
        """
        super().__init__(env)
        
        self.num_stack = num_stack
        self.stacked_obs = None
        
        logger.info(f"Initialized StackingWrapper (num_stack={num_stack})")
    
    def reset(self) -> np.ndarray:
        """Reset and initialize stacked observations."""
        obs = self.env.reset()
        
        # Initialize stack with current observation
        self.stacked_obs = [obs.copy() for _ in range(self.num_stack)]
        
        return self._get_stacked_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take step and update stacked observations."""
        obs, reward, done, info = self.env.step(action)
        
        # Shift stack and add new observation
        self.stacked_obs = self.stacked_obs[1:] + [obs.copy()]
        
        stacked = self._get_stacked_obs()
        
        return stacked, reward, done, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observation."""
        return np.concatenate(self.stacked_obs, axis=0)
