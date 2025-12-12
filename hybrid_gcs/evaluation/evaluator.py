"""
Evaluation Framework
File: hybrid_gcs/evaluation/evaluator.py

Main evaluation engine for testing policies and trajectories.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    num_episodes: int = 100
    max_steps_per_episode: int = 500
    render: bool = False
    seed: Optional[int] = None
    
    # Metrics to compute
    compute_success_rate: bool = True
    compute_path_length: bool = True
    compute_time_efficiency: bool = True
    compute_smoothness: bool = True
    compute_energy: bool = True
    compute_safety: bool = True
    
    # Thresholds
    success_threshold: float = 0.1
    collision_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)


class Evaluator:
    """
    Main evaluation class for policies.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        env,
        policy: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            env: Environment for evaluation
            policy: Policy network to evaluate
        """
        self.config = config
        self.env = env
        self.policy = policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if policy is not None:
            self.policy.to(self.device)
        
        logger.info(f"Initialized Evaluator with {config.num_episodes} episodes")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation for {self.config.num_episodes} episodes")
        
        episode_results = []
        
        for episode in range(self.config.num_episodes):
            result = self._run_episode(episode)
            episode_results.append(result)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Completed {episode + 1}/{self.config.num_episodes} episodes")
        
        # Aggregate metrics
        metrics = self._aggregate_metrics(episode_results)
        
        logger.info(f"Evaluation complete. Summary: {metrics}")
        return metrics
    
    def _run_episode(self, episode_id: int) -> Dict[str, Any]:
        """Run single evaluation episode."""
        obs, _ = self.env.reset()
        
        episode_data = {
            'episode_id': episode_id,
            'states': [obs],
            'actions': [],
            'rewards': [],
            'dones': [],
            'positions': [obs[:3]],  # Assuming first 3 dims are position
            'velocities': [],
            'total_reward': 0,
            'path_length': 0,
            'success': False,
            'collisions': 0,
        }
        
        for step in range(self.config.max_steps_per_episode):
            # Get action from policy
            if self.policy is not None:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _ = self.policy.sample_action(obs_tensor, deterministic=True)
                    action = action.cpu().numpy().squeeze()
            else:
                action = self.env.action_space.sample()
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done or truncated)
            episode_data['states'].append(obs)
            episode_data['total_reward'] += reward
            
            # Compute path length
            if len(episode_data['positions']) > 0:
                prev_pos = episode_data['positions'][-1]
                curr_pos = obs[:3]
                segment_length = np.linalg.norm(curr_pos - prev_pos)
                episode_data['path_length'] += segment_length
                episode_data['positions'].append(curr_pos)
            
            # Check for success
            if 'is_success' in info and info['is_success']:
                episode_data['success'] = True
                logger.debug(f"Episode {episode_id}: Success at step {step}")
            
            # Check for collisions
            if 'collision' in info and info['collision']:
                episode_data['collisions'] += 1
            
            if done or truncated:
                break
        
        return episode_data
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from episodes."""
        metrics = {}
        
        # Success rate
        if self.config.compute_success_rate:
            successes = sum(1 for r in results if r['success'])
            metrics['success_rate'] = successes / len(results)
            logger.info(f"Success rate: {metrics['success_rate']:.2%}")
        
        # Path length statistics
        if self.config.compute_path_length:
            path_lengths = [r['path_length'] for r in results]
            metrics['path_length_mean'] = np.mean(path_lengths)
            metrics['path_length_std'] = np.std(path_lengths)
            metrics['path_length_min'] = np.min(path_lengths)
            metrics['path_length_max'] = np.max(path_lengths)
        
        # Reward statistics
        rewards = [r['total_reward'] for r in results]
        metrics['reward_mean'] = np.mean(rewards)
        metrics['reward_std'] = np.std(rewards)
        metrics['reward_min'] = np.min(rewards)
        metrics['reward_max'] = np.max(rewards)
        
        # Collision statistics
        if self.config.compute_safety:
            collisions = [r['collisions'] for r in results]
            metrics['collision_rate'] = sum(c > 0 for c in collisions) / len(results)
            metrics['collisions_mean'] = np.mean(collisions)
        
        # Episode length statistics
        episode_lengths = [
            len(r['rewards']) for r in results
        ]
        metrics['episode_length_mean'] = np.mean(episode_lengths)
        metrics['episode_length_std'] = np.std(episode_lengths)
        
        return metrics
