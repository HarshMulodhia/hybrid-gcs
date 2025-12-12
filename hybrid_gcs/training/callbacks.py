"""
Custom Training Callbacks
File: hybrid_gcs/training/callbacks.py

Callback system for monitoring and controlling training.
"""

import logging
from typing import Dict, Optional, Any, List
import torch
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class for training hooks."""
    
    def on_train_begin(self, trainer: 'OptimizedTrainer') -> None:
        """Called at training start."""
        pass
    
    def on_train_end(self, trainer: 'OptimizedTrainer') -> None:
        """Called at training end."""
        pass
    
    def on_step(self, trainer: 'OptimizedTrainer') -> None:
        """Called after each step."""
        pass
    
    def on_episode_end(self, trainer: 'OptimizedTrainer', reward: float) -> None:
        """Called after each episode."""
        pass
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Called during logging."""
        pass
    
    def on_checkpoint(self, trainer: 'OptimizedTrainer') -> None:
        """Called during checkpoint."""
        pass


class CheckpointCallback(Callback):
    """
    Checkpoints model based on performance.
    
    Saves best models and periodic checkpoints.
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_interval: int = 100_000,
        keep_best: int = 3,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            save_interval: Steps between checkpoints
            keep_best: Number of best checkpoints to keep
        """
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.keep_best = keep_best
        
        self.best_rewards = []
        self.checkpoint_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
    
    def on_checkpoint(self, trainer: 'OptimizedTrainer') -> None:
        """Save checkpoint when requested."""
        path = trainer.save_checkpoint()
        self.checkpoint_count += 1
        logger.info(f"Checkpoint saved: {path}")
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Track best rewards."""
        if 'episode_reward_mean' in metrics:
            reward = metrics['episode_reward_mean']
            
            self.best_rewards.append(reward)
            self.best_rewards.sort(reverse=True)
            self.best_rewards = self.best_rewards[:self.keep_best]
            
            if reward == self.best_rewards[0]:
                self._save_best(trainer, reward)
    
    def _save_best(self, trainer: 'OptimizedTrainer', reward: float) -> None:
        """Save best model."""
        path = os.path.join(
            self.save_dir,
            f"best_model_{reward:.2f}.pt"
        )
        torch.save({
            'actor': trainer.actor.state_dict(),
            'critic': trainer.critic.state_dict(),
            'timestep': trainer.timestep,
            'reward': reward,
        }, path)
        logger.info(f"Best model saved to {path} (reward={reward:.2f})")


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on validation performance.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Episodes without improvement before stopping
            min_delta: Minimum improvement threshold
        """
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_reward = float('-inf')
        self.wait_count = 0
        self.should_stop = False
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Check for early stopping."""
        if 'episode_reward_mean' not in metrics:
            return
        
        reward = metrics['episode_reward_mean']
        
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.wait_count = 0
            logger.info(f"New best reward: {reward:.2f}")
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                self.should_stop = True
                logger.warning(
                    f"Early stopping triggered after "
                    f"{self.wait_count} episodes without improvement"
                )


class LoggingCallback(Callback):
    """
    Advanced logging and monitoring.
    """
    
    def __init__(
        self,
        log_interval: int = 1000,
        log_file: Optional[str] = None,
    ):
        """
        Initialize logging callback.
        
        Args:
            log_interval: Logging interval in steps
            log_file: Optional file to log to
        """
        self.log_interval = log_interval
        self.log_file = log_file
        
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0
        
        if log_file:
            os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    
    def on_step(self, trainer: 'OptimizedTrainer') -> None:
        """Log step information."""
        self.step_count += 1
    
    def on_episode_end(self, trainer: 'OptimizedTrainer', reward: float) -> None:
        """Log episode information."""
        self.episode_count += 1
        self.total_reward += reward
        
        if self.episode_count % self.log_interval == 0:
            avg_reward = self.total_reward / self.log_interval
            msg = (
                f"Episode {self.episode_count}: "
                f"avg_reward={avg_reward:.2f}, "
                f"steps={self.step_count}"
            )
            logger.info(msg)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{datetime.now().isoformat()}: {msg}\n")
            
            self.total_reward = 0
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Log metrics."""
        logger.debug(f"Metrics: {metrics}")


class LearningRateScheduleCallback(Callback):
    """
    Monitors and logs learning rate schedule.
    """
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Log learning rate."""
        lr = trainer._get_learning_rate()
        metrics['learning_rate'] = lr
        logger.debug(f"Learning rate: {lr:.6f}")


class GradientMonitorCallback(Callback):
    """
    Monitors gradient statistics.
    """
    
    def __init__(self, log_interval: int = 1000):
        """Initialize gradient monitor."""
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Log gradient statistics."""
        self.step_count += 1
        
        if self.step_count % self.log_interval != 0:
            return
        
        # Monitor actor gradients
        actor_grads = [
            p.grad.norm().item()
            for p in trainer.actor.parameters()
            if p.grad is not None
        ]
        if actor_grads:
            metrics['actor_grad_mean'] = sum(actor_grads) / len(actor_grads)
            metrics['actor_grad_max'] = max(actor_grads)
        
        # Monitor critic gradients
        critic_grads = [
            p.grad.norm().item()
            for p in trainer.critic.parameters()
            if p.grad is not None
        ]
        if critic_grads:
            metrics['critic_grad_mean'] = sum(critic_grads) / len(critic_grads)
            metrics['critic_grad_max'] = max(critic_grads)


class MetricsCallback(Callback):
    """
    Tracks and reports training metrics.
    """
    
    def __init__(self):
        """Initialize metrics callback."""
        self.metrics_history = []
    
    def on_log(self, trainer: 'OptimizedTrainer', metrics: Dict) -> None:
        """Record metrics."""
        metrics['timestep'] = trainer.timestep
        metrics['episode'] = trainer.episode
        self.metrics_history.append(metrics.copy())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        for key in self.metrics_history[0].keys():
            values = [m.get(key, 0) for m in self.metrics_history]
            if all(isinstance(v, (int, float)) for v in values):
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)
        
        return summary
