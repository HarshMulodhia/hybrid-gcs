"""
Reward Shaping Functions
File: hybrid_gcs/training/reward_shaping.py

Implements reward shaping strategies for improved learning.
"""

import logging
from typing import Dict, Callable, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class RewardShaper(ABC):
    """Base class for reward shaping."""
    
    @abstractmethod
    def compute_reward(self, **kwargs) -> float:
        """
        Compute shaped reward.
        
        Args:
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Shaped reward value
        """
        pass
    
    def __call__(self, **kwargs) -> float:
        """Make shaper callable."""
        return self.compute_reward(**kwargs)


class DistanceReward(RewardShaper):
    """
    Reward based on distance to goal.
    
    Encourages reaching the goal efficiently.
    """
    
    def __init__(
        self,
        goal_threshold: float = 0.1,
        max_reward: float = 1.0,
        distance_scale: float = 1.0,
    ):
        """
        Initialize distance reward.
        
        Args:
            goal_threshold: Distance threshold for success
            max_reward: Maximum reward at goal
            distance_scale: Scale factor for distance metric
        """
        self.goal_threshold = goal_threshold
        self.max_reward = max_reward
        self.distance_scale = distance_scale
    
    def compute_reward(
        self,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        previous_distance: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Compute distance-based reward.
        
        Args:
            current_position: Current position
            goal_position: Goal position
            previous_distance: Previous distance (for delta reward)
            
        Returns:
            Reward value
        """
        distance = np.linalg.norm(current_position - goal_position)
        
        # Success reward
        if distance < self.goal_threshold:
            return self.max_reward
        
        # Distance-based reward (inverse distance)
        distance_reward = self.max_reward / (
            1 + self.distance_scale * distance
        )
        
        # Progress reward (moving closer)
        progress_reward = 0.0
        if previous_distance is not None:
            if distance < previous_distance:
                progress_reward = 0.1 * (previous_distance - distance)
        
        return distance_reward + progress_reward


class SmoothPathReward(RewardShaper):
    """
    Reward smooth, efficient paths.
    
    Penalizes jerky movements and encourages smooth trajectories.
    """
    
    def __init__(
        self,
        smoothness_scale: float = 0.1,
        max_acceleration: float = 1.0,
    ):
        """
        Initialize smooth path reward.
        
        Args:
            smoothness_scale: Scale for smoothness penalty
            max_acceleration: Maximum allowed acceleration
        """
        self.smoothness_scale = smoothness_scale
        self.max_acceleration = max_acceleration
    
    def compute_reward(
        self,
        velocity: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute smoothness reward.
        
        Args:
            velocity: Current velocity
            previous_velocity: Previous velocity
            
        Returns:
            Smoothness reward
        """
        # Velocity norm penalty (smoother = lower velocity)
        velocity_cost = -0.01 * np.linalg.norm(velocity)
        
        # Acceleration penalty
        acceleration_penalty = 0.0
        if previous_velocity is not None:
            acceleration = np.linalg.norm(velocity - previous_velocity)
            if acceleration > self.max_acceleration:
                acceleration_penalty = (
                    -self.smoothness_scale * (acceleration - self.max_acceleration)
                )
        
        return velocity_cost + acceleration_penalty


class EfficiencyReward(RewardShaper):
    """
    Reward efficient path planning.
    
    Combines distance, smoothness, and energy efficiency.
    """
    
    def __init__(
        self,
        distance_weight: float = 0.5,
        smoothness_weight: float = 0.3,
        time_weight: float = 0.2,
    ):
        """
        Initialize efficiency reward.
        
        Args:
            distance_weight: Weight for distance-to-goal
            smoothness_weight: Weight for path smoothness
            time_weight: Weight for time efficiency
        """
        self.distance_weight = distance_weight
        self.smoothness_weight = smoothness_weight
        self.time_weight = time_weight
        
        self.distance_shaper = DistanceReward()
        self.smoothness_shaper = SmoothPathReward()
    
    def compute_reward(
        self,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        velocity: np.ndarray,
        previous_velocity: Optional[np.ndarray] = None,
        steps_taken: int = 0,
        max_steps: int = 500,
        **kwargs,
    ) -> float:
        """
        Compute combined efficiency reward.
        
        Args:
            current_position: Current position
            goal_position: Goal position
            velocity: Current velocity
            previous_velocity: Previous velocity
            steps_taken: Number of steps taken
            max_steps: Maximum allowed steps
            
        Returns:
            Combined efficiency reward
        """
        # Distance reward
        distance_reward = self.distance_shaper.compute_reward(
            current_position=current_position,
            goal_position=goal_position,
        )
        
        # Smoothness reward
        smoothness_reward = self.smoothness_shaper.compute_reward(
            velocity=velocity,
            previous_velocity=previous_velocity,
        )
        
        # Time efficiency reward (penalize time)
        time_penalty = 0.0
        if max_steps > 0:
            time_efficiency = 1 - (steps_taken / max_steps)
            time_penalty = self.time_weight * time_efficiency
        
        # Combine weighted rewards
        total_reward = (
            self.distance_weight * distance_reward +
            self.smoothness_weight * smoothness_reward +
            self.time_weight * time_penalty
        )
        
        return total_reward


class CompositeReward(RewardShaper):
    """
    Combines multiple reward functions.
    """
    
    def __init__(
        self,
        shapers: Dict[str, Tuple[RewardShaper, float]],
    ):
        """
        Initialize composite reward.
        
        Args:
            shapers: Dict of {name: (shaper, weight)}
        """
        self.shapers = shapers
        
        # Normalize weights
        total_weight = sum(w for _, w in shapers.values())
        self.weights = {
            name: w / total_weight
            for name, (_, w) in shapers.items()
        }
    
    def compute_reward(self, **kwargs) -> float:
        """
        Compute composite reward.
        
        Args:
            **kwargs: Arguments for all shapers
            
        Returns:
            Weighted sum of rewards
        """
        total_reward = 0.0
        
        for name, (shaper, _) in self.shapers.items():
            reward = shaper.compute_reward(**kwargs)
            weighted_reward = self.weights[name] * reward
            total_reward += weighted_reward
        
        return total_reward


class SuccessReward(RewardShaper):
    """
    Simple binary success reward.
    """
    
    def __init__(
        self,
        success_reward: float = 100.0,
        failure_penalty: float = -1.0,
    ):
        """
        Initialize success reward.
        
        Args:
            success_reward: Reward for success
            failure_penalty: Penalty for failure
        """
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
    
    def compute_reward(
        self,
        is_success: bool,
        **kwargs,
    ) -> float:
        """
        Compute success reward.
        
        Args:
            is_success: Whether task was successful
            
        Returns:
            Reward or penalty
        """
        return self.success_reward if is_success else self.failure_penalty


class CollisionPenalty(RewardShaper):
    """
    Penalizes collisions.
    """
    
    def __init__(
        self,
        collision_penalty: float = -50.0,
        penalty_scale: float = 1.0,
    ):
        """
        Initialize collision penalty.
        
        Args:
            collision_penalty: Penalty per collision
            penalty_scale: Scale factor for severity
        """
        self.collision_penalty = collision_penalty
        self.penalty_scale = penalty_scale
    
    def compute_reward(
        self,
        num_collisions: int = 0,
        collision_severity: float = 1.0,
        **kwargs,
    ) -> float:
        """
        Compute collision penalty.
        
        Args:
            num_collisions: Number of collisions
            collision_severity: Severity scale (0-1)
            
        Returns:
            Penalty value
        """
        if num_collisions == 0:
            return 0.0
        
        return (
            self.collision_penalty *
            num_collisions *
            (self.penalty_scale * collision_severity)
        )
