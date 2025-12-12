"""
Performance Metrics Computation
File: hybrid_gcs/evaluation/metrics.py

Computes various performance metrics for trajectories and policies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    success_rate: float = 0.0
    path_length: float = 0.0
    execution_time: float = 0.0
    smoothness: float = 0.0
    energy_cost: float = 0.0
    collision_count: int = 0
    safety_margin_min: float = 0.0
    effectiveness: float = 0.0  # Combined metric
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'success_rate': self.success_rate,
            'path_length': self.path_length,
            'execution_time': self.execution_time,
            'smoothness': self.smoothness,
            'energy_cost': self.energy_cost,
            'collision_count': self.collision_count,
            'safety_margin_min': self.safety_margin_min,
            'effectiveness': self.effectiveness,
        }


class MetricsComputer:
    """Computes trajectory and policy performance metrics."""
    
    def __init__(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        config_space_dim: int = 6,
    ):
        """
        Initialize metrics computer.
        
        Args:
            start_config: Start configuration
            goal_config: Goal configuration
            config_space_dim: Dimension of configuration space
        """
        self.start_config = start_config
        self.goal_config = goal_config
        self.config_space_dim = config_space_dim
    
    def compute_trajectory_metrics(
        self,
        trajectory: np.ndarray,  # (N, dim)
        execution_time: float,
        collision_count: int = 0,
    ) -> PerformanceMetrics:
        """
        Compute metrics for a trajectory.
        
        Args:
            trajectory: (N, dim) trajectory waypoints
            execution_time: Total execution time
            collision_count: Number of collisions
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        # Path length
        metrics.path_length = self._compute_path_length(trajectory)
        
        # Execution time
        metrics.execution_time = execution_time
        
        # Success (if close to goal)
        final_dist = np.linalg.norm(trajectory[-1] - self.goal_config)
        metrics.success_rate = float(final_dist < 0.1)
        
        # Smoothness
        metrics.smoothness = self._compute_smoothness(trajectory)
        
        # Energy cost
        metrics.energy_cost = self._compute_energy_cost(trajectory)
        
        # Safety
        metrics.collision_count = collision_count
        metrics.safety_margin_min = 1.0 if collision_count == 0 else 0.0
        
        # Effectiveness (combined normalized metric)
        metrics.effectiveness = self._compute_effectiveness(metrics)
        
        return metrics
    
    def _compute_path_length(self, trajectory: np.ndarray) -> float:
        """Compute total path length."""
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))
    
    def _compute_smoothness(self, trajectory: np.ndarray) -> float:
        """
        Compute trajectory smoothness.
        
        Lower is smoother.
        """
        if len(trajectory) < 3:
            return 0.0
        
        # Compute second derivatives (acceleration)
        accelerations = np.diff(trajectory, n=2, axis=0)
        smoothness = float(np.mean(np.linalg.norm(accelerations, axis=1)))
        
        return smoothness
    
    def _compute_energy_cost(self, trajectory: np.ndarray) -> float:
        """
        Compute energy cost (integral of squared velocities).
        """
        if len(trajectory) < 2:
            return 0.0
        
        velocities = np.diff(trajectory, axis=0)
        velocity_norms = np.linalg.norm(velocities, axis=1)
        energy = float(np.sum(velocity_norms ** 2))
        
        return energy
    
    def _compute_effectiveness(self, metrics: PerformanceMetrics) -> float:
        """
        Compute combined effectiveness metric.
        
        Normalized to [0, 1] where 1 is best.
        """
        # Normalize each component
        success = metrics.success_rate
        
        # Lower path length is better
        path_length_norm = max(0, 1 - metrics.path_length / 100)
        
        # Lower smoothness (acceleration) is better
        smoothness_norm = max(0, 1 - metrics.smoothness / 10)
        
        # Lower energy is better
        energy_norm = max(0, 1 - metrics.energy_cost / 1000)
        
        # Safety
        safety = metrics.safety_margin_min
        
        # Combined
        effectiveness = (
            0.3 * success +
            0.2 * path_length_norm +
            0.2 * smoothness_norm +
            0.1 * energy_norm +
            0.2 * safety
        )
        
        return float(np.clip(effectiveness, 0, 1))
    
    def compare_trajectories(
        self,
        trajectory1: np.ndarray,
        trajectory2: np.ndarray,
        execution_times: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """
        Compare two trajectories.
        
        Args:
            trajectory1: First trajectory
            trajectory2: Second trajectory
            execution_times: Optional execution times (t1, t2)
            
        Returns:
            Comparison metrics
        """
        t1_time = execution_times[0] if execution_times else 1.0
        t2_time = execution_times[1] if execution_times else 1.0
        
        metrics1 = self.compute_trajectory_metrics(trajectory1, t1_time)
        metrics2 = self.compute_trajectory_metrics(trajectory2, t2_time)
        
        return {
            'path_length_ratio': metrics1.path_length / (metrics2.path_length + 1e-6),
            'time_ratio': t1_time / (t2_time + 1e-6),
            'smoothness_ratio': metrics1.smoothness / (metrics2.smoothness + 1e-6),
            'energy_ratio': metrics1.energy_cost / (metrics2.energy_cost + 1e-6),
            'effectiveness_diff': metrics1.effectiveness - metrics2.effectiveness,
        }


class MetricsAggregator:
    """Aggregates metrics across multiple trajectories."""
    
    def __init__(self):
        """Initialize aggregator."""
        self.metrics_list: List[PerformanceMetrics] = []
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add metrics."""
        self.metrics_list.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_list:
            return {}
        
        summary = {}
        
        for key in ['success_rate', 'path_length', 'execution_time', 
                    'smoothness', 'energy_cost', 'effectiveness']:
            values = [getattr(m, key) for m in self.metrics_list]
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_min"] = np.min(values)
            summary[f"{key}_max"] = np.max(values)
        
        # Collision statistics
        collisions = [m.collision_count for m in self.metrics_list]
        summary['collision_total'] = sum(collisions)
        summary['collision_rate'] = sum(c > 0 for c in collisions) / len(collisions)
        
        return summary
    
    def clear(self) -> None:
        """Clear metrics."""
        self.metrics_list = []
