"""
Metrics Utility Module
File: hybrid_gcs/utils/metrics.py

Computes trajectory and performance metrics.
"""

import logging
from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    Compute total trajectory length.
    
    Args:
        trajectory: Trajectory array (shape: (N, D))
        
    Returns:
        Total length
    """
    if len(trajectory) < 2:
        return 0.0
    
    differences = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    total_length = float(np.sum(distances))
    
    logger.debug(f"Computed trajectory length: {total_length:.4f}")
    
    return total_length


def compute_smoothness(trajectory: np.ndarray) -> float:
    """
    Compute trajectory smoothness (jerk).
    
    Args:
        trajectory: Trajectory array (shape: (N, D))
        
    Returns:
        Smoothness measure (lower is better)
    """
    if len(trajectory) < 3:
        return 0.0
    
    # First derivative (velocity)
    velocity = np.diff(trajectory, axis=0)
    
    # Second derivative (acceleration)
    acceleration = np.diff(velocity, axis=0)
    
    # Third derivative (jerk)
    jerk = np.diff(acceleration, axis=0)
    
    # Smoothness = integral of squared jerk
    jerk_norms = np.linalg.norm(jerk, axis=1)
    smoothness = float(np.sum(jerk_norms ** 2))
    
    logger.debug(f"Computed smoothness: {smoothness:.4f}")
    
    return smoothness


def compute_energy(trajectory: np.ndarray, mass: float = 1.0) -> float:
    """
    Compute trajectory energy (kinetic + potential).
    
    Args:
        trajectory: Trajectory array (shape: (N, D))
        mass: Mass of the object
        
    Returns:
        Total energy
    """
    if len(trajectory) < 2:
        return 0.0
    
    # Velocity
    velocity = np.diff(trajectory, axis=0)
    velocity_norms = np.linalg.norm(velocity, axis=1)
    
    # Kinetic energy (0.5 * m * v^2)
    kinetic = 0.5 * mass * np.sum(velocity_norms ** 2)
    
    # Potential energy (m * g * z, assuming z is height)
    if trajectory.shape[1] >= 3:
        z = trajectory[:, 2]
        potential = mass * 9.81 * np.mean(z)
    else:
        potential = 0.0
    
    total_energy = float(kinetic + potential)
    
    logger.debug(f"Computed energy: kinetic={kinetic:.4f}, potential={potential:.4f}")
    
    return total_energy


def compute_clearance(
    trajectory: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
) -> float:
    """
    Compute minimum clearance from trajectory to obstacles.
    
    Args:
        trajectory: Trajectory array (shape: (N, D))
        obstacles: List of (center, radius) tuples
        
    Returns:
        Minimum clearance distance
    """
    if not obstacles or len(trajectory) == 0:
        return float('inf')
    
    min_clearance = float('inf')
    
    for point in trajectory:
        for center, radius in obstacles:
            distance = np.linalg.norm(point - center)
            clearance = distance - radius
            min_clearance = min(min_clearance, clearance)
    
    logger.debug(f"Computed clearance: {min_clearance:.4f}")
    
    return float(min_clearance)


def compute_curvature(trajectory: np.ndarray) -> float:
    """
    Compute mean curvature of trajectory.
    
    Args:
        trajectory: Trajectory array (shape: (N, D))
        
    Returns:
        Mean curvature
    """
    if len(trajectory) < 3:
        return 0.0
    
    # First derivative
    first_deriv = np.diff(trajectory, axis=0)
    
    # Second derivative
    second_deriv = np.diff(first_deriv, axis=0)
    
    # Curvature = ||first × second|| / ||first||^3
    curvatures = []
    
    for i in range(len(first_deriv) - 1):
        v = first_deriv[i]
        a = second_deriv[i]
        
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-6:
            cross = np.linalg.norm(np.cross(v, a))
            curvature = cross / (v_norm ** 3 + 1e-6)
            curvatures.append(curvature)
    
    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0
    
    logger.debug(f"Computed mean curvature: {mean_curvature:.4f}")
    
    return mean_curvature


def compute_distance_to_goal(trajectory: np.ndarray, goal: np.ndarray) -> float:
    """
    Compute final distance to goal.
    
    Args:
        trajectory: Trajectory array
        goal: Goal position
        
    Returns:
        Distance to goal
    """
    if len(trajectory) == 0:
        return float('inf')
    
    final_pos = trajectory[-1]
    distance = euclidean(final_pos, goal)
    
    logger.debug(f"Distance to goal: {distance:.4f}")
    
    return float(distance)


def compute_success_rate(
    trajectories: List[np.ndarray],
    goal: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Compute success rate for trajectories.
    
    Args:
        trajectories: List of trajectory arrays
        goal: Goal position
        threshold: Distance threshold for success
        
    Returns:
        Success rate (0-1)
    """
    if not trajectories:
        return 0.0
    
    successes = 0
    for trajectory in trajectories:
        distance = compute_distance_to_goal(trajectory, goal)
        if distance <= threshold:
            successes += 1
    
    success_rate = successes / len(trajectories)
    
    logger.debug(f"Success rate: {success_rate:.1%}")
    
    return success_rate


def compute_time_to_goal(
    trajectory: np.ndarray,
    goal: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Compute time steps to reach goal.
    
    Args:
        trajectory: Trajectory array
        goal: Goal position
        threshold: Distance threshold for goal
        
    Returns:
        Time steps (or trajectory length if goal not reached)
    """
    for i, point in enumerate(trajectory):
        distance = euclidean(point, goal)
        if distance <= threshold:
            return float(i)
    
    return float(len(trajectory))
