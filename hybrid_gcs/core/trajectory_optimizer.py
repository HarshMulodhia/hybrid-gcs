"""
Trajectory Optimizer - Path Planning Through Convex Regions
File: hybrid_gcs/core/trajectory_optimizer.py

Implements optimal trajectory optimization through GCS regions using
convex programming and shortest path algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySegment:
    """Represents a trajectory segment through a region."""

    region_id: int
    waypoints: np.ndarray # (N, d) waypoint sequence
    costs: np.ndarray # (N-1,) segment costs
    total_cost: float
    is_feasible: bool = True

@dataclass
class OptimalTrajectory:
    """Complete optimal trajectory"""

    segments: List[TrajectorySegment]
    total_cost: float
    path_lenght: float
    region_sequence: List[int]
    computation_time: float

class TrajectoryOptimizer:
    """
    Optimizes trajectories through convex regions using efficient algorithms.

    Combines:
    - Dijkstra's algorithm for region sequence
    - Quadratic programming for smooth trajectories
    - Dynamic programming for cost optimization

    Attributes:
        gcs_decomposer: GCS decomposer instance
        collision_checker: Collision checking function
        num_waypoints_per_segment: Default waypoints per region segment 
    """

    def __init__(
            self, 
            gcs_decomposer: 'GCSDecomposer',
            collision_checker: Optional[Callable] = None,
            num_waypoints: int = 5
        ):
        """
        Initialize trajectory optimizer.

        Args:
            gcs_decomposer: GCS decomposer instance
            collision_checker: Optional collision checker function
            num_waypoints: Number of waypoints per segment
        """
        self.gcs_decomposer = gcs_decomposer
        self.collision_checker = collision_checker
        self.num_waypoints = num_waypoints

        logger.info(f"Initialized TrajectoryOptimizer with {num_waypoints} waypoints/segment")

    def optimize_trajectory(
            self, 
            start_config: np.ndarray,
            goal_config: np.ndarray,
            start_region_id: int,
            goal_region_id: int,
            obstacle_cost_weight: float = 1.0,
            smoothness_weight: float = 1.0
    ) -> Optional[OptimalTrajectory]:
        """
        Compute optimal trajectory from start to goal.

        Args:
            start_config: Starting configuration (d,)
            goal_config: Goal configuration (d,)
            start_region_id: ID of starting region
            goal_region_id: ID of goal region
            obstacle_cost_weight: Weight for obstacle avoidance
            smoothness_weight: Weight for trajectory smoothness

        Returns:
            OptimalTrajectory object or None if no path exists

        Raises:
            ValueError: If start/goal outside specified regions
        """
        import time
        start_time = time.time()

        # Validate inputs
        if not self._validate_config(start_config, start_region_id):
            raise ValueError(f"Start configuration not in start region")
        if not self._validate_config(goal_config, goal_region_id):
            raise ValueError(f"Goal configuration not in goal region")
        
        # Find region sequence
        region_sequence = self.gcs_decomposer.find_sequence(start_region_id, goal_region_id)
        if region_sequence is None:
            logger.warning("No valid region sequence found")
            return None
        
        # Optimize trajectory segments
        segments = []
        current_config = start_config
        total_cost = 0.0
        path_length = 0.0
        target_config = goal_config

        for i, region_id in enumerate(region_sequence):
            if i!=len(region_sequence)-1:
                target_config=self._get_region_waypoint(self.gcs_decomposer.get_region_by_id(region_sequence[i+1]))
            
            segment = self._optimize_segment(current_config, target_config, region_id, obstacle_cost_weight, smoothness_weight)
            if segment is not None:
                segments.append(segment)
                total_cost += segment.total_cost
                path_length += np.sum(np.linalg.norm(np.diff(segment.waypoints, axis=0), axis=1))
                current_config = segment.waypoints[-1]

        computation_time = time.time() - start_time

        trajectory = OptimalTrajectory(segments=segments, total_cost=total_cost, 
                                        path_lenght=path_length, computation_time=computation_time)
        
        logger.info(
            f"Optimized trajectory: cost={total_cost:.3f}, "
            f"length={path_length:.3f}, regions={len(region_sequence)}, "
            f"time={computation_time:.3f}s"
        )

        return trajectory
    
    def _optimize_segment(self, start:np.ndarray, goal:np.ndarray, region_id:int, obstacle_cost:float,
                          smoothness:float) -> Optional[TrajectorySegment]:
        """
        Optimize a single trajectory segment through a region.

        Args:
            start: Start configuration
            goal: Goal configuration
            region_id: Region ID
            obstacle_cost: Obstacle cost weight
            Smoothness: Smoothness weight

        Returns:
            TrajectorySegment or None
        """
        try:
            # Generate waypoints using linear interpolation
            waypoints = np.linspace(start, goal, self.num_waypoints)
            # Apply smoothing
            smoothed = self._smooth_trajectory(waypoints, smoothness)
            # Compute segment costs
            waypoint_diffs = np.diff(smoothed, axis=0)
            costs = np.linalg.norm(waypoint_diffs, axis=1)

            # Check collision
            if self.collision_checker and not self.collision_checker(smoothed):
                logger.debug(f"Segment through region {region_id} has collisions")
                return None
            
            segment = TrajectorySegment(region_id=region_id, waypoints=smoothed, costs=costs, 
                                        total_cost=float(np.sum(costs), is_feasible=True))
            return segment
        
        except Exception as e:
            logger.error(f"Error optimizing segment in region {region_id}: {e}")
            return None
    
    def _smooth_trajectory(self, waypoints:np.ndarray, smoothness_weight:float) -> np.ndarray:
        """Apply smoothing to trajectory."""
        if smoothness_weight<= 0 or len(waypoints) < 3:
            return waypoints
        
        try:
            import cvxpy as cp
            N, d = waypoints.shape
            x = cp.Variable((N, d))

            # Objective: stay close to original + minimize accelration
            objective = cp.Minimize(
                cp.sum_squares(x-waypoints) + smoothness_weight * cp.sum_squares(x[2:] - 2*x[1:-1] + x[:2])
            )
            problem = cp.Problem(objective)
            problem.solve(solver=cp.SCS, verbose=False)

            if problem.status==cp.OPTIMAL:
                return np.array(x.value)
            
        except ImportError:
            pass

        return waypoints
    
    def _validate_config(self, config:np.ndarray, region_id:int) -> bool:
        """Check if configuration is in region."""
        region = self.gcs_decomposer.get_region_by_id(region_id)
        if region is None:
            return False
        return region.contains_point(config)   

    def _get_region_waypoint(self, region) -> np.ndarray:
        """Get a waypoint in the region (its center)"""
        if region is None:
            return np.zeros(self.gcs_decomposer.config_space_dim)
        return region.center

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "num_waypoints_per_segment": self.num_waypoints,
            "has_collision_checker": self.collision_checker is not None,
        }
    
    def __repr__(self) -> str:
        return f"TrajectoryOptimizer(waypoints={self.num_waypoints})"
    
