"""
Safe Corridor Planning Module
File: hybrid_gcs/integration/corridor_planning.py

Plans safe corridors for robot navigation.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafeCorridor:
    """Represents a safe corridor for navigation."""
    
    waypoints: np.ndarray  # (N, D) trajectory waypoints
    clearances: np.ndarray  # Clearance at each waypoint
    min_clearance: float
    corridor_length: float
    feasible: bool = True


class CorridorPlanner:
    """
    Plans safe corridors for navigation.
    
    Combines GCS regions with collision-free path planning.
    """
    
    def __init__(
        self,
        safety_margin: float = 0.1,
        max_corridor_iterations: int = 100,
    ):
        """
        Initialize corridor planner.
        
        Args:
            safety_margin: Minimum clearance from obstacles
            max_corridor_iterations: Maximum iterations for corridor optimization
        """
        self.safety_margin = safety_margin
        self.max_iterations = max_corridor_iterations
        self.corridors: List[SafeCorridor] = []
        
        logger.info(
            f"Initialized CorridorPlanner "
            f"(safety_margin={safety_margin}, max_iter={max_corridor_iterations})"
        )
    
    def plan_corridor(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
        gcs_regions: Optional[List] = None,
    ) -> Optional[SafeCorridor]:
        """
        Plan a safe corridor from start to goal.
        
        Args:
            start: Start position
            goal: Goal position
            obstacles: List of (center, radius) tuples
            gcs_regions: Optional GCS regions for planning
            
        Returns:
            SafeCorridor or None if no feasible corridor found
        """
        # Generate initial path
        waypoints = self._generate_initial_path(start, goal, 10)
        
        # Optimize corridor
        waypoints = self._optimize_corridor(waypoints, obstacles)
        
        # Compute clearances
        clearances = self._compute_clearances(waypoints, obstacles)
        min_clearance = float(np.min(clearances))
        
        # Check feasibility
        feasible = min_clearance >= self.safety_margin
        
        # Compute corridor length
        corridor_length = float(np.sum(
            np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        ))
        
        corridor = SafeCorridor(
            waypoints=waypoints,
            clearances=clearances,
            min_clearance=min_clearance,
            corridor_length=corridor_length,
            feasible=feasible,
        )
        
        self.corridors.append(corridor)
        
        if feasible:
            logger.info(f"Planned feasible corridor (clearance={min_clearance:.4f})")
        else:
            logger.warning(f"Planned corridor is not feasible (clearance={min_clearance:.4f})")
        
        return corridor
    
    def _generate_initial_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        num_waypoints: int = 10,
    ) -> np.ndarray:
        """
        Generate initial straight-line path.
        
        Args:
            start: Start position
            goal: Goal position
            num_waypoints: Number of waypoints
            
        Returns:
            Waypoint array
        """
        t = np.linspace(0, 1, num_waypoints)
        waypoints = np.array([
            start + t_i * (goal - start)
            for t_i in t
        ])
        
        return waypoints
    
    def _optimize_corridor(
        self,
        waypoints: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Optimize corridor waypoints for safety and smoothness.
        
        Args:
            waypoints: Initial waypoints
            obstacles: List of obstacles
            
        Returns:
            Optimized waypoints
        """
        optimized = waypoints.copy()
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try to move each waypoint away from obstacles
            for i in range(1, len(optimized) - 1):
                gradient = self._compute_obstacle_gradient(optimized[i], obstacles)
                
                # Move away from obstacles
                step = gradient * 0.01
                new_point = optimized[i] + step
                
                # Check if new point improves corridor
                old_clearance = self._min_clearance_near_point(
                    optimized[i], optimized[i-1], optimized[i+1], obstacles
                )
                new_clearance = self._min_clearance_near_point(
                    new_point, optimized[i-1], optimized[i+1], obstacles
                )
                
                if new_clearance > old_clearance:
                    optimized[i] = new_point
                    improved = True
            
            if not improved:
                break
        
        return optimized
    
    def _compute_obstacle_gradient(
        self,
        point: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Compute gradient pointing away from nearest obstacle.
        
        Args:
            point: Query point
            obstacles: List of obstacles
            
        Returns:
            Gradient direction
        """
        gradient = np.zeros_like(point)
        
        for obs_center, obs_radius in obstacles:
            diff = point - obs_center
            dist = np.linalg.norm(diff)
            
            if dist < obs_radius + 0.5:
                if dist > 1e-6:
                    gradient += (diff / dist) * (1.0 / (dist + 1e-6))
        
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm
        
        return gradient
    
    def _compute_clearances(
        self,
        waypoints: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
    ) -> np.ndarray:
        """
        Compute clearance at each waypoint.
        
        Args:
            waypoints: Waypoint array
            obstacles: List of obstacles
            
        Returns:
            Clearance array
        """
        clearances = np.full(len(waypoints), float('inf'))
        
        for i, waypoint in enumerate(waypoints):
            for obs_center, obs_radius in obstacles:
                dist_to_obstacle = np.linalg.norm(waypoint - obs_center) - obs_radius
                clearances[i] = min(clearances[i], dist_to_obstacle)
        
        return clearances
    
    def _min_clearance_near_point(
        self,
        point: np.ndarray,
        prev_point: np.ndarray,
        next_point: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]],
    ) -> float:
        """
        Compute minimum clearance along segment through point.
        
        Args:
            point: Center point
            prev_point: Previous waypoint
            next_point: Next waypoint
            obstacles: List of obstacles
            
        Returns:
            Minimum clearance
        """
        min_clearance = float('inf')
        
        # Sample along segment
        for t in np.linspace(0, 1, 5):
            sample_point = prev_point + t * (point - prev_point)
            
            for obs_center, obs_radius in obstacles:
                dist = np.linalg.norm(sample_point - obs_center) - obs_radius
                min_clearance = min(min_clearance, dist)
        
        for t in np.linspace(0, 1, 5):
            sample_point = point + t * (next_point - point)
            
            for obs_center, obs_radius in obstacles:
                dist = np.linalg.norm(sample_point - obs_center) - obs_radius
                min_clearance = min(min_clearance, dist)
        
        return min_clearance
    
    def refine_corridor(
        self,
        corridor: SafeCorridor,
        obstacles: List[Tuple[np.ndarray, float]],
        target_clearance: float = 0.2,
    ) -> SafeCorridor:
        """
        Refine corridor to meet target clearance.
        
        Args:
            corridor: Original corridor
            obstacles: List of obstacles
            target_clearance: Desired minimum clearance
            
        Returns:
            Refined corridor
        """
        waypoints = corridor.waypoints.copy()
        
        # Add more waypoints where clearance is low
        for i in range(len(waypoints) - 1):
            clearance = corridor.clearances[i]
            
            if clearance < target_clearance:
                # Add intermediate waypoint
                mid_point = (waypoints[i] + waypoints[i + 1]) / 2
                waypoints = np.vstack([
                    waypoints[:i + 1],
                    mid_point,
                    waypoints[i + 1:],
                ])
        
        # Re-optimize with more waypoints
        return self.plan_corridor(
            waypoints[0],
            waypoints[-1],
            obstacles,
        ) or corridor
    
    def get_corridor_info(self, corridor: SafeCorridor) -> Dict:
        """Get corridor information."""
        return {
            'num_waypoints': len(corridor.waypoints),
            'length': corridor.corridor_length,
            'min_clearance': corridor.min_clearance,
            'feasible': corridor.feasible,
            'average_clearance': float(np.mean(corridor.clearances)),
        }
