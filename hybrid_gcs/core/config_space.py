"""
Configuration Space Utilities
File: hybrid_gcs/core/config_space.py

Utilities for managing configuration space, collision checking, and bounds.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)


class CollisionType(Enum):
    """Types of collision checks."""
    POINT = 0  # Single configuration point
    LINE = 1   # Linear interpolation between configs
    PATH = 2   # Path with multiple waypoints
    SELF = 3   # Self-collision


@dataclass
class ConfigBounds:
    """Configuration space bounds."""
    
    lower: np.ndarray  # (d,) lower bounds
    upper: np.ndarray  # (d,) upper bounds
    
    @property
    def dim(self) -> int:
        """Dimension of configuration space."""
        return len(self.lower)
    
    def contains(self, config: np.ndarray) -> bool:
        """Check if configuration is within bounds."""
        return np.all(config >= self.lower) and np.all(config <= self.upper)
    
    def clip(self, config: np.ndarray) -> np.ndarray:
        """Clip configuration to bounds."""
        return np.clip(config, self.lower, self.upper)


class ConfigSpace:
    """
    Configuration Space Manager.
    
    Manages configuration space bounds, collision checking,
    and interpolation utilities.
    
    Attributes:
        dim: Dimension of configuration space
        bounds: Configuration space bounds
        collision_checker: Collision checking function
    """
    
    def __init__(self, dim: int, bounds: Optional[ConfigBounds] = None, 
                 collision_checker: Optional[Callable] = None):
        """
        Initialize configuration space.
        
        Args:
            dim: Dimension of configuration space
            bounds: Configuration bounds (defaults to [-pi, pi]^d)
            collision_checker: Collision checking function
            
        Raises:
            ValueError: If invalid dimension
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        
        self.dim = dim
        
        if bounds is None:
            self.bounds = ConfigBounds(
                lower=np.full(dim, -np.pi),
                upper=np.full(dim, np.pi),
            )
        else:
            if bounds.dim != dim:
                raise ValueError(f"Bounds dimension {bounds.dim} != config dim {dim}")
            self.bounds = bounds
        
        self.collision_checker = collision_checker
        
        logger.info(f"Initialized ConfigSpace(dim={dim})")
    
    def is_valid(self, config: np.ndarray) -> bool:
        """
        Check if configuration is valid (within bounds and collision-free).
        
        Args:
            config: Configuration point
            
        Returns:
            True if valid, False otherwise
        """
        # Check bounds
        if not self.bounds.contains(config):
            return False
        
        # Check collision
        if self.collision_checker and not self.collision_checker(config):
            return False
        
        return True
    
    def interpolate(self, config1: np.ndarray, config2: np.ndarray, num_points: int) -> np.ndarray:
        """
        Interpolate between two configurations.
        
        Args:
            config1: Start configuration
            config2: End configuration
            num_points: Number of interpolation points
            
        Returns:
            (num_points, dim) array of interpolated configurations
        """
        if len(config1) != self.dim or len(config2) != self.dim:
            raise ValueError("Configuration dimension mismatch")
        
        t = np.linspace(0, 1, num_points)
        configs = np.outer(1 - t, config1) + np.outer(t, config2)
        
        return configs
    
    def straight_line_valid(self, config1: np.ndarray, config2: np.ndarray, num_checks: int = 10) -> bool:
        """
        Check if straight-line path between configs is collision-free.
        
        Args:
            config1: Start configuration
            config2: End configuration
            num_checks: Number of intermediate checks
            
        Returns:
            True if path is valid
        """
        if not self.collision_checker:
            return True
        
        configs = self.interpolate(config1, config2, num_checks)
        
        for config in configs:
            if not self.is_valid(config):
                return False
        
        return True
    
    def euclidean_distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """
        Compute Euclidean distance between configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(config1 - config2))
    
    def shortest_distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """
        Compute shortest distance considering bounds wrapping.
        
        For periodic dimensions (like angles), considers wrapping.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Shortest distance
        """
        diff = np.abs(config1 - config2)
        range_half = (self.bounds.upper - self.bounds.lower) / 2
        
        # For each dimension, take minimum of direct distance and wrapped distance
        wrapped_diff = np.minimum(diff, range_half * 2 - diff)
        
        return float(np.linalg.norm(wrapped_diff))
    
    def random_config(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random valid configuration.
        
        Args:
            seed: Random seed
            
        Returns:
            Random configuration in bounds
        """
        if seed is not None:
            np.random.seed(seed)
        
        max_attempts = 100
        for _ in range(max_attempts):
            config = np.random.uniform(self.bounds.lower, self.bounds.upper)
            if self.is_valid(config):
                return config
        
        logger.warning("Could not generate valid random configuration")
        return self.bounds.clip(np.random.uniform(self.bounds.lower, self.bounds.upper))


class CollisionChecker:
    """
    Base collision checker class.
    
    Can be extended for specific robot/environment setups.
    """
    
    def __init__(self, config_space: ConfigSpace):
        """
        Initialize collision checker.
        
        Args:
            config_space: Configuration space instance
        """
        self.config_space = config_space
        logger.info("Initialized CollisionChecker")
    
    def check_point(self, config: np.ndarray) -> bool:
        """
        Check collision for a single configuration.
        
        Args:
            config: Configuration point
            
        Returns:
            True if collision-free, False otherwise
        """
        raise NotImplementedError
    
    def check_path(self, configs: np.ndarray) -> bool:
        """
        Check collision along a path.
        
        Args:
            configs: (N, d) array of configurations
            
        Returns:
            True if entire path is collision-free
        """
        for config in configs:
            if not self.check_point(config):
                return False
        return True
    
    def check_line(self, config1: np.ndarray, config2: np.ndarray, num_checks: int = 10) -> bool:
        """
        Check collision along straight line.
        
        Args:
            config1: Start configuration
            config2: End configuration
            num_checks: Number of intermediate checks
            
        Returns:
            True if line is collision-free
        """
        configs = self.config_space.interpolate(config1, config2, num_checks)
        return self.check_path(configs)
    
    def __call__(self, config: np.ndarray) -> bool:
        """Make checker callable."""
        return self.check_point(config)