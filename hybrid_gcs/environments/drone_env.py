"""
Drone Navigation Environment
File: hybrid_gcs/environments/drone_env.py

RL environment for autonomous drone navigation.
"""

import logging
from typing import Dict, Tuple, Any
import numpy as np
from hybrid_gcs.environments.base_env import BaseEnvironment, EnvironmentConfig

logger = logging.getLogger(__name__)


class DroneNavigationEnvironment(BaseEnvironment):
    """
    Drone navigation environment.
    
    Agent learns to navigate drone to goal while avoiding obstacles.
    """
    
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 4,
        max_episode_steps: int = 500,
        world_size: float = 10.0,
        num_obstacles: int = 5,
    ):
        """
        Initialize drone navigation environment.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension (4D: throttle, roll, pitch, yaw)
            max_episode_steps: Max steps per episode
            world_size: Size of world
            num_obstacles: Number of obstacles
        """
        config = EnvironmentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_episode_steps=max_episode_steps,
            action_bounds=(-1.0, 1.0),
        )
        
        super().__init__(config)
        
        self.world_size = world_size
        self.num_obstacles = num_obstacles
        
        # Drone state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.goal = np.array([world_size / 2, world_size / 2, 5.0])
        
        # Obstacles
        self.obstacles = self._generate_obstacles()
        
        # Metrics
        self.distance_traveled = 0.0
        self.collisions = 0
        
        logger.info(
            f"Initialized DroneNavigationEnvironment "
            f"(world_size={world_size}, obstacles={num_obstacles})"
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.step_count = 0
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.zeros(3)
        self.distance_traveled = 0.0
        self.collisions = 0
        self.goal = np.random.uniform(
            [0, 0, 2],
            [self.world_size, self.world_size, 10]
        )
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        action = self._clip_action(action)
        self.step_count += 1
        
        # Update velocity from action
        throttle, roll, pitch, yaw = action
        
        # Simple physics model
        acc = np.array([
            roll * 5.0,
            pitch * 5.0,
            (throttle + 1) * 2.5 - 10.0,  # Gravity compensation
        ])
        
        self.velocity = self.velocity * 0.95 + acc * 0.05
        self.velocity = np.clip(self.velocity, -5.0, 5.0)
        
        # Update position
        delta_pos = self.velocity * 0.01
        self.position += delta_pos
        self.distance_traveled += np.linalg.norm(delta_pos)
        
        # Check collisions
        collision = self._check_collision()
        if collision:
            self.collisions += 1
        
        # Boundary check
        self.position = np.clip(
            self.position,
            [0, 0, 0.5],
            [self.world_size, self.world_size, 10]
        )
        
        # Compute reward
        reward = self._compute_reward(collision)
        
        # Check done
        distance_to_goal = np.linalg.norm(self.position - self.goal)
        done = (
            self.step_count >= self.config.max_episode_steps
            or distance_to_goal < 0.5
            or self.position[2] < 0.1
        )
        
        info = {
            'distance_to_goal': float(distance_to_goal),
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'collisions': self.collisions,
        }
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _generate_obstacles(self) -> list:
        """Generate random obstacles."""
        obstacles = []
        for _ in range(self.num_obstacles):
            pos = np.random.uniform([0, 0, 1], [self.world_size, self.world_size, 9])
            radius = np.random.uniform(0.3, 1.0)
            obstacles.append((pos, radius))
        return obstacles
    
    def _check_collision(self) -> bool:
        """Check if drone collides with obstacle."""
        drone_radius = 0.2
        
        for obs_pos, obs_radius in self.obstacles:
            distance = np.linalg.norm(self.position - obs_pos)
            if distance < drone_radius + obs_radius:
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.config.state_dim)
        
        # Position and velocity
        obs[:3] = self.position
        obs[3:6] = self.velocity
        
        # Goal
        obs[6:9] = self.goal
        
        # Distance to goal
        obs[9] = np.linalg.norm(self.position - self.goal)
        
        # Obstacle distances (nearest 3)
        distances = []
        for obs_pos, _ in self.obstacles:
            dist = np.linalg.norm(self.position - obs_pos)
            distances.append(dist)
        
        distances.sort()
        for i, dist in enumerate(distances[:min(3, len(distances))]):
            obs[10 + i] = dist
        
        return obs
    
    def _compute_reward(self, collision: bool) -> float:
        """Compute reward."""
        reward = 0.0
        
        # Distance to goal
        distance = np.linalg.norm(self.position - self.goal)
        reward -= distance * 0.1
        
        # Collision penalty
        if collision:
            reward -= 5.0
        
        # Goal reached bonus
        if distance < 0.5:
            reward += 10.0
        
        return float(reward)


class MultiDroneEnvironment(DroneNavigationEnvironment):
    """Multi-drone navigation environment."""
    
    def __init__(
        self,
        state_dim: int = 40,
        action_dim: int = 8,
        num_drones: int = 2,
    ):
        """Initialize multi-drone environment."""
        super().__init__(state_dim, action_dim)
        self.num_drones = num_drones
        self.positions = [np.zeros(3) for _ in range(num_drones)]
        self.velocities = [np.zeros(3) for _ in range(num_drones)]
        
        logger.info(f"Initialized MultiDroneEnvironment with {num_drones} drones")
    
    def reset(self) -> np.ndarray:
        """Reset multi-drone environment."""
        self.step_count = 0
        
        for i in range(self.num_drones):
            self.positions[i] = np.random.uniform([0, 0, 1], [5, 5, 5])
            self.velocities[i] = np.zeros(3)
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation from all drones."""
        obs = np.zeros(self.config.state_dim)
        
        for i, (pos, vel) in enumerate(zip(self.positions, self.velocities)):
            idx = i * 10
            obs[idx:idx + 3] = pos
            obs[idx + 3:idx + 6] = vel
        
        return obs
