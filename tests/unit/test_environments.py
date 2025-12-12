"""
Environment Tests
File: tests/unit/test_environments.py

Unit tests for environment modules.
"""

import pytest
import numpy as np
from hybrid_gcs.environments import (
    YCBGraspEnvironment,
    DroneEnvironment,
    ManipulationEnvironment
)


class TestYCBGraspEnvironment:
    """Test YCB grasping environment."""
    
    def test_initialization(self, state_dim, action_dim):
        """Test environment initialization."""
        env = YCBGraspEnvironment(
            state_dim=state_dim,
            action_dim=action_dim,
            max_steps=100
        )
        
        assert env is not None
        assert env.state_dim == state_dim
        assert env.action_dim == action_dim
        assert env.max_steps == 100
    
    def test_reset(self, state_dim):
        """Test environment reset."""
        env = YCBGraspEnvironment(state_dim=state_dim, action_dim=6)
        obs = env.reset()
        
        assert obs is not None
        assert obs.shape == (state_dim,)
        assert isinstance(obs, np.ndarray)
    
    def test_step(self, state_dim, action_dim):
        """Test environment step."""
        env = YCBGraspEnvironment(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        obs = env.reset()
        action = np.random.randn(action_dim) * 0.1
        
        obs, reward, done, info = env.step(action)
        
        assert obs.shape == (state_dim,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestDroneEnvironment:
    """Test drone environment."""
    
    def test_initialization(self):
        """Test drone environment initialization."""
        env = DroneEnvironment(
            state_dim=18,
            action_dim=4,
            num_drones=1
        )
        
        assert env is not None
        assert env.state_dim == 18
        assert env.action_dim == 4
        assert env.num_drones == 1
    
    def test_multi_agent(self):
        """Test multi-agent drone environment."""
        num_drones = 3
        env = DroneEnvironment(
            state_dim=18,
            action_dim=4,
            num_drones=num_drones
        )
        
        assert env.num_drones == num_drones
    
    def test_step_multi_agent(self):
        """Test multi-agent step."""
        num_drones = 2
        env = DroneEnvironment(
            state_dim=18,
            action_dim=4,
            num_drones=num_drones
        )
        
        obs = env.reset()
        actions = [np.random.randn(4) * 0.1 for _ in range(num_drones)]
        
        results = env.step(actions)
        
        assert results is not None


class TestManipulationEnvironment:
    """Test manipulation environment."""
    
    def test_initialization(self):
        """Test manipulation environment initialization."""
        env = ManipulationEnvironment(
            state_dim=25,
            action_dim=6,
            max_steps=200
        )
        
        assert env is not None
        assert env.state_dim == 25
        assert env.action_dim == 6
    
    def test_task_selection(self):
        """Test task selection."""
        tasks = ['reach', 'pick', 'push', 'stack']
        
        for task in tasks:
            env = ManipulationEnvironment(
                state_dim=25,
                action_dim=6,
                task=task
            )
            
            assert env is not None
    
    def test_joint_limits(self):
        """Test joint limit enforcement."""
        env = ManipulationEnvironment(state_dim=25, action_dim=6)
        
        # Should not exceed joint limits
        assert len(env.joint_limits) == 6
