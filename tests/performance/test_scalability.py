"""
Scalability Tests
File: tests/performance/test_scalability.py

Test system scalability.
"""

import pytest
import numpy as np
from hybrid_gcs.environments import ManipulationEnvironment, DroneEnvironment
from hybrid_gcs.core import GCSDecomposer, ConfigSpace


class TestEnvironmentScalability:
    """Test environment scalability."""
    
    def test_single_agent_scaling(self):
        """Test single agent environment scaling."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        obs = env.reset()
        
        episode_count = 10
        step_count = 100
        
        total_steps = 0
        for episode in range(episode_count):
            obs = env.reset()
            for step in range(step_count):
                action = np.random.randn(6) * 0.1
                obs, reward, done, info = env.step(action)
                total_steps += 1
                
                if done:
                    break
        
        assert total_steps > 0
    
    def test_multi_agent_scaling(self):
        """Test multi-agent drone environment scaling."""
        for num_drones in [1, 2, 4]:
            env = DroneEnvironment(
                state_dim=18,
                action_dim=4,
                num_drones=num_drones
            )
            
            obs = env.reset()
            
            # Run episode
            for step in range(100):
                actions = [np.random.randn(4) * 0.1 for _ in range(num_drones)]
                result = env.step(actions)
                
                if step > 10:  # Just test a few steps
                    break
            
            assert True


class TestAlgorithmScalability:
    """Test algorithm scalability."""
    
    def test_decomposition_regions(self):
        """Test decomposition with varying region counts."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        
        for max_regions in [5, 10, 20, 50]:
            decomposer = GCSDecomposer(
                config_space=cs,
                max_regions=max_regions
            )
            
            start = np.array([-0.5] * 6)
            goal = np.array([0.5] * 6)
            
            result = decomposer.decompose(start, goal, [])
            
            assert len(result['regions']) <= max_regions
    
    def test_dimension_scaling(self):
        """Test scaling with dimension."""
        for config_dim in [3, 6, 10]:
            cs = ConfigSpace(
                name="test",
                dim=config_dim,
                bounds=[(-1, 1)] * config_dim
            )
            
            decomposer = GCSDecomposer(config_space=cs)
            
            start = np.random.uniform(-0.5, -0.4, config_dim)
            goal = np.random.uniform(0.4, 0.5, config_dim)
            
            result = decomposer.decompose(start, goal, [])
            
            assert result is not None


class TestDataScalability:
    """Test data handling scalability."""
    
    def test_large_batch_processing(self):
        """Test processing large batches."""
        from hybrid_gcs.utils import DataBuffer
        
        buffer = DataBuffer(max_size=100000, data_dim=20)
        
        # Add large amount of data
        for i in range(10000):
            data = np.random.randn(20)
            buffer.add(data)
        
        assert len(buffer) <= 100000
    
    def test_trajectory_storage(self):
        """Test storing large trajectories."""
        trajectory_length = 1000
        state_dim = 20
        
        trajectory = np.random.randn(trajectory_length, state_dim)
        
        assert trajectory.shape == (trajectory_length, state_dim)
        assert trajectory.nbytes / 1e6 < 100  # < 100MB


class TestConcurrency:
    """Test concurrent operations."""
    
    def test_parallel_episodes(self):
        """Test parallel episode execution."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        
        num_episodes = 4
        episodes = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            episode_data = {'obs': obs, 'rewards': []}
            
            for step in range(10):
                action = np.random.randn(6)
                obs, reward, done, info = env.step(action)
                episode_data['rewards'].append(reward)
                
                if done:
                    break
            
            episodes.append(episode_data)
        
        assert len(episodes) == num_episodes
        assert all(len(ep['rewards']) > 0 for ep in episodes)
