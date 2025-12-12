"""
Performance Tests
File: tests/performance/test_speed.py

Performance benchmarks for speed.
"""

import pytest
import time
import numpy as np
from hybrid_gcs.core import GCSDecomposer, ConfigSpace
from hybrid_gcs.environments import ManipulationEnvironment


class TestSpeed:
    """Speed benchmarks."""
    
    def test_decomposition_speed(self):
        """Test GCS decomposition speed."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs, max_regions=10)
        
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        obstacles = [(np.array([0, 0, 0, 0, 0, 0]), 0.1)]
        
        start_time = time.time()
        result = decomposer.decompose(start, goal, obstacles)
        elapsed = time.time() - start_time
        
        # Should be reasonably fast
        assert elapsed < 5.0  # 5 second budget
    
    def test_environment_step_speed(self):
        """Test environment step speed."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        obs = env.reset()
        
        start_time = time.time()
        for _ in range(100):
            action = np.random.randn(6) * 0.1
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        elapsed = time.time() - start_time
        
        avg_step_time = elapsed / 100
        
        # Should be able to run at least 100 Hz (10ms per step)
        assert avg_step_time < 0.01  # 10ms per step


class TestSpeedRegression:
    """Test for speed regressions."""
    
    def test_decomposition_not_slower(self):
        """Test decomposition doesn't get slower."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs)
        
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        
        times = []
        for _ in range(5):
            start_time = time.time()
            decomposer.decompose(start, goal, [])
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        
        # Average should be reasonable
        assert avg_time < 1.0
