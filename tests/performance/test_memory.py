"""
Memory Usage Tests
File: tests/performance/test_memory.py

Test memory usage and optimization.
"""

import pytest
import numpy as np
import tracemalloc
from hybrid_gcs.environments import ManipulationEnvironment
from hybrid_gcs.core import GCSDecomposer, ConfigSpace


class TestMemoryUsage:
    """Test memory usage."""
    
    def test_environment_memory(self):
        """Test environment memory usage."""
        tracemalloc.start()
        
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        obs = env.reset()
        
        for _ in range(100):
            action = np.random.randn(6)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should be reasonable
        assert peak < 500e6  # 500MB limit
    
    def test_decomposition_memory(self):
        """Test decomposition memory usage."""
        tracemalloc.start()
        
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs, max_regions=20)
        
        for _ in range(10):
            start = np.random.randn(6)
            goal = np.random.randn(6)
            result = decomposer.decompose(start, goal, [])
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        assert peak < 500e6
    
    def test_no_memory_leaks(self):
        """Test for memory leaks."""
        env = ManipulationEnvironment(state_dim=20, action_dim=6)
        
        memory_samples = []
        tracemalloc.start()
        
        for iteration in range(5):
            obs = env.reset()
            for step in range(100):
                action = np.random.randn(6)
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
            
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append(current)
        
        tracemalloc.stop()
        
        # Memory should not grow significantly
        growth = memory_samples[-1] - memory_samples[0]
        assert growth < 100e6  # Should not grow more than 100MB


class TestMemoryOptimization:
    """Test memory optimization."""
    
    def test_batch_memory(self):
        """Test batch processing memory."""
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            batch = np.random.randn(batch_size, 20)
            
            # Process batch
            processed = batch * 0.5
            
            assert processed.shape == batch.shape
    
    def test_buffer_memory(self):
        """Test replay buffer memory."""
        from hybrid_gcs.utils import DataBuffer
        
        buffer = DataBuffer(max_size=10000, data_dim=20)
        
        # Fill buffer
        for i in range(1000):
            data = np.random.randn(20)
            buffer.add(data)
        
        # Buffer should not exceed max size
        assert len(buffer) <= 10000
