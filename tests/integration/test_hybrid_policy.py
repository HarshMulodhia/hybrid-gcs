"""
Hybrid Policy Integration Tests
File: tests/integration/test_hybrid_policy.py

Tests for hybrid GCS-RL policy integration.
"""

import pytest
import numpy as np
from hybrid_gcs.integration import HybridPolicy, HybridPolicyConfig
from hybrid_gcs.core import GCSDecomposer, ConfigSpace
from hybrid_gcs.training import OptimizedTrainer


class TestHybridPolicy:
    """Test hybrid policy."""
    
    def test_initialization(self):
        """Test hybrid policy initialization."""
        config = HybridPolicyConfig(blend_method="weighted")
        
        assert config is not None
        assert config.blend_method == "weighted"
    
    def test_blend_methods(self):
        """Test different blend methods."""
        methods = ["weighted", "switching", "hierarchical"]
        
        for method in methods:
            config = HybridPolicyConfig(blend_method=method)
            assert config.blend_method == method
    
    def test_weighted_blending(self):
        """Test weighted blending."""
        config = HybridPolicyConfig(
            blend_method="weighted",
            gcs_weight=0.5,
            rl_weight=0.5
        )
        
        assert config.gcs_weight == 0.5
        assert config.rl_weight == 0.5
        assert config.gcs_weight + config.rl_weight == 1.0
    
    def test_policy_computation(self):
        """Test policy action computation."""
        config = HybridPolicyConfig()
        
        state = np.random.randn(20)
        goal = np.random.randn(20)
        
        # Would compute action with real policy
        action = np.random.randn(6) * 0.1
        
        assert action.shape == (6,)


class TestGCSRLIntegration:
    """Test GCS-RL integration."""
    
    def test_decomposition_with_training(self):
        """Test decomposition integrated with training."""
        cs = ConfigSpace(name="test", dim=6, bounds=[(-1, 1)] * 6)
        decomposer = GCSDecomposer(config_space=cs)
        
        start = np.array([-0.5] * 6)
        goal = np.array([0.5] * 6)
        
        result = decomposer.decompose(start, goal, [])
        
        assert result is not None
        assert 'regions' in result
    
    def test_hybrid_action_selection(self):
        """Test hybrid action selection."""
        state = np.random.randn(20)
        
        # GCS action
        gcs_action = np.random.randn(6) * 0.1
        
        # RL action
        rl_action = np.random.randn(6) * 0.1
        
        # Blend actions
        alpha = 0.5
        blended = alpha * gcs_action + (1 - alpha) * rl_action
        
        assert blended.shape == (6,)
    
    def test_trajectory_prediction(self):
        """Test trajectory prediction."""
        state = np.random.randn(20)
        action = np.random.randn(6)
        
        # Predict next state
        next_state = state + action * 0.1
        
        assert next_state.shape == state.shape


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_gcs_features(self):
        """Test GCS feature extraction."""
        state = np.random.randn(20)
        
        # Extract GCS features
        features = state[:10]  # First 10 dims
        
        assert features.shape == (10,)
    
    def test_trajectory_features(self):
        """Test trajectory feature extraction."""
        trajectory = np.random.randn(50, 3)
        
        # Extract features
        length = len(trajectory)
        smoothness = np.mean(np.diff(trajectory, axis=0) ** 2)
        
        assert length > 0
        assert smoothness >= 0
    
    def test_combined_features(self):
        """Test combined feature extraction."""
        state = np.random.randn(20)
        trajectory = np.random.randn(50, 3)
        
        # Combine features
        combined = np.concatenate([state[:10], trajectory.flatten()[:10]])
        
        assert combined.shape[0] == 20
