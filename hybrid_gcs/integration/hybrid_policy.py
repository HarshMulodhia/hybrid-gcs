"""
Hybrid Policy Module
File: hybrid_gcs/integration/hybrid_policy.py

Combines GCS planning with RL policy for hybrid decision-making.
"""

import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridPolicyConfig:
    """Configuration for hybrid policy."""
    
    use_gcs_planning: bool = True
    use_rl_policy: bool = True
    gcs_weight: float = 0.5
    rl_weight: float = 0.5
    blend_method: str = "weighted"  # "weighted", "switching", "hierarchical"
    planning_horizon: int = 10
    safety_margin: float = 0.1


class HybridPolicy:
    """
    Hybrid policy combining GCS planning with RL.
    
    Integrates geometric planning with learning for robust control.
    """
    
    def __init__(
        self,
        gcs_decomposer,
        rl_policy,
        config: HybridPolicyConfig,
    ):
        """
        Initialize hybrid policy.
        
        Args:
            gcs_decomposer: GCS decomposer instance
            rl_policy: RL policy (actor network)
            config: Hybrid policy configuration
        """
        self.gcs = gcs_decomposer
        self.rl_policy = rl_policy
        self.config = config
        
        self.gcs_trajectory = None
        self.rl_action = None
        self.hybrid_action = None
        
        logger.info(
            f"Initialized HybridPolicy "
            f"(blend={config.blend_method}, gcs={config.gcs_weight}, rl={config.rl_weight})"
        )
    
    def compute_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: Optional[list] = None,
    ) -> np.ndarray:
        """
        Compute hybrid action.
        
        Args:
            state: Current state
            goal: Goal state
            obstacles: Optional list of obstacles
            
        Returns:
            Hybrid action
        """
        action = np.zeros_like(state)
        
        if self.config.use_gcs_planning:
            gcs_action = self._gcs_action(state, goal, obstacles)
        else:
            gcs_action = np.zeros_like(state)
        
        if self.config.use_rl_policy:
            rl_action = self._rl_action(state)
        else:
            rl_action = np.zeros_like(state)
        
        # Blend actions
        if self.config.blend_method == "weighted":
            action = (
                self.config.gcs_weight * gcs_action +
                self.config.rl_weight * rl_action
            )
        
        elif self.config.blend_method == "switching":
            # Switch between GCS and RL based on confidence
            gcs_confidence = self._compute_gcs_confidence(state, goal)
            if gcs_confidence > 0.5:
                action = gcs_action
            else:
                action = rl_action
        
        elif self.config.blend_method == "hierarchical":
            # Use GCS for high-level planning, RL for low-level control
            action = gcs_action + rl_action * 0.1
        
        self.hybrid_action = action
        return action
    
    def _gcs_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: Optional[list] = None,
    ) -> np.ndarray:
        """Compute GCS-based action."""
        # Direction towards goal (normalized)
        direction = goal - state
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction = direction / distance
        
        # Scale by planning horizon
        action = direction * min(distance, 0.1)
        
        self.gcs_action = action
        return action
    
    def _rl_action(self, state: np.ndarray) -> np.ndarray:
        """Compute RL policy action."""
        try:
            # Convert to tensor if needed
            state_tensor = state.reshape(1, -1)
            
            # Get action from RL policy
            with np.errstate(all='ignore'):
                action, _ = self.rl_policy.sample_action(state_tensor)
            
            self.rl_action = action.flatten()
            return action.flatten()
        
        except Exception as e:
            logger.warning(f"RL policy error: {e}, using zero action")
            return np.zeros_like(state)
    
    def _compute_gcs_confidence(self, state: np.ndarray, goal: np.ndarray) -> float:
        """
        Compute confidence in GCS planning.
        
        Returns:
            Confidence score (0-1)
        """
        distance = np.linalg.norm(goal - state)
        
        # Confidence decreases with distance
        confidence = 1.0 / (1.0 + distance)
        
        return float(confidence)
    
    def set_weights(self, gcs_weight: float, rl_weight: float) -> None:
        """
        Update blend weights.
        
        Args:
            gcs_weight: Weight for GCS action
            rl_weight: Weight for RL action
        """
        total = gcs_weight + rl_weight
        self.config.gcs_weight = gcs_weight / total
        self.config.rl_weight = rl_weight / total
        
        logger.info(f"Updated weights: GCS={self.config.gcs_weight:.2f}, RL={self.config.rl_weight:.2f}")
    
    def get_trajectory(self, state: np.ndarray, goal: np.ndarray, horizon: int = 10) -> np.ndarray:
        """
        Get predicted trajectory.
        
        Args:
            state: Current state
            goal: Goal state
            horizon: Planning horizon
            
        Returns:
            Trajectory array
        """
        trajectory = [state.copy()]
        current = state.copy()
        
        for _ in range(horizon):
            action = self.compute_action(current, goal)
            current = current + action
            trajectory.append(current.copy())
        
        return np.array(trajectory)
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        return {
            'blend_method': self.config.blend_method,
            'gcs_weight': self.config.gcs_weight,
            'rl_weight': self.config.rl_weight,
            'last_gcs_action': self.gcs_action if hasattr(self, 'gcs_action') else None,
            'last_rl_action': self.rl_action if hasattr(self, 'rl_action') else None,
            'last_hybrid_action': self.hybrid_action if hasattr(self, 'hybrid_action') else None,
        }
