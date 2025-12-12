"""
Feature Extraction Module
File: hybrid_gcs/integration/feature_extractor.py

Extracts features from GCS and trajectories for RL training.
"""

import logging
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Base class for feature extraction."""
    
    @abstractmethod
    def extract(self, data: Dict) -> np.ndarray:
        """
        Extract features from data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Feature vector
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        pass


class GCSFeatureExtractor(FeatureExtractor):
    """
    Extracts features from GCS decomposition.
    
    Features include region information, transitions, and distances.
    """
    
    def __init__(self, max_regions: int = 20):
        """
        Initialize GCS feature extractor.
        
        Args:
            max_regions: Maximum number of regions to consider
        """
        self.max_regions = max_regions
        self.feature_dim = max_regions * 5  # 5 features per region
        
        logger.info(f"Initialized GCSFeatureExtractor (max_regions={max_regions})")
    
    def extract(self, gcs_data: Dict) -> np.ndarray:
        """
        Extract GCS features.
        
        Args:
            gcs_data: GCS decomposition data
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.feature_dim)
        
        if 'regions' in gcs_data:
            regions = gcs_data['regions']
            n_regions = min(len(regions), self.max_regions)
            
            for i, region in enumerate(regions[:n_regions]):
                idx = i * 5
                
                # Region volume
                if hasattr(region, 'volume'):
                    features[idx] = region.volume
                
                # Region distance to start
                if 'start' in gcs_data:
                    features[idx + 1] = np.linalg.norm(region.center - gcs_data['start'])
                
                # Region distance to goal
                if 'goal' in gcs_data:
                    features[idx + 2] = np.linalg.norm(region.center - gcs_data['goal'])
                
                # Number of constraints
                if hasattr(region, 'constraints'):
                    features[idx + 3] = len(region.constraints)
                
                # Feasibility score
                if hasattr(region, 'feasible'):
                    features[idx + 4] = float(region.feasible)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim


class TrajectoryFeatureExtractor(FeatureExtractor):
    """
    Extracts features from trajectories.
    
    Features include length, smoothness, energy, and clearance metrics.
    """
    
    def __init__(self, trajectory_length: int = 100):
        """
        Initialize trajectory feature extractor.
        
        Args:
            trajectory_length: Expected trajectory length
        """
        self.trajectory_length = trajectory_length
        self.feature_dim = 12  # 12 trajectory metrics
        
        logger.info(f"Initialized TrajectoryFeatureExtractor")
    
    def extract(self, trajectory_data: Dict) -> np.ndarray:
        """
        Extract trajectory features.
        
        Args:
            trajectory_data: Trajectory data with 'trajectory', 'obstacles', etc.
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.feature_dim)
        
        if 'trajectory' not in trajectory_data:
            return features
        
        trajectory = np.array(trajectory_data['trajectory'])
        
        # 1. Length
        if len(trajectory) > 1:
            diffs = np.diff(trajectory, axis=0)
            features[0] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Smoothness (jerk)
        if len(trajectory) > 3:
            velocity = np.diff(trajectory, axis=0)
            acceleration = np.diff(velocity, axis=0)
            jerk = np.diff(acceleration, axis=0)
            jerk_norms = np.linalg.norm(jerk, axis=1)
            features[1] = float(np.sum(jerk_norms ** 2))
        
        # 3. Energy (velocity squared)
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            features[2] = float(np.sum(np.linalg.norm(velocity, axis=1) ** 2))
        
        # 4. Curvature
        if len(trajectory) > 2:
            diffs = np.diff(trajectory, axis=0)
            curvatures = []
            for i in range(len(diffs) - 1):
                v = diffs[i]
                a = diffs[i + 1]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-6:
                    cross = np.linalg.norm(np.cross(v, a))
                    curvature = cross / (v_norm ** 3 + 1e-6)
                    curvatures.append(curvature)
            if curvatures:
                features[3] = float(np.mean(curvatures))
        
        # 5-7. Start, middle, end distances
        if 'goal' in trajectory_data:
            goal = trajectory_data['goal']
            features[4] = float(np.linalg.norm(trajectory[0] - goal))
            if len(trajectory) > 0:
                features[5] = float(np.linalg.norm(trajectory[len(trajectory) // 2] - goal))
            features[6] = float(np.linalg.norm(trajectory[-1] - goal))
        
        # 8. Clearance to obstacles
        if 'obstacles' in trajectory_data:
            min_clearance = float('inf')
            for point in trajectory:
                for obs_pos, obs_radius in trajectory_data['obstacles']:
                    dist = np.linalg.norm(point - obs_pos) - obs_radius
                    min_clearance = min(min_clearance, dist)
            features[7] = min_clearance if min_clearance != float('inf') else 1.0
        
        # 9. Path efficiency
        if 'goal' in trajectory_data:
            goal = trajectory_data['goal']
            straight_line_dist = np.linalg.norm(trajectory[-1] - trajectory[0])
            if straight_line_dist > 1e-6:
                features[8] = features[0] / straight_line_dist
            else:
                features[8] = 1.0
        
        # 10-12. Velocity statistics
        if len(trajectory) > 1:
            velocity = np.diff(trajectory, axis=0)
            vel_norms = np.linalg.norm(velocity, axis=1)
            features[9] = float(np.mean(vel_norms))
            features[10] = float(np.std(vel_norms))
            features[11] = float(np.max(vel_norms))
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim


class CombinedFeatureExtractor(FeatureExtractor):
    """Combines multiple feature extractors."""
    
    def __init__(
        self,
        extractors: List[FeatureExtractor],
        names: Optional[List[str]] = None,
    ):
        """
        Initialize combined extractor.
        
        Args:
            extractors: List of feature extractors
            names: Optional names for extractors
        """
        self.extractors = extractors
        self.names = names or [f"extractor_{i}" for i in range(len(extractors))]
        self.feature_dim = sum(e.get_feature_dim() for e in extractors)
        
        logger.info(f"Initialized CombinedFeatureExtractor with {len(extractors)} extractors")
    
    def extract(self, data: Dict) -> np.ndarray:
        """
        Extract combined features.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Combined feature vector
        """
        features_list = []
        
        for extractor, name in zip(self.extractors, self.names):
            try:
                features = extractor.extract(data)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Error in {name}: {e}")
                features_list.append(np.zeros(extractor.get_feature_dim()))
        
        return np.concatenate(features_list)
    
    def get_feature_dim(self) -> int:
        """Get combined feature dimension."""
        return self.feature_dim
