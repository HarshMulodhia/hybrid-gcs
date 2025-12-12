"""
Base Configuration Class
File: hybrid_gcs/training/configs/base_config.py

Base configuration classes for training.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Environment
    env_name: str = "manipulation"
    state_dim: int = 20
    action_dim: int = 6
    
    # Training
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    batch_size: int = 32
    
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    
    # Network
    hidden_sizes: list = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    
    # Curriculum
    use_curriculum: bool = True
    curriculum_type: str = "linear"
    
    # Checkpointing
    checkpoint_freq: int = 100
    save_dir: str = "results/models"
    
    # Logging
    log_freq: int = 10
    log_dir: str = "results/logs"


class BaseConfig:
    """Base configuration class."""
    
    def __init__(self, **kwargs):
        """Initialize configuration."""
        self.config: Dict[str, Any] = {}
        self.update(**kwargs)
        logger.info(f"Initialized config with {len(self.config)} keys")
    
    def update(self, **kwargs) -> None:
        """Update configuration."""
        for key, value in kwargs.items():
            self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()
    
    def to_yaml(self, filepath: str) -> None:
        """Save to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved config to {filepath}")
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseConfig":
        """Load from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config from {filepath}")
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = ['learning_rate', 'batch_size', 'num_episodes']
        
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Missing required key: {key}")
                return False
        
        return True
