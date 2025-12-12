"""
Configuration Loader for Hybrid-GCS Training

File: hybrid_gcs/training/configs/config_loader.py

Provides:
- YAML-based configuration loading
- Configuration validation
- Default configurations
- Configuration merging
- Type safety
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Type, List
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidator:
    """Validates training configurations."""
    
    @staticmethod
    def validate_learning_rate(lr: float) -> None:
        """Validate learning rate."""
        if lr <= 0 or lr > 1:
            raise ValueError(f"Learning rate must be in (0, 1], got {lr}")
    
    @staticmethod
    def validate_batch_size(batch_size: int) -> None:
        """Validate batch size."""
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if batch_size & (batch_size - 1) != 0:
            logger.warning(f"Batch size {batch_size} is not a power of 2")
    
    @staticmethod
    def validate_gamma(gamma: float) -> None:
        """Validate discount factor."""
        if gamma <= 0 or gamma > 1:
            raise ValueError(f"Gamma must be in (0, 1], got {gamma}")
    
    @staticmethod
    def validate_clip_ratio(clip_ratio: float) -> None:
        """Validate PPO clip ratio."""
        if clip_ratio <= 0 or clip_ratio > 1:
            raise ValueError(f"Clip ratio must be in (0, 1], got {clip_ratio}")
    
    @staticmethod
    def validate_entropy_coeff(coeff: float) -> None:
        """Validate entropy coefficient."""
        if coeff < 0:
            raise ValueError(f"Entropy coefficient must be >= 0, got {coeff}")


class ConfigLoader:
    """
    Load and manage training configurations.
    
    Features:
    - Load from YAML files
    - Load from JSON files
    - Load from dictionaries
    - Configuration validation
    - Default configurations
    - Configuration merging
    """
    
    # Default configuration directory
    DEFAULT_CONFIG_DIR = Path(__file__).parent / "defaults"
    
    # Available preset configurations
    PRESETS = {
        'fast_train': {
            'num_envs': 4,
            'max_steps_per_episode': 100,
            'total_timesteps': 10_000,
            'batch_size': 64,
            'n_epochs': 3,
        },
        'standard': {
            'num_envs': 16,
            'max_steps_per_episode': 500,
            'total_timesteps': 1_000_000,
            'batch_size': 256,
            'n_epochs': 10,
        },
        'high_quality': {
            'num_envs': 32,
            'max_steps_per_episode': 1000,
            'total_timesteps': 10_000_000,
            'batch_size': 512,
            'n_epochs': 20,
            'learning_rate': 1e-4,
        },
        'distributed': {
            'num_envs': 64,
            'max_steps_per_episode': 500,
            'total_timesteps': 50_000_000,
            'batch_size': 1024,
            'n_epochs': 5,
            'lr_schedule': 'exponential',
        },
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Custom configuration directory
        """
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self._ensure_defaults()
    
    def _ensure_defaults(self):
        """Ensure default configuration files exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default configs if they don't exist
        for preset_name, preset_config in self.PRESETS.items():
            default_path = self.config_dir / f"{preset_name}.yaml"
            if not default_path.exists():
                self._save_yaml(default_path, preset_config)
                logger.info(f"Created default config: {default_path}")
    
    @staticmethod
    def _save_yaml(path: Path, data: Dict[str, Any]):
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def load_from_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        yaml_path = Path(path)
        if not yaml_path.is_absolute():
            yaml_path = self.config_dir / yaml_path
        
        config = self._load_yaml(yaml_path)
        logger.info(f"Loaded config from {yaml_path}")
        
        return config
    
    def load_from_json(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        json_path = Path(path)
        if not json_path.is_absolute():
            json_path = self.config_dir / json_path
        
        config = self._load_json(json_path)
        logger.info(f"Loaded config from {json_path}")
        
        return config
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load preset configuration.
        
        Args:
            preset_name: Name of preset ('fast_train', 'standard', etc.)
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If preset not found
        """
        if preset_name not in self.PRESETS:
            available = ', '.join(self.PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {available}"
            )
        
        config = copy.deepcopy(self.PRESETS[preset_name])
        logger.info(f"Loaded preset config: {preset_name}")
        
        return config
    
    def merge_configs(
        self,
        base: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge configuration dictionaries.
        
        Args:
            base: Base configuration
            overrides: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(base)
        
        for key, value in overrides.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator()
        
        if 'learning_rate' in config:
            validator.validate_learning_rate(config['learning_rate'])
        
        if 'batch_size' in config:
            validator.validate_batch_size(config['batch_size'])
        
        if 'gamma' in config:
            validator.validate_gamma(config['gamma'])
        
        if 'clip_ratio' in config:
            validator.validate_clip_ratio(config['clip_ratio'])
        
        if 'entropy_coeff' in config:
            validator.validate_entropy_coeff(config['entropy_coeff'])
        
        logger.info("Configuration validation passed")
    
    def load_config(
        self,
        path: Optional[str] = None,
        preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration with optional overrides.
        
        Args:
            path: Path to config file (YAML or JSON)
            preset: Preset name
            overrides: Configuration overrides
            validate: Whether to validate config
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If both path and preset are None or invalid
        """
        config = {}
        
        # Load base config
        if preset:
            config = self.load_preset(preset)
        elif path:
            if path.endswith('.yaml') or path.endswith('.yml'):
                config = self.load_from_yaml(path)
            elif path.endswith('.json'):
                config = self.load_from_json(path)
            else:
                raise ValueError(f"Unknown config format: {path}")
        else:
            # Load default config
            config = self.load_preset('standard')
        
        # Apply overrides
        if overrides:
            config = self.merge_configs(config, overrides)
        
        # Validate
        if validate:
            self.validate_config(config)
        
        return config
    
    def save_config(
        self,
        config: Dict[str, Any],
        path: str,
        format: str = 'yaml'
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            path: Output path
            format: File format ('yaml' or 'json')
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            self._save_yaml(out_path, config)
        elif format == 'json':
            with open(out_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved config to {out_path}")
    
    def get_available_presets(self) -> List[str]:
        """Get list of available presets."""
        return list(self.PRESETS.keys())
    
    def get_config_files(self) -> List[str]:
        """Get list of available config files."""
        files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            files.extend(self.config_dir.glob(ext))
        return [str(f.relative_to(self.config_dir)) for f in files]


# Convenience function
def load_training_config(
    path: Optional[str] = None,
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load training configuration.
    
    Args:
        path: Path to config file
        preset: Preset name
        overrides: Configuration overrides
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_config(path=path, preset=preset, overrides=overrides)
