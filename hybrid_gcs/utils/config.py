"""
Configuration Management Module
File: hybrid_gcs/utils/config.py

Handles configuration loading and saving.
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration files and parameters.
    
    Supports YAML and JSON formats.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_file: Optional path to config file
        """
        self.config: Dict[str, Any] = {}
        
        if config_file:
            self.load(config_file)
    
    def load(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file (.yaml or .json)
        """
        path = Path(config_file)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".yaml" or suffix == ".yml":
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
                
        elif suffix == ".json":
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Loaded config from {config_file}")
    
    def save(self, config_file: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to output config file
        """
        path = Path(config_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        suffix = path.suffix.lower()
        
        if suffix == ".yaml" or suffix == ".yml":
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
        elif suffix == ".json":
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Saved config to {config_file}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Config key (supports dot notation: "section.subsection.key")
            default: Default value if key not found
            
        Returns:
            Config value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Config key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dict.
        
        Args:
            updates: Dict of updates to apply
        """
        self.config.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dict."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        return f"ConfigManager({json.dumps(self.config, indent=2)})"


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to config file
        
    Returns:
        Configuration dict
    """
    manager = ConfigManager(config_file)
    return manager.to_dict()


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dict
        config_file: Output path
    """
    manager = ConfigManager()
    manager.config = config
    manager.save(config_file)
