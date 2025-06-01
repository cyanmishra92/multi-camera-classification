"""Configuration file parsing utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigParser:
    """Parse and validate configuration files."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        with open(path, 'r') as f:
            config = json.load(f)
            
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        required_sections = ['network', 'energy', 'accuracy']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
                
        # Validate network config
        network = config['network']
        if network.get('num_cameras', 0) <= 0:
            logger.error("num_cameras must be positive")
            return False
            
        if network.get('num_classes', 0) <= 0:
            logger.error("num_classes must be positive")
            return False
            
        # Validate energy config
        energy = config['energy']
        if energy.get('battery_capacity', 0) <= 0:
            logger.error("battery_capacity must be positive")
            return False
            
        if energy.get('classification_cost', 0) > energy.get('battery_capacity', 0):
            logger.warning("classification_cost exceeds battery_capacity")
            
        # Validate accuracy config
        accuracy = config['accuracy']
        if not 0 < accuracy.get('max_accuracy', 0) <= 1:
            logger.error("max_accuracy must be in (0, 1]")
            return False
            
        return True
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        import copy
        
        merged = copy.deepcopy(base_config)
        
        def recursive_merge(base: Dict, override: Dict) -> None:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    recursive_merge(base[key], value)
                else:
                    base[key] = value
                    
        recursive_merge(merged, override_config)
        return merged