"""
ConfigManager implementation for system configuration management.

ConfigManager provides centralized configuration management with
validation, environment variable support, and configuration merging.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    type: str
    required: bool = True
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    env_var: Optional[str] = None


class ConfigManager:
    """
    Configuration manager for BugForge systems.
    
    Provides centralized configuration with validation,
    environment variable support, and hierarchical configuration.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self._config: Dict[str, Any] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._config_file = config_file
        
        # Load initial configuration
        if config_file:
            self.load_from_file(config_file)
        
        # Load from environment variables
        self.load_from_environment()
    
    def define_schema(self, key: str, schema: ConfigSchema) -> None:
        """
        Define configuration schema for a key.
        
        Args:
            key: Configuration key
            schema: Schema definition
        """
        self._schemas[key] = schema
    
    def set(self, key: str, value: Any, validate: bool = True) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
            validate: Whether to validate against schema
        """
        if validate and key in self._schemas:
            self._validate_value(key, value)
        
        # Handle dot notation for nested keys
        keys = key.split('.')
        current = self._config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Handle dot notation
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            # Check schema for default
            if key in self._schemas and self._schemas[key].default is not None:
                return self._schemas[key].default
            return default
    
    def has(self, key: str) -> bool:
        """
        Check if configuration key exists.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return True
        except (KeyError, TypeError):
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete configuration key.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key was deleted
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys[:-1]:
                current = current[k]
            
            if keys[-1] in current:
                del current[keys[-1]]
                return True
            return False
        except (KeyError, TypeError):
            return False
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yml', '.yaml']:
                file_config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        # Merge with existing configuration
        self._merge_config(file_config)
    
    def save_to_file(self, file_path: str, format: str = "json") -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format ("json" or "yaml")
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            else:
                json.dump(self._config, f, indent=2, default=str)
    
    def load_from_environment(self, prefix: str = "BUGFORGE_") -> None:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON, fall back to string
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                self.set(config_key, parsed_value, validate=False)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self.get(section, {})
    
    def validate_all(self) -> List[str]:
        """
        Validate all configuration against schemas.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for key, schema in self._schemas.items():
            if schema.required and not self.has(key):
                errors.append(f"Required configuration missing: {key}")
                continue
            
            if self.has(key):
                try:
                    self._validate_value(key, self.get(key))
                except ValueError as e:
                    errors.append(f"Invalid configuration for {key}: {e}")
        
        return errors
    
    def get_schema_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all configuration schemas.
        
        Returns:
            Dictionary of schema information
        """
        schema_info = {}
        
        for key, schema in self._schemas.items():
            schema_info[key] = {
                "type": schema.type,
                "required": schema.required,
                "default": schema.default,
                "description": schema.description,
                "choices": schema.choices,
                "min_value": schema.min_value,
                "max_value": schema.max_value,
                "env_var": schema.env_var,
                "current_value": self.get(key) if self.has(key) else None
            }
        
        return schema_info
    
    def export_config(self, include_defaults: bool = False) -> Dict[str, Any]:
        """
        Export configuration dictionary.
        
        Args:
            include_defaults: Whether to include schema defaults
            
        Returns:
            Configuration dictionary
        """
        config = self._config.copy()
        
        if include_defaults:
            for key, schema in self._schemas.items():
                if not self.has(key) and schema.default is not None:
                    self._set_nested_value(config, key, schema.default)
        
        return config
    
    def reset(self) -> None:
        """Reset configuration to empty state."""
        self._config.clear()
    
    def _validate_value(self, key: str, value: Any) -> None:
        """Validate value against schema."""
        if key not in self._schemas:
            return
        
        schema = self._schemas[key]
        
        # Type validation
        if schema.type == "string" and not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")
        elif schema.type == "integer" and not isinstance(value, int):
            raise ValueError(f"Expected integer, got {type(value).__name__}")
        elif schema.type == "float" and not isinstance(value, (int, float)):
            raise ValueError(f"Expected number, got {type(value).__name__}")
        elif schema.type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Expected boolean, got {type(value).__name__}")
        elif schema.type == "list" and not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value).__name__}")
        elif schema.type == "dict" and not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")
        
        # Choices validation
        if schema.choices and value not in schema.choices:
            raise ValueError(f"Value must be one of {schema.choices}, got {value}")
        
        # Range validation
        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                raise ValueError(f"Value must be >= {schema.min_value}, got {value}")
            if schema.max_value is not None and value > schema.max_value:
                raise ValueError(f"Value must be <= {schema.max_value}, got {value}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing."""
        def merge_dict(base: Dict, update: Dict) -> None:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, new_config)
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value


# Global configuration instance
_global_config = ConfigManager()


def define_config(key: str, **kwargs) -> None:
    """
    Define configuration schema globally.
    
    Args:
        key: Configuration key
        **kwargs: Schema parameters
    """
    schema = ConfigSchema(**kwargs)
    _global_config.define_schema(key, schema)


def get_config() -> ConfigManager:
    """
    Get global configuration manager.
    
    Returns:
        Global configuration manager instance
    """
    return _global_config


def config(key: str, default: Any = None) -> Any:
    """
    Get configuration value globally.
    
    Args:
        key: Configuration key
        default: Default value
        
    Returns:
        Configuration value
    """
    return _global_config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set configuration value globally.
    
    Args:
        key: Configuration key
        value: Configuration value
    """
    _global_config.set(key, value)
