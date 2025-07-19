"""
Configuration Manager for MCP System.

Provides centralized configuration management with support for:
- YAML and JSON configuration files
- Environment variable substitution
- Configuration validation
- Hot-reload capability
- Environment-specific overrides
"""

import os
import json
import yaml
import re
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import threading
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigFileNotFoundError(Exception):
    """Raised when configuration file is not found."""
    pass

class SecurityConfig(BaseModel):
    """Security configuration validation."""
    api_keys: Dict[str, Union[str, int]]
    rate_limiting: Dict[str, Union[bool, int]]
    authentication: Dict[str, Union[bool, int]]
    cors: Dict[str, Union[bool, List[str]]]
    headers: Dict[str, Union[str, int]]

class AgentConfig(BaseModel):
    """Agent configuration validation."""
    name: str
    host: str = "localhost"
    port: int = Field(gt=1000, lt=65536)
    module_path: str
    features: Dict[str, bool] = {}
    timeouts: Dict[str, int] = {}
    resources: Dict[str, int] = {}

class SystemConfig(BaseModel):
    """Main system configuration validation."""
    name: str
    version: str
    environment: str = Field(pattern="^(development|staging|production)$")

class ConfigFileHandler(FileSystemEventHandler):
    """Handles configuration file changes for hot-reload."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path == str(self.config_manager.config_file):
            logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager._reload_config())

class ConfigManager:
    """Manages configuration loading, validation, and hot-reload."""
    
    def __init__(self, config_file: Union[str, Path], environment: str = "development"):
        self.config_file = Path(config_file)
        self.environment = environment
        self.config: Dict[str, Any] = {}
        self.watchers: List[callable] = []
        self.observer: Optional[Observer] = None
        self._lock = threading.RLock()
        
        # Validation schemas
        self.validators = {
            'system': SystemConfig,
            'security': SecurityConfig,
            'agents': Dict[str, AgentConfig]
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from file."""
        with self._lock:
            try:
                if not self.config_file.exists():
                    raise ConfigFileNotFoundError(f"Configuration file not found: {self.config_file}")
                
                # Load configuration file
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                        raw_config = yaml.safe_load(f)
                    elif self.config_file.suffix.lower() == '.json':
                        raw_config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {self.config_file.suffix}")
                
                # Substitute environment variables
                config = self._substitute_env_vars(raw_config)
                
                # Apply environment-specific overrides
                config = self._apply_environment_overrides(config)
                
                # Validate configuration
                self._validate_config(config)
                
                self.config = config
                logger.info(f"Configuration loaded successfully from {self.config_file}")
                
                # Notify watchers
                self._notify_watchers()
                
                return self.config
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                raise
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern for environment variable substitution: ${VAR_NAME} or ${VAR_NAME:default_value}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) or ""
                return os.getenv(var_name, default_value)
            
            return re.sub(pattern, replace_var, config)
        else:
            return config
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        if self.environment in config:
            override_config = config[self.environment]
            config = self._deep_merge(config, override_config)
            logger.info(f"Applied {self.environment} environment overrides")
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration using Pydantic models."""
        try:
            # Validate system configuration
            if 'system' in config:
                SystemConfig(**config['system'])
            
            # Validate security configuration
            if 'security' in config:
                SecurityConfig(**config['security'])
            
            # Validate agent configurations
            if 'agents' in config:
                for agent_name, agent_config in config['agents'].items():
                    AgentConfig(**agent_config)
            
            logger.info("Configuration validation passed")
            
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'mcp_server.port')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        with self._lock:
            keys = key_path.split('.')
            config = self.config
            
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            config[keys[-1]] = value
            logger.info(f"Configuration updated: {key_path} = {value}")
            
            # Notify watchers
            self._notify_watchers()
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.get(f'agents.{agent_name}', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.get('security', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get('monitoring', {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.get(f'features.{feature_name}', False)
    
    def start_hot_reload(self):
        """Start watching configuration file for changes."""
        if self.observer is not None:
            logger.warning("Hot reload is already started")
            return
        
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            self.observer.schedule(handler, str(self.config_file.parent), recursive=False)
            self.observer.start()
            logger.info(f"Started hot reload for {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to start hot reload: {e}")
    
    def stop_hot_reload(self):
        """Stop watching configuration file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped hot reload")
    
    async def _reload_config(self):
        """Reload configuration from file."""
        try:
            old_config = self.config.copy()
            self.load_config()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            # Restore old config on failure
            self.config = old_config
    
    def add_watcher(self, callback: callable):
        """Add a callback to be notified when configuration changes."""
        self.watchers.append(callback)
    
    def remove_watcher(self, callback: callable):
        """Remove a configuration change callback."""
        if callback in self.watchers:
            self.watchers.remove(callback)
    
    def _notify_watchers(self):
        """Notify all watchers of configuration changes."""
        for watcher in self.watchers:
            try:
                watcher(self.config)
            except Exception as e:
                logger.error(f"Error notifying configuration watcher: {e}")
    
    def export_config(self, filepath: Union[str, Path], format: str = "yaml"):
        """Export current configuration to file."""
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Configuration exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information about the configuration."""
        return {
            "config_file": str(self.config_file),
            "environment": self.environment,
            "last_loaded": datetime.now().isoformat(),
            "hot_reload_active": self.observer is not None,
            "watchers_count": len(self.watchers),
            "config_size": len(str(self.config))
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def initialize_config(config_file: Union[str, Path], environment: str = "development") -> ConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_file, environment)
    _config_manager.load_config()
    return _config_manager

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    if _config_manager is None:
        raise RuntimeError("Configuration manager not initialized. Call initialize_config() first.")
    return _config_manager

def get_config(key_path: str = None, default: Any = None) -> Any:
    """Get configuration value using the global configuration manager."""
    manager = get_config_manager()
    if key_path is None:
        return manager.config
    return manager.get(key_path, default)

# Convenience functions for common configuration access
def get_agent_port(agent_name: str) -> int:
    """Get port for a specific agent."""
    return get_config(f'agents.{agent_name}.port', 0)

def get_mcp_server_config() -> Dict[str, Any]:
    """Get MCP server configuration."""
    return get_config('mcp_server', {})

def get_api_key(service: str = 'mcp_server') -> str:
    """Get API key for a service."""
    return get_config(f'security.api_keys.{service}', '')

def is_development_mode() -> bool:
    """Check if running in development mode."""
    return get_config('system.environment', 'development') == 'development'


if __name__ == "__main__":
    # Example usage
    config_file = Path(__file__).parent.parent / "config" / "system_config.yaml"
    
    try:
        manager = initialize_config(config_file, "development")
        manager.start_hot_reload()
        
        print(f"System: {get_config('system.name')}")
        print(f"MCP Port: {get_agent_port('mcp_server')}")
        print(f"API Key: {get_api_key()}")
        print(f"Development Mode: {is_development_mode()}")
        
        # Keep running to test hot reload
        input("Press Enter to stop...")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if _config_manager:
            _config_manager.stop_hot_reload() 