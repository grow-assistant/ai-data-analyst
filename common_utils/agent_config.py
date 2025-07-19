#!/usr/bin/env python3

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agent Configuration Manager - Centralized configuration for A2A agents.

This module provides configuration management for agents, loading settings
from the system config file and providing a consistent interface for
accessing agent endpoints, ports, and other settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentConfigManager:
    """Manages configuration for A2A agents."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self.agent_endpoints = self._build_agent_endpoints()
    
    def _find_config_file(self) -> str:
        """Find the system configuration file."""
        # Look for config file in common locations
        possible_paths = [
            "config/system_config.yaml",
            "../config/system_config.yaml",
            "../../config/system_config.yaml",
            "../../../config/system_config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).absolute())
        
        # Fallback to environment variable or default
        env_path = os.getenv("AGENT_CONFIG_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        
        raise FileNotFoundError(
            f"Could not find system_config.yaml in any of these locations: {possible_paths}"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    
    def _build_agent_endpoints(self) -> Dict[str, str]:
        """Build agent endpoints dictionary from configuration."""
        endpoints = {}
        agents_config = self.config.get("agents", {})
        
        for agent_name, agent_config in agents_config.items():
            if isinstance(agent_config, dict):
                host = agent_config.get("host", "localhost")
                port = agent_config.get("port")
                
                if port:
                    endpoint = f"http://{host}:{port}"
                    endpoints[agent_name] = endpoint
                    logger.debug(f"Configured endpoint for {agent_name}: {endpoint}")
        
        # Handle legacy agent name mappings
        endpoint_mappings = {
            "data_loader": endpoints.get("data_loader", "http://localhost:10006"),
            "data_cleaning": endpoints.get("data_cleaning", "http://localhost:10008"),
            "data_enrichment": endpoints.get("data_enrichment", "http://localhost:10009"),
            "data_analyst": endpoints.get("data_analyst", "http://localhost:10007"),
            "presentation": endpoints.get("presentation", "http://localhost:10010"),
            "schema_profiler": endpoints.get("schema_profiler_agent", "http://localhost:10012"),
            "rootcause_analyst": endpoints.get("rootcause_analyst_agent", "http://localhost:10011")
        }
        
        logger.info(f"Built agent endpoints: {endpoint_mappings}")
        return endpoint_mappings
    
    def get_agent_endpoint(self, agent_name: str) -> Optional[str]:
        """Get endpoint URL for a specific agent."""
        return self.agent_endpoints.get(agent_name)
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get full configuration for a specific agent."""
        agents_config = self.config.get("agents", {})
        return agents_config.get(agent_name, {})
    
    def get_agent_port(self, agent_name: str) -> Optional[int]:
        """Get port number for a specific agent."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("port")
    
    def get_agent_host(self, agent_name: str) -> str:
        """Get host for a specific agent."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("host", "localhost")
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.config.get("security", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("enabled", True)  # Default to enabled
    
    def get_agent_timeouts(self, agent_name: str) -> Dict[str, int]:
        """Get timeout configuration for an agent."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("timeouts", {
            "startup": 30,
            "health_check": 10,
            "task_execution": 300
        })
    
    def get_agent_resource_limits(self, agent_name: str) -> Dict[str, Any]:
        """Get resource limits for an agent."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("resources", {
            "max_memory_mb": 512,
            "max_cpu_percent": 50
        })
    
    def update_agent_config(self, agent_name: str, updates: Dict[str, Any]) -> None:
        """Update agent configuration (in memory only)."""
        if "agents" not in self.config:
            self.config["agents"] = {}
        
        if agent_name not in self.config["agents"]:
            self.config["agents"][agent_name] = {}
        
        self.config["agents"][agent_name].update(updates)
        
        # Rebuild endpoints if host/port changed
        if "host" in updates or "port" in updates:
            self.agent_endpoints = self._build_agent_endpoints()
        
        logger.info(f"Updated configuration for agent {agent_name}")

# Global instance for easy access
_config_manager = None

def get_agent_config_manager() -> AgentConfigManager:
    """Get the global agent configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = AgentConfigManager()
    return _config_manager

def get_agent_endpoints() -> Dict[str, str]:
    """Get all agent endpoints from configuration."""
    return get_agent_config_manager().agent_endpoints

def get_agent_endpoint(agent_name: str) -> Optional[str]:
    """Get endpoint for a specific agent."""
    return get_agent_config_manager().get_agent_endpoint(agent_name) 