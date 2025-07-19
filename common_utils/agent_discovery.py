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
Agent Discovery System - Dynamic agent registration and discovery for A2A framework.

This module provides a registry where agents can register themselves on startup
and be discovered by other agents, particularly the orchestrator.
"""

import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

from .types import AgentCard

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Central registry for A2A agent discovery.
    
    Maintains a registry of active agents with their capabilities and endpoints.
    Provides health checking and automatic cleanup of inactive agents.
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file or "agents_registry.json"
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.load_registry()
        
    def register_agent(self, agent_card: Dict[str, Any]) -> bool:
        """Register an agent in the registry."""
        agent_name = agent_card.get("name")
        if not agent_name:
            logger.error("Cannot register agent without name")
            return False
        
        agent_info = {
            "card": agent_card,
            "registered_at": datetime.now().isoformat(),
            "last_health_check": None,
            "status": "active"
        }
        
        self.agents[agent_name] = agent_info
        self.save_registry()
        
        logger.info(f"Registered agent: {agent_name} at {agent_card.get('url')}")
        return True
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Remove an agent from the registry."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.save_registry()
            logger.info(f"Unregistered agent: {agent_name}")
            return True
        return False
    
    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent information by name."""
        return self.agents.get(agent_name)
    
    def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find agents that have a specific capability (skill)."""
        matching_agents = []
        
        for agent_name, agent_info in self.agents.items():
            agent_card = agent_info.get("card", {})
            skills = agent_card.get("skills", [])
            
            for skill in skills:
                if skill.get("id") == capability or skill.get("name") == capability:
                    matching_agents.append(agent_info)
                    break
                # Also check tags
                tags = skill.get("tags", [])
                if capability in tags:
                    matching_agents.append(agent_info)
                    break
        
        return matching_agents
    
    def list_agents(self, only_active: bool = True) -> Dict[str, Dict[str, Any]]:
        """List all registered agents."""
        if only_active:
            return {name: info for name, info in self.agents.items() 
                   if info.get("status") == "active"}
        return self.agents.copy()
    
    def get_agent_endpoints(self) -> Dict[str, str]:
        """Get a mapping of agent names to their endpoints."""
        endpoints = {}
        for agent_name, agent_info in self.agents.items():
            if agent_info.get("status") == "active":
                agent_card = agent_info.get("card", {})
                url = agent_card.get("url")
                if url:
                    endpoints[agent_name] = url
        return endpoints
    
    async def health_check_agent(self, agent_name: str) -> bool:
        """Check if an agent is responding to health checks."""
        agent_info = self.agents.get(agent_name)
        if not agent_info:
            return False
        
        agent_card = agent_info.get("card", {})
        url = agent_card.get("url")
        if not url:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/health")
                is_healthy = response.status_code == 200
                
                # Update health check status
                agent_info["last_health_check"] = datetime.now().isoformat()
                agent_info["status"] = "active" if is_healthy else "unhealthy"
                
                return is_healthy
                
        except Exception as e:
            logger.warning(f"Health check failed for {agent_name}: {e}")
            agent_info["last_health_check"] = datetime.now().isoformat()
            agent_info["status"] = "unreachable"
            return False
    
    async def health_check_all_agents(self) -> Dict[str, bool]:
        """Perform health checks on all registered agents."""
        results = {}
        
        for agent_name in self.agents:
            try:
                is_healthy = await self.health_check_agent(agent_name)
                results[agent_name] = is_healthy
            except Exception as e:
                logger.error(f"Error checking {agent_name}: {e}")
                results[agent_name] = False
        
        self.save_registry()
        return results
    
    def cleanup_inactive_agents(self, max_age_hours: int = 24) -> int:
        """Remove agents that haven't been seen for a while."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        inactive_agents = []
        
        for agent_name, agent_info in self.agents.items():
            last_check = agent_info.get("last_health_check")
            if last_check:
                last_check_time = datetime.fromisoformat(last_check)
                if last_check_time < cutoff_time and agent_info.get("status") != "active":
                    inactive_agents.append(agent_name)
            else:
                # No health check recorded and registered more than max_age ago
                registered_at = datetime.fromisoformat(agent_info.get("registered_at"))
                if registered_at < cutoff_time:
                    inactive_agents.append(agent_name)
        
        for agent_name in inactive_agents:
            self.unregister_agent(agent_name)
            logger.info(f"Cleaned up inactive agent: {agent_name}")
        
        return len(inactive_agents)
    
    def save_registry(self):
        """Save registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.agents, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def load_registry(self):
        """Load registry from file."""
        try:
            if Path(self.registry_file).exists():
                with open(self.registry_file, 'r') as f:
                    self.agents = json.load(f)
                logger.info(f"Loaded {len(self.agents)} agents from registry")
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            self.agents = {}

# Global registry instance
_registry = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

async def register_agent_with_registry(agent_card: Dict[str, Any]) -> bool:
    """Register an agent with the global registry."""
    registry = get_agent_registry()
    return registry.register_agent(agent_card)

async def discover_agents_by_capability(capability: str) -> List[Dict[str, Any]]:
    """Discover agents that have a specific capability."""
    registry = get_agent_registry()
    return registry.get_agent_by_capability(capability)

async def get_all_agent_endpoints() -> Dict[str, str]:
    """Get all active agent endpoints."""
    registry = get_agent_registry()
    return registry.get_agent_endpoints()

class AgentDiscoveryClient:
    """
    Client for agents to register themselves and discover other agents.
    """
    
    def __init__(self, agent_name: str, agent_url: str, registry_url: Optional[str] = None):
        self.agent_name = agent_name
        self.agent_url = agent_url
        self.registry_url = registry_url or "http://localhost:8000"  # Default orchestrator
        
    async def register_self(self) -> bool:
        """Register this agent with the discovery service."""
        try:
            # Get our agent card
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.agent_url}/agent_card")
                if response.status_code != 200:
                    logger.error(f"Failed to get agent card: {response.status_code}")
                    return False
                
                agent_card = response.json()
                
                # Register with the registry (if there's a central registry service)
                # For now, use local registry
                return await register_agent_with_registry(agent_card)
                
        except Exception as e:
            logger.error(f"Failed to register agent {self.agent_name}: {e}")
            return False
    
    async def discover_agents(self, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover other agents, optionally filtered by capability."""
        try:
            if capability:
                return await discover_agents_by_capability(capability)
            else:
                registry = get_agent_registry()
                return list(registry.list_agents().values())
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
            return [] 