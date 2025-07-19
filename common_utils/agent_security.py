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
Agent Security Utilities - Helper functions for secure inter-agent communication.

This module provides utilities for agents to easily integrate with the security
framework for authentication and authorization.
"""

import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

class AgentSecurityHelper:
    """Helper class for agent security operations."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.api_key = None
        self.security_manager = None
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize security manager and get API key for this agent."""
        try:
            from .security import security_manager
            self.security_manager = security_manager
            
            # Get or register API key for this agent
            self.api_key = self.security_manager.get_agent_api_key(self.agent_name)
            if not self.api_key:
                self.api_key = self.security_manager.register_agent_api_key(self.agent_name)
            
            logger.info(f"Security initialized for agent: {self.agent_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize security for {self.agent_name}: {e}")
            self.security_manager = None
            self.api_key = None
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for inter-agent requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def make_secure_request(self, url: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        """Make a secure HTTP request to another agent."""
        headers = self.get_security_headers()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(f"Authentication failed for {self.agent_name} -> {url}")
                raise Exception("Authentication failed - invalid API key")
            elif e.response.status_code == 403:
                logger.error(f"Authorization failed for {self.agent_name} -> {url}")
                raise Exception("Authorization failed - insufficient permissions")
            else:
                logger.error(f"HTTP error {e.response.status_code} for {self.agent_name} -> {url}")
                raise Exception(f"HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Request failed for {self.agent_name} -> {url}: {e}")
            raise
    
    def validate_incoming_request(self, api_key: str) -> bool:
        """Validate an incoming API key."""
        if not self.security_manager:
            return True  # Security disabled
        
        return self.security_manager.validate_api_key(api_key) is not None
    
    async def log_security_event(self, action: str, resource: str, result: str, details: Optional[Dict[str, Any]] = None):
        """Log a security-related event."""
        if self.security_manager:
            await self.security_manager.audit_logger.log_action(
                agent_id=self.agent_name,
                action=action,
                resource=resource,
                result=result,
                details=details
            )

# Global instances for each agent type
_security_helpers: Dict[str, AgentSecurityHelper] = {}

def get_agent_security_helper(agent_name: str) -> AgentSecurityHelper:
    """Get or create a security helper for an agent."""
    if agent_name not in _security_helpers:
        _security_helpers[agent_name] = AgentSecurityHelper(agent_name)
    return _security_helpers[agent_name]

def create_secure_headers(agent_name: str) -> Dict[str, str]:
    """Quick utility to create secure headers for an agent."""
    helper = get_agent_security_helper(agent_name)
    return helper.get_security_headers()

async def make_secure_agent_call(agent_name: str, target_url: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    """Make a secure call from one agent to another."""
    helper = get_agent_security_helper(agent_name)
    return await helper.make_secure_request(target_url, payload, timeout) 