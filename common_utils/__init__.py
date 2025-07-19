"""
Common utilities package for the multi-agent system.
Provides shared utilities for MCP-based multi-agent communication.
"""

from . import types
from . import constants
from . import config
from . import exceptions
from . import health
from . import data_handle_manager
from . import base_agent_server
from . import agent_config
from . import agent_discovery
# A2A imports removed - deprecated in favor of MCP

# Export key classes for easy access
from .base_agent_server import BaseA2AAgent, BaseAgentServer, create_agent_server
from .agent_config import AgentConfigManager, get_agent_config_manager, get_agent_endpoints, get_agent_endpoint
from .agent_discovery import AgentRegistry, get_agent_registry, AgentDiscoveryClient
 
__version__ = "0.1.0" 