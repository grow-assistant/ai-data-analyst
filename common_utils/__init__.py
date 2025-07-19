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
from . import security
from . import agent_security
from .mcp_server import tool_server
from . import session_manager
from . import observability
from . import enhanced_logging
from . import workflow_manager
# A2A imports removed - deprecated in favor of MCP

# Export key classes for easy access
from .base_agent_server import BaseA2AAgent, BaseAgentServer, create_agent_server
from .agent_config import AgentConfigManager, get_agent_config_manager, get_agent_endpoints, get_agent_endpoint
from .agent_discovery import AgentRegistry, get_agent_registry, AgentDiscoveryClient
from .agent_security import AgentSecurityHelper, get_agent_security_helper, create_secure_headers
from .security import SecurityManager, security_manager
from .mcp_server.tool_server import BaseTool, McpToolServer, ToolInput, ToolDefinition
from .session_manager import SessionManager, get_session_manager, Session, State, Event
from .observability import ObservabilityManager, get_observability_manager, trace_operation, instrument
from .enhanced_logging import get_logger, logging_context, correlated_operation, add_logging_context
from .workflow_manager import WorkflowManager, get_workflow_manager, WorkflowDefinition, TaskDefinition
 
__version__ = "0.1.0" 