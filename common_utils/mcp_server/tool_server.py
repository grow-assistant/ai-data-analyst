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
MCP Tool Server - A generic server for hosting and exposing tools via MCP.

This module provides a standardized way to expose agent capabilities as tools,
following the principles of the Model Context Protocol (MCP).
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn
import inspect

from ..security import SecurityManager, security_manager
from ..types import AgentCapabilities

logger = logging.getLogger(__name__)

class ToolInput(BaseModel):
    """Standard input schema for invoking a tool."""
    tool_name: str = Field(..., description="The name of the tool to invoke.")
    parameters: Dict[str, Any] = Field({}, description="The parameters to pass to the tool.")

class ToolDefinition(BaseModel):
    """Standard schema for defining a tool's metadata."""
    name: str
    description: str
    parameters: Dict[str, Any] # Using dict for OpenAPI schema generation

class BaseTool:
    """
    Abstract base class for a tool that can be hosted by the MCP Tool Server.
    """
    name: str = "base_tool"
    description: str = "A base tool."

    def __init__(self):
        # Extract schema from the `execute` method's signature
        self.parameters = self._get_params_from_signature()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        The main execution method for the tool.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Tool must implement the 'execute' method.")

    def get_definition(self) -> ToolDefinition:
        """Returns the tool's definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )

    def _get_params_from_signature(self) -> Dict[str, Any]:
        """Inspects the `execute` method signature to build an OpenAPI-compatible schema."""
        try:
            sig = inspect.signature(self.execute)
            properties = {}
            required = []
            for name, param in sig.parameters.items():
                if name in ('self', 'kwargs'):
                    continue
                
                param_type = param.annotation
                schema_type = "string" # default
                if param_type == str:
                    schema_type = "string"
                elif param_type == int or param_type == float:
                    schema_type = "number"
                elif param_type == bool:
                    schema_type = "boolean"
                elif param_type == list:
                    schema_type = "array"
                elif param_type == dict:
                    schema_type = "object"

                properties[name] = {"title": name.replace('_', ' ').title(), "type": schema_type}
                if param.default is inspect.Parameter.empty:
                    required.append(name)
                else:
                    properties[name]["default"] = param.default
            
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
        except Exception as e:
            logger.error(f"Could not generate schema for tool {self.name}: {e}")
            return {"type": "object", "properties": {}}


class McpToolServer:
    """
    A FastAPI server for hosting and exposing tools via MCP.
    """
    def __init__(self, title: str, description: str, version: str = "1.0.0"):
        self.app = FastAPI(title=title, description=description, version=version)
        self.tools: Dict[str, BaseTool] = {}
        self.security_manager = security_manager
        self._add_routes()

    def register_tool(self, tool: BaseTool):
        """Registers a tool with the server."""
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' is already registered. Overwriting.")
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def _add_routes(self):
        @self.app.get("/tools", response_model=List[ToolDefinition])
        async def list_tools():
            """Returns a list of all available tools."""
            return [tool.get_definition() for tool in self.tools.values()]

        @self.app.post("/invoke")
        async def invoke_tool(request: Request, tool_input: ToolInput):
            """Invokes a tool with the given parameters."""
            api_key = request.headers.get("X-API-Key")
            
            # Security Validation
            agent_id = self.security_manager.validate_api_key(api_key)
            if not agent_id:
                raise HTTPException(status_code=401, detail="Invalid API Key")

            tool = self.tools.get(tool_input.tool_name)
            if not tool:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_input.tool_name}' not found.")
            
            # Authorization (future enhancement: check ACLs)
            resource = f"mcp:tool:{tool.name}"
            await self.security_manager.audit_logger.log_action(agent_id, "invoke_tool", resource, "initiated", {"params": tool_input.parameters})

            try:
                # Validate parameters against tool's signature (basic)
                expected_params = set(tool.parameters.get("properties", {}).keys())
                provided_params = set(tool_input.parameters.keys())
                if not provided_params.issubset(expected_params):
                    logger.warning(f"Extra parameters provided for tool '{tool.name}'")

                result = await tool.execute(**tool_input.parameters)
                await self.security_manager.audit_logger.log_action(agent_id, "invoke_tool", resource, "success")
                return result
            except Exception as e:
                logger.exception(f"Error executing tool '{tool.name}': {e}")
                await self.security_manager.audit_logger.log_action(agent_id, "invoke_tool", resource, "failure", {"error": str(e)})
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8100):
        logger.info(f"Starting MCP Tool Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port) 