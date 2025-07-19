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
Base Agent Server - Standardized FastAPI server for A2A agents.

This module provides a base class for creating consistent A2A agent servers,
eliminating duplication and ensuring standardized interfaces across all agents.
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .security import SecurityManager
from .types import AgentCard
from .health import HealthChecker
from .agent_discovery import AgentDiscoveryClient, register_agent_with_registry

logger = logging.getLogger(__name__)

class BaseA2AAgent:
    """Base class for A2A agents with standardized interface."""
    
    def __init__(self, executor, agent_name: str, agent_description: str = ""):
        self.executor = executor
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.skills = self._extract_skills()
        
    def _extract_skills(self) -> Dict[str, Callable]:
        """Extract skills from the executor based on methods ending with '_skill'."""
        skills = {}
        for attr_name in dir(self.executor):
            if attr_name.endswith('_skill') and callable(getattr(self.executor, attr_name)):
                # Create skill name by removing '_skill' suffix
                skill_name = attr_name[:-6]  # Remove '_skill'
                skills[skill_name] = getattr(self.executor, attr_name)
        return skills
    
    async def execute_skill(self, method: str, params: Dict[str, Any]) -> Any:
        """Execute a skill method with error handling."""
        if method not in self.skills:
            raise ValueError(f"Method not found: {method}")
        
        skill_func = self.skills[method]
        return await skill_func(**params)

class BaseAgentServer:
    """
    Base FastAPI server for A2A agents.
    
    Provides standardized endpoints, error handling, security integration,
    and health checks for all agents in the framework.
    """
    
    def __init__(
        self,
        agent: BaseA2AAgent,
        title: str,
        version: str = "1.0.0",
        port: int = 8000,
        enable_security: bool = True,
        custom_health_data: Optional[Dict[str, Any]] = None
    ):
        self.agent = agent
        self.title = title
        self.version = version
        self.port = port
        self.enable_security = enable_security
        self.custom_health_data = custom_health_data or {}
        
        # Initialize security manager if enabled
        self.security_manager = SecurityManager() if enable_security else None
        self.security_scheme = HTTPBearer(auto_error=False) if enable_security else None
        
        # Initialize health checker
        self.health_checker = HealthChecker()
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with standardized configuration."""
        app = FastAPI(
            title=self.title,
            description=f"A2A Agent: {self.agent.agent_description}",
            version=self.version,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on environment
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add standardized routes to the FastAPI app."""
        
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            health_data = {
                "status": "healthy",
                "agent": self.agent.agent_name,
                "version": self.version,
                "skills": list(self.agent.skills.keys()),
                **self.custom_health_data
            }
            
            # Add health checker results
            health_status = await self.health_checker.check_health()
            health_data.update(health_status)
            
            return health_data
        
        @app.get("/capabilities")
        async def capabilities():
            """Agent capabilities endpoint."""
            return {
                "agent_name": self.agent.agent_name,
                "description": self.agent.agent_description,
                "skills": list(self.agent.skills.keys()),
                "version": self.version
            }
        
        @app.get("/agent_card")
        async def agent_card():
            """Return agent card for A2A discovery."""
            from .types import AgentCard, AgentSkill, AgentCapabilities
            
            # Generate skills with better metadata
            skills = []
            for skill_name in self.agent.skills.keys():
                skill = AgentSkill(
                    id=skill_name,
                    name=skill_name.replace('_', ' ').title(),
                    description=f"Execute {skill_name} operation",
                    tags=[self.agent.agent_name, "a2a", "skill"]
                )
                skills.append(skill.model_dump() if hasattr(skill, 'model_dump') else skill.__dict__)
            
            # Create proper AgentCard structure
            card = AgentCard(
                name=self.agent.agent_name,
                description=self.agent.agent_description,
                url=f"http://localhost:{self.port}",
                version=self.version,
                capabilities=AgentCapabilities(),
                defaultInputModes=["json"],
                defaultOutputModes=["json"],
                skills=skills
            )
            
            return card.model_dump() if hasattr(card, 'model_dump') else card.__dict__
        
        @app.post("/")
        async def handle_jsonrpc_request(
            request: Request,
            credentials = Depends(self.security_scheme) if self.enable_security else None
        ):
            """Handle A2A JSON-RPC requests."""
            try:
                # Security validation
                if self.enable_security and self.security_manager:
                    # Extract API key from headers or bearer token
                    api_key = request.headers.get("X-API-Key")
                    if not api_key and credentials:
                        api_key = credentials.credentials
                    
                    if api_key:
                        if not self.security_manager.validate_api_key(api_key):
                            raise HTTPException(status_code=401, detail="Invalid API key")
                    # For now, allow requests without API key for internal communication
                
                # Parse JSON-RPC request
                request_data = await request.json()
                method = request_data.get("method")
                params = request_data.get("params", {})
                request_id = request_data.get("id")
                
                # Validate JSON-RPC format
                if not method:
                    return self._create_error_response(
                        -32600, "Invalid Request", request_id
                    )
                
                # Log the request
                logger.info(
                    f"A2A Request: {method} from {request.client.host if request.client else 'unknown'}"
                )
                
                # Execute the skill
                try:
                    result = await self.agent.execute_skill(method, params)
                    
                    return {
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": request_id
                    }
                    
                except ValueError as e:
                    # Method not found
                    return self._create_error_response(
                        -32601, f"Method not found: {method}", request_id
                    )
                except Exception as e:
                    # Internal error
                    logger.exception(f"Error executing skill {method}: {e}")
                    return self._create_error_response(
                        -32603, f"Internal error: {str(e)}", request_id
                    )
                    
            except Exception as e:
                logger.exception(f"Error handling A2A request: {e}")
                return self._create_error_response(
                    -32700, "Parse error", None
                )
        
        @app.post("/execute")
        async def handle_legacy_execute(
            request: Request,
            credentials = Depends(self.security_scheme) if self.enable_security else None
        ):
            """Legacy /execute endpoint for backward compatibility."""
            # Redirect to main JSON-RPC handler
            return await handle_jsonrpc_request(request, credentials)
    
    def _create_error_response(self, code: int, message: str, request_id: Optional[str]) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id
        }
    
    def run(self, host: str = "0.0.0.0", auto_register: bool = True, **uvicorn_kwargs):
        """Run the agent server."""
        logger.info(f"Starting {self.title} on {host}:{self.port}")
        logger.info(f"Available skills: {list(self.agent.skills.keys())}")
        
        # Auto-register with discovery service after a short delay
        if auto_register:
            async def register_after_startup():
                await asyncio.sleep(2)  # Wait for server to be ready
                try:
                    agent_url = f"http://{host}:{self.port}"
                    discovery_client = AgentDiscoveryClient(
                        agent_name=self.agent.agent_name,
                        agent_url=agent_url
                    )
                    success = await discovery_client.register_self()
                    if success:
                        logger.info(f"✅ Agent {self.agent.agent_name} registered with discovery service")
                    else:
                        logger.warning(f"⚠️ Failed to register {self.agent.agent_name} with discovery service")
                except Exception as e:
                    logger.error(f"Error during agent registration: {e}")
            
            # Schedule registration
            asyncio.create_task(register_after_startup())
        
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info",
            **uvicorn_kwargs
        )

def create_agent_server(
    executor,
    agent_name: str,
    title: str,
    port: int,
    agent_description: str = "",
    version: str = "1.0.0",
    enable_security: bool = True,
    custom_health_data: Optional[Dict[str, Any]] = None
) -> BaseAgentServer:
    """
    Factory function to create a standardized agent server.
    
    Args:
        executor: The agent executor with skill methods
        agent_name: Name of the agent
        title: Title for the FastAPI app
        port: Port to run the server on
        agent_description: Description of the agent
        version: Version of the agent
        enable_security: Whether to enable security features
        custom_health_data: Additional data to include in health checks
    
    Returns:
        BaseAgentServer instance ready to run
    """
    agent = BaseA2AAgent(executor, agent_name, agent_description)
    return BaseAgentServer(
        agent=agent,
        title=title,
        version=version,
        port=port,
        enable_security=enable_security,
        custom_health_data=custom_health_data
    ) 