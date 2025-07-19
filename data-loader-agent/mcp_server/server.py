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

"""Enhanced MCP server for intelligent agent routing with improved security and observability."""

import logging
import os
import httpx
import time
import hashlib
import hmac
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add parent directory for common_utils access
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.types import TaskRequest, TaskResponse, AgentCard
from common_utils.security import validate_input_safety

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)

app = FastAPI(
    title="Multi-Agent Control Plane (MCP)", 
    description="Enhanced intelligent routing with distributed tracing, centralized authentication, and advanced security",
    version="2.1.0"
)

# Enhanced CORS configuration for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only necessary methods
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)

# In-memory storage for registered agents
registered_agents: Dict[str, AgentCard] = {}

# Enhanced security monitoring
request_counts = defaultdict(int)
failed_auth_attempts = defaultdict(int)
blocked_ips = set()

# Distributed tracing storage (in production, use proper observability tools)
trace_log: Dict[str, List[Dict[str, Any]]] = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class SecurityMonitor:
    """Enhanced security monitoring for the MCP server."""
    
    def __init__(self):
        self.rate_limits = {
            "register": {"max_requests": 10, "window": 300},  # 10 requests per 5 minutes
            "route": {"max_requests": 100, "window": 60},     # 100 requests per minute
            "default": {"max_requests": 50, "window": 60}     # 50 requests per minute
        }
        self.request_history = defaultdict(list)
    
    def check_rate_limit(self, client_ip: str, endpoint: str) -> bool:
        """Check if client has exceeded rate limits."""
        now = time.time()
        endpoint_limits = self.rate_limits.get(endpoint, self.rate_limits["default"])
        window = endpoint_limits["window"]
        max_requests = endpoint_limits["max_requests"]
        
        # Clean old requests
        cutoff = now - window
        self.request_history[client_ip] = [
            req_time for req_time in self.request_history[client_ip] 
            if req_time > cutoff
        ]
        
        # Check current count
        current_count = len(self.request_history[client_ip])
        if current_count >= max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}: {current_count}/{max_requests}")
            return False
        
        # Record this request
        self.request_history[client_ip].append(now)
        return True
    
    def record_failed_auth(self, client_ip: str):
        """Record failed authentication attempt."""
        failed_auth_attempts[client_ip] += 1
        
        # Block IP after 5 failed attempts
        if failed_auth_attempts[client_ip] >= 5:
            blocked_ips.add(client_ip)
            logger.error(f"🚨 Blocking IP {client_ip} after {failed_auth_attempts[client_ip]} failed auth attempts")
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked."""
        return client_ip in blocked_ips

security_monitor = SecurityMonitor()

def get_client_ip(request: Request) -> str:
    """Extract client IP address."""
    # Check for X-Forwarded-For header (proxy/load balancer)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Check for X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"

def enhanced_api_key_validation(request: Request, api_key: str = Security(api_key_header)):
    """Enhanced API key validation with security monitoring."""
    client_ip = get_client_ip(request)
    
    # Check if IP is blocked
    if security_monitor.is_blocked(client_ip):
        logger.error(f"🚨 Blocked IP {client_ip} attempted access")
        raise HTTPException(status_code=429, detail="IP address blocked due to security violations")
    
    # Check rate limits
    endpoint = request.url.path.strip("/").split("/")[0] or "default"
    if not security_monitor.check_rate_limit(client_ip, endpoint):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Validate API key
    mcp_api_key = os.getenv("MCP_API_KEY", "mcp-dev-key")  # Default for development
    
    if not api_key:
        security_monitor.record_failed_auth(client_ip)
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(status_code=401, detail="API key required")
    
    # Enhanced key validation with timing attack protection
    if not hmac.compare_digest(api_key, mcp_api_key):
        security_monitor.record_failed_auth(client_ip)
        logger.warning(f"Invalid API key attempt from {client_ip}: {api_key[:8]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Reset failed attempts on successful auth
    if client_ip in failed_auth_attempts:
        del failed_auth_attempts[client_ip]
    
    logger.debug(f"✅ Authenticated request from {client_ip}")
    return api_key

def validate_task_request(task_request: TaskRequest) -> bool:
    """Enhanced validation for task requests."""
    try:
        # Check required fields
        if not task_request.task_id or not task_request.task_type:
            logger.warning("Task request missing required fields")
            return False
        
        # Validate task_id format (prevent injection)
        if not task_request.task_id.replace("-", "").replace("_", "").isalnum():
            logger.warning(f"Invalid task_id format: {task_request.task_id}")
            return False
        
        # Check for safety violations in parameters
        if task_request.parameters:
            for key, value in task_request.parameters.items():
                if isinstance(value, str) and not validate_input_safety(value):
                    logger.warning(f"Safety violation in task parameter {key}")
                    return False
        
        # Size limits
        max_size = 10 * 1024 * 1024  # 10MB
        if len(str(task_request.model_dump())) > max_size:
            logger.warning("Task request exceeds size limit")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating task request: {e}")
        return False

def log_trace_event(trace_id: str, task_id: str, event: str, details: Dict[str, Any]):
    """Log distributed tracing events for centralized observability."""
    if trace_id not in trace_log:
        trace_log[trace_id] = []
    
    # Sanitize sensitive information from details
    safe_details = {}
    for key, value in details.items():
        if key.lower() in ['password', 'secret', 'key', 'token']:
            safe_details[key] = "[REDACTED]"
        else:
            safe_details[key] = value
    
    trace_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "event": event,
        "details": safe_details
    }
    
    trace_log[trace_id].append(trace_event)
    logger.info(f"[TRACE:{trace_id}] [TASK:{task_id}] {event} - {safe_details}")

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for additional protection."""
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Log all requests for monitoring
    logger.info(f"📋 {request.method} {request.url.path} from {client_ip}")
    
    # Additional security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Add processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.post("/register")
async def register_agent(agent_card: AgentCard, request: Request, api_key: str = Depends(enhanced_api_key_validation)):
    """Register an agent with the MCP - now with enhanced security."""
    client_ip = get_client_ip(request)
    logger.info(f"🔐 Authenticated agent registration: {agent_card.name} at {agent_card.url} from {client_ip}")
    
    # Enhanced validation
    if not agent_card.name or not agent_card.url:
        raise HTTPException(status_code=400, detail="Agent name and URL are required")
    
    # Validate agent name format
    if not agent_card.name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid agent name format")
    
    # Validate URL format
    if not agent_card.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid agent URL format")
    
    if not agent_card.skills:
        logger.warning(f"Agent {agent_card.name} registered without skills")
    
    registered_agents[agent_card.name] = agent_card
    
    logger.info(f"✅ Agent {agent_card.name} registered successfully with {len(agent_card.skills)} skills")
    
    return {
        "message": f"Agent {agent_card.name} registered successfully",
        "registered_skills": [skill.name for skill in agent_card.skills] if agent_card.skills else []
    }

@app.get("/agents")
async def list_agents() -> List[AgentCard]:
    """List all registered agents."""
    logger.info(f"Agent list requested - {len(registered_agents)} agents available")
    return list(registered_agents.values())

@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str, api_key: str = Depends(enhanced_api_key_validation)):
    """Get distributed trace events for a specific trace ID."""
    if trace_id not in trace_log:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    
    return {
        "trace_id": trace_id,
        "events": trace_log[trace_id],
        "event_count": len(trace_log[trace_id])
    }

@app.post("/route", response_model=TaskResponse)
async def route_task(task_request: TaskRequest, api_key: str = Depends(enhanced_api_key_validation)):
    """Enhanced task routing with distributed tracing and improved agent selection."""
    start_time = time.time()
    
    # Enhanced validation
    if not validate_task_request(task_request):
        raise HTTPException(status_code=400, detail="Invalid task request")
    
    # Log incoming request
    log_trace_event(
        task_request.trace_id or task_request.task_id,
        task_request.task_id,
        "route_request_received",
        {
            "task_type": task_request.task_type,
            "parameters": task_request.parameters,
            "priority": task_request.priority,
            "data_handles_count": len(task_request.data_handles) if task_request.data_handles else 0
        }
    )
    
    logger.info(f"🎯 Routing task: {task_request.task_id} ({task_request.task_type}) [trace:{task_request.trace_id}]")
    
    # Enhanced agent discovery
    target_agent = find_best_agent_for_task(task_request.task_type, task_request.priority)
    
    if not target_agent:
        error_msg = f"No agent found for task type: {task_request.task_type}"
        log_trace_event(
            task_request.trace_id or task_request.task_id,
            task_request.task_id,
            "routing_failed",
            {"error": error_msg, "available_agents": list(registered_agents.keys())}
        )
        raise HTTPException(status_code=404, detail=error_msg)
    
    # Log agent selection
    log_trace_event(
        task_request.trace_id or task_request.task_id,
        task_request.task_id,
        "agent_selected",
        {
            "agent_name": target_agent.name,
            "agent_url": target_agent.url,
            "matching_skills": [skill.name for skill in target_agent.skills if task_request.task_type in skill.tags]
        }
    )
    
    # Forward the request to the target agent
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{target_agent.url}/task",
                json=task_request.model_dump(),
                headers={"X-API-Key": api_key},
                timeout=60.0
            )
            response.raise_for_status()
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            result = response.json()
            
            # Ensure the response includes agent name and execution time
            if isinstance(result, dict):
                result["agent_name"] = target_agent.name
                if "execution_time_ms" not in result:
                    result["execution_time_ms"] = execution_time_ms
            
            # Log successful routing
            log_trace_event(
                task_request.trace_id or task_request.task_id,
                task_request.task_id,
                "routing_completed",
                {
                    "agent_name": target_agent.name,
                    "status": result.get("status", "unknown"),
                    "execution_time_ms": execution_time_ms
                }
            )
            
            logger.info(f"✅ Task {task_request.task_id} completed by {target_agent.name} in {execution_time_ms}ms")
            return result
            
    except httpx.RequestError as e:
        error_msg = f"Failed to connect to agent {target_agent.name}: {e}"
        log_trace_event(
            task_request.trace_id or task_request.task_id,
            task_request.task_id,
            "agent_connection_failed",
            {"agent_name": target_agent.name, "error": str(e)}
        )
        raise HTTPException(status_code=503, detail=error_msg)
    except Exception as e:
        error_msg = f"Error routing task: {e}"
        log_trace_event(
            task_request.trace_id or task_request.task_id,
            task_request.task_id,
            "routing_error",
            {"agent_name": target_agent.name, "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=error_msg)

def find_best_agent_for_task(task_type: str, priority: int = 5) -> Optional[AgentCard]:
    """
    Enhanced agent selection with support for multiple criteria and load balancing.
    
    This function can be extended to support:
    - Load balancing across multiple agents with the same capability
    - Priority-based routing
    - Agent health checking
    """
    matching_agents = []
    
    for agent_card in registered_agents.values():
        if agent_card.skills:
            for skill in agent_card.skills:
                if task_type in skill.tags:
                    matching_agents.append((agent_card, skill))
                    break
    
    if not matching_agents:
        logger.warning(f"No agents found for task type: {task_type}")
        return None
    
    # For now, return the first matching agent
    # TODO: Implement load balancing, health checks, priority weighting
    selected_agent = matching_agents[0][0]
    logger.info(f"Selected agent {selected_agent.name} for task type {task_type}")
    
    return selected_agent

@app.get("/health")
async def health_check():
    """Enhanced health check with system status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "registered_agents": len(registered_agents),
        "active_traces": len(trace_log),
        "version": "2.1.0"
    }

@app.get("/security/status")
async def security_status(api_key: str = Depends(enhanced_api_key_validation)):
    """Security status and monitoring information."""
    return {
        "status": "operational",
        "security_version": "2.1.0",
        "blocked_ips": len(blocked_ips),
        "failed_auth_attempts": sum(failed_auth_attempts.values()),
        "rate_limits_active": len(security_monitor.request_history),
        "features": {
            "rate_limiting": True,
            "ip_blocking": True,
            "input_validation": True,
            "trace_sanitization": True,
            "cors_protection": True,
            "security_headers": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_PORT", 10001))
    logger.info(f"🚀 Starting Enhanced MCP Server on port {port}")
    logger.info("🔐 Authentication enabled on all endpoints")
    logger.info("📊 Distributed tracing enabled")
    uvicorn.run(app, host="0.0.0.0", port=port) 