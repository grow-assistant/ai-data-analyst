# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Health Check Dashboard for Multi-Agent System."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import common utils
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from common_utils.config import settings
    from common_utils.circuit_breaker import CircuitBreakerException
except ImportError:
    # Fallback configuration
    class FallbackSettings:
        a2a_host = "localhost"
        a2a_port = 10000
        data_loader_port = 10006
        data_analyst_port = 10007
        mcp_port = 10001
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class AgentHealth:
    name: str
    url: str
    status: HealthStatus
    response_time_ms: Optional[float]
    last_check: datetime
    circuit_breakers: Dict[str, Any]
    capabilities: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SystemHealth:
    overall_status: HealthStatus
    agents: List[AgentHealth]
    last_updated: datetime
    healthy_agents: int
    total_agents: int

class HealthDashboard:
    """Unified health monitoring dashboard for all agents."""
    
    def __init__(self):
        self.app = FastAPI(title="Multi-Agent Health Dashboard")
        self.client = httpx.AsyncClient(timeout=10.0)
        self.health_history: Dict[str, List[AgentHealth]] = {}
        self.check_interval = 30  # seconds
        
        # Define agent endpoints
        self.agents = {
                    "orchestrator": f"http://localhost:{settings.orchestrator_port}",
        "data_loader": f"http://localhost:{settings.data_loader_port}",
        "data_analyst": f"http://localhost:{settings.data_analyst_port}",
        "mcp_server": settings.mcp_server_url
        }
        
        self.setup_routes()
        logger.info("Health dashboard initialized")
    
    def setup_routes(self):
        """Setup FastAPI routes for the dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard HTML."""
            return self.generate_dashboard_html()
        
        @self.app.get("/api/health")
        async def get_system_health():
            """Get current system health status."""
            system_health = await self.check_all_agents()
            return asdict(system_health)
        
        @self.app.get("/api/health/history/{agent_name}")
        async def get_agent_history(agent_name: str):
            """Get health history for a specific agent."""
            if agent_name not in self.health_history:
                raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
            
            # Return last 24 hours of data
            cutoff = datetime.now() - timedelta(hours=24)
            recent_history = [
                asdict(health) for health in self.health_history[agent_name]
                if health.last_check > cutoff
            ]
            return {"agent": agent_name, "history": recent_history}
        
        @self.app.get("/api/health/check")
        async def force_health_check():
            """Force an immediate health check of all agents."""
            system_health = await self.check_all_agents()
            return {"message": "Health check completed", "system": asdict(system_health)}
    
    async def check_agent_health(self, name: str, url: str) -> AgentHealth:
        """Check health of a specific agent."""
        start_time = time.time()
        
        try:
            # Try detailed health endpoint first
            response = await self.client.get(f"{url}/health/detailed")
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                status = HealthStatus.HEALTHY
                circuit_breakers = data.get("circuit_breakers", {})
                capabilities = data.get("capabilities", {})
                
                # Check if any circuit breakers are open
                if circuit_breakers and any(cb.get("state") == "OPEN" for cb in circuit_breakers.values()):
                    status = HealthStatus.DEGRADED
                
                return AgentHealth(
                    name=name,
                    url=url,
                    status=status,
                    response_time_ms=response_time,
                    last_check=datetime.now(),
                    circuit_breakers=circuit_breakers,
                    capabilities=capabilities
                )
            else:
                # Try basic health endpoint
                basic_response = await self.client.get(f"{url}/health")
                if basic_response.status_code == 200:
                    return AgentHealth(
                        name=name,
                        url=url,
                        status=HealthStatus.DEGRADED,
                        response_time_ms=response_time,
                        last_check=datetime.now(),
                        circuit_breakers={},
                        capabilities={},
                        error_message="Detailed health endpoint unavailable"
                    )
                else:
                    raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
        
        except httpx.TimeoutException:
            return AgentHealth(
                name=name,
                url=url,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.now(),
                circuit_breakers={},
                capabilities={},
                error_message="Request timeout"
            )
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return AgentHealth(
                name=name,
                url=url,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=None,
                last_check=datetime.now(),
                circuit_breakers={},
                capabilities={},
                error_message=str(e)
            )
    
    async def check_all_agents(self) -> SystemHealth:
        """Check health of all agents and return system status."""
        agent_healths = []
        
        # Check all agents concurrently
        tasks = [
            self.check_agent_health(name, url)
            for name, url in self.agents.items()
        ]
        
        agent_healths = await asyncio.gather(*tasks)
        
        # Store history
        for health in agent_healths:
            if health.name not in self.health_history:
                self.health_history[health.name] = []
            
            self.health_history[health.name].append(health)
            
            # Keep only last 1000 entries per agent
            if len(self.health_history[health.name]) > 1000:
                self.health_history[health.name] = self.health_history[health.name][-1000:]
        
        # Determine overall system status
        healthy_count = sum(1 for h in agent_healths if h.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for h in agent_healths if h.status == HealthStatus.DEGRADED)
        
        if healthy_count == len(agent_healths):
            overall_status = HealthStatus.HEALTHY
        elif healthy_count + degraded_count == len(agent_healths):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        return SystemHealth(
            overall_status=overall_status,
            agents=agent_healths,
            last_updated=datetime.now(),
            healthy_agents=healthy_count,
            total_agents=len(agent_healths)
        )
    
    def generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Health Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { background: white; border-radius: 8px; padding: 20px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-healthy { border-left: 5px solid #4CAF50; }
        .status-degraded { border-left: 5px solid #FF9800; }
        .status-unhealthy { border-left: 5px solid #F44336; }
        .status-unknown { border-left: 5px solid #9E9E9E; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px; }
        .refresh-btn { background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #1976D2; }
        .timestamp { color: #666; font-size: 0.9em; }
        .error { color: #F44336; font-weight: bold; }
        .capabilities { margin-top: 10px; }
        .capability { display: inline-block; background: #E3F2FD; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Agent System Health Dashboard</h1>
            <button class="refresh-btn" onclick="refreshData()">Refresh Now</button>
            <p class="timestamp">Last updated: <span id="lastUpdated">Loading...</span></p>
        </div>
        
        <div id="systemOverview" class="status-card">
            <h2>System Overview</h2>
            <div id="overviewContent">Loading...</div>
        </div>
        
        <div class="grid" id="agentGrid">
            Loading agent status...
        </div>
    </div>

    <script>
        async function fetchHealthData() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to fetch health data:', error);
                document.getElementById('overviewContent').innerHTML = '<p class="error">Failed to load health data</p>';
            }
        }

        function updateDashboard(data) {
            // Update timestamp
            document.getElementById('lastUpdated').textContent = new Date(data.last_updated).toLocaleString();
            
            // Update system overview
            const overviewContent = document.getElementById('overviewContent');
            overviewContent.innerHTML = `
                <div class="metric">Overall Status: <strong class="status-${data.overall_status}">${data.overall_status.toUpperCase()}</strong></div>
                <div class="metric">Healthy Agents: ${data.healthy_agents}/${data.total_agents}</div>
            `;
            
            // Update agent grid
            const agentGrid = document.getElementById('agentGrid');
            agentGrid.innerHTML = data.agents.map(agent => `
                <div class="status-card status-${agent.status}">
                    <h3>${agent.name}</h3>
                    <div class="metric">Status: <strong>${agent.status.toUpperCase()}</strong></div>
                    <div class="metric">URL: ${agent.url}</div>
                    ${agent.response_time_ms ? `<div class="metric">Response Time: ${agent.response_time_ms.toFixed(2)}ms</div>` : ''}
                    <div class="metric">Last Check: ${new Date(agent.last_check).toLocaleString()}</div>
                    ${agent.error_message ? `<p class="error">Error: ${agent.error_message}</p>` : ''}
                    
                    ${Object.keys(agent.capabilities).length > 0 ? `
                        <div class="capabilities">
                            <strong>Capabilities:</strong><br>
                            ${Object.entries(agent.capabilities).map(([key, value]) => 
                                `<span class="capability">${key}: ${value}</span>`
                            ).join('')}
                        </div>
                    ` : ''}
                    
                    ${Object.keys(agent.circuit_breakers).length > 0 ? `
                        <div class="capabilities">
                            <strong>Circuit Breakers:</strong><br>
                            ${Object.entries(agent.circuit_breakers).map(([key, cb]) => 
                                `<span class="capability ${cb.state === 'OPEN' ? 'error' : ''}">${key}: ${cb.state}</span>`
                            ).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }

        function refreshData() {
            fetchHealthData();
        }

        // Initial load and auto-refresh every 30 seconds
        fetchHealthData();
        setInterval(fetchHealthData, 30000);
    </script>
</body>
</html>
        """
    
    async def start_background_monitoring(self):
        """Start background health monitoring."""
        logger.info("Starting background health monitoring")
        while True:
            try:
                await self.check_all_agents()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the health dashboard server."""
        # Start background monitoring
        monitor_task = asyncio.create_task(self.start_background_monitoring())
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            logger.info(f"Starting health dashboard on {host}:{port}")
            await server.serve()
        finally:
            monitor_task.cancel()
            await self.client.aclose()

# CLI entry point
async def main():
    dashboard = HealthDashboard()
    await dashboard.run()

if __name__ == "__main__":
    asyncio.run(main()) 