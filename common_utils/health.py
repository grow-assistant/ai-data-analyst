"""Health check utilities for multi-agent system."""

import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgentHealthStatus:
    """Health status for an individual agent."""
    name: str
    port: int
    url: str
    is_healthy: bool
    response_time_ms: Optional[float]
    error: Optional[str]
    details: Dict[str, any]

class HealthChecker:
    """Multi-agent health checker."""
    
    def __init__(self):
        self.default_agents = [
            ("orchestrator_agent", 10000),
            ("data_loader_agent", 10006),
            ("data_analyst_agent", 10007),
        ]
    
    async def check_agent_health(self, name: str, port: int, timeout: float = 5.0) -> AgentHealthStatus:
        """Check health of a single agent."""
        url = f"http://localhost:{port}"
        health_url = f"{url}/health"
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Try health endpoint first
                try:
                    response = await client.get(health_url)
                    response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        return AgentHealthStatus(
                            name=name,
                            port=port,
                            url=url,
                            is_healthy=True,
                            response_time_ms=response_time,
                            error=None,
                            details=health_data
                        )
                    else:
                        return AgentHealthStatus(
                            name=name,
                            port=port,
                            url=url,
                            is_healthy=False,
                            response_time_ms=response_time,
                            error=f"HTTP {response.status_code}",
                            details={"status_code": response.status_code}
                        )
                        
                except httpx.RequestError as e:
                    # Try well-known endpoint as fallback
                    try:
                        response = await client.get(f"{url}/.well-known/agent.json")
                        response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                        
                        if response.status_code == 200:
                            agent_data = response.json()
                            return AgentHealthStatus(
                                name=name,
                                port=port,
                                url=url,
                                is_healthy=True,
                                response_time_ms=response_time,
                                error="Health endpoint unavailable, using well-known",
                                details=agent_data
                            )
                    except Exception:
                        pass
                    
                    return AgentHealthStatus(
                        name=name,
                        port=port,
                        url=url,
                        is_healthy=False,
                        response_time_ms=None,
                        error=str(e),
                        details={}
                    )
                    
        except Exception as e:
            return AgentHealthStatus(
                name=name,
                port=port,
                url=url,
                is_healthy=False,
                response_time_ms=None,
                error=str(e),
                details={}
            )
    
    async def check_all_agents(self, agents: Optional[List[Tuple[str, int]]] = None) -> List[AgentHealthStatus]:
        """Check health of all agents concurrently."""
        if agents is None:
            agents = self.default_agents
        
        tasks = []
        for name, port in agents:
            task = self.check_agent_health(name, port)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def print_health_report(self, statuses: List[AgentHealthStatus]) -> None:
        """Print a formatted health report."""
        print("\n" + "="*60)
        print("🏥 AGENT HEALTH REPORT")
        print("="*60)
        
        healthy_count = sum(1 for status in statuses if status.is_healthy)
        total_count = len(statuses)
        
        print(f"Overall Status: {healthy_count}/{total_count} agents healthy")
        print()
        
        for status in statuses:
            status_icon = "✅" if status.is_healthy else "❌"
            response_time = f"{status.response_time_ms:.1f}ms" if status.response_time_ms else "N/A"
            
            print(f"{status_icon} {status.name:<20} Port: {status.port:<5} Response: {response_time:<8}")
            print(f"   URL: {status.url}")
            
            if status.error:
                print(f"   Error: {status.error}")
            
            if status.is_healthy and status.details:
                agent_name = status.details.get('agent', 'Unknown')
                print(f"   Agent: {agent_name}")
            
            print()
        
        print("="*60)

async def quick_health_check() -> bool:
    """Quick health check returning True if all agents are healthy."""
    checker = HealthChecker()
    statuses = await checker.check_all_agents()
    return all(status.is_healthy for status in statuses)

async def detailed_health_check() -> None:
    """Detailed health check with full report."""
    checker = HealthChecker()
    statuses = await checker.check_all_agents()
    checker.print_health_report(statuses)

if __name__ == "__main__":
    asyncio.run(detailed_health_check()) 