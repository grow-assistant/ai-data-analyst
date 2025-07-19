import pytest
import asyncio
import subprocess
import time
import socket
import httpx
import os
import signal
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup test environment
from tests.setup_test_env import setup_test_environment
setup_test_environment()

# Ports from config
MCP_PORT = 10001
ORCHESTRATOR_PORT = 10000
DATA_LOADER_PORT = 10006
DATA_ANALYST_PORT = 10007

class AgentTestRunner:
    def __init__(self):
        self.processes = {}
        
    async def start_all_agents(self):
        """Start all agents in the correct order"""
        try:
            # Start MCP Server first
            print("Starting MCP Server...")
            mcp_cmd = [sys.executable, "mcp_server/server.py"]
            self.processes['mcp'] = subprocess.Popen(
                mcp_cmd, 
                cwd=str(PROJECT_ROOT / 'data-loader-agent'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await self.wait_for_port(MCP_PORT, "MCP Server")
            
            # Start Data Loader Agent
            print("Starting Data Loader Agent...")
            loader_cmd = [sys.executable, "-m", "data_loader"]
            self.processes['loader'] = subprocess.Popen(
                loader_cmd, 
                cwd=str(PROJECT_ROOT / 'data-loader-agent'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await self.wait_for_port(DATA_LOADER_PORT, "Data Loader Agent")
            
            # Start Data Analyst Agent
            print("Starting Data Analyst Agent...")
            analyst_cmd = [sys.executable, "-m", "data_analyst"]
            self.processes['analyst'] = subprocess.Popen(
                analyst_cmd, 
                cwd=str(PROJECT_ROOT / 'data-analyst-agent'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await self.wait_for_port(DATA_ANALYST_PORT, "Data Analyst Agent")
            
            # Start Orchestrator Agent
            print("Starting Orchestrator Agent...")
            orch_cmd = [sys.executable, "-m", "orchestrator_agent"]
            self.processes['orch'] = subprocess.Popen(
                orch_cmd, 
                cwd=str(PROJECT_ROOT / 'orchestrator-agent'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await self.wait_for_port(ORCHESTRATOR_PORT, "Orchestrator Agent")
            
            print("All agents started successfully!")
            
        except Exception as e:
            print(f"Error starting agents: {e}")
            await self.cleanup()
            raise
    
    async def wait_for_port(self, port: int, service_name: str, host: str = 'localhost', timeout: float = 60.0):
        """Wait for a port to become available"""
        print(f"Waiting for {service_name} on port {port}...")
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=2):
                    print(f"{service_name} is ready on port {port}")
                    return
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(1)
        raise TimeoutError(f"{service_name} not available on port {port} after {timeout} seconds")
    
    async def cleanup(self):
        """Stop all running processes"""
        print("Cleaning up processes...")
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                print(f"Terminating {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    proc.kill()
                    proc.wait()

# Configure pytest properly for async tests
pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    policy = asyncio.WindowsProactorEventLoopPolicy() if os.name == 'nt' else asyncio.DefaultEventLoopPolicy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def agent_runner():
    """Start all agents and provide cleanup"""
    runner = AgentTestRunner()
    await runner.start_all_agents()
    yield runner
    await runner.cleanup()

@pytest.mark.asyncio
async def test_mcp_server_health(agent_runner):
    """Test that MCP server is responding"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"http://localhost:{MCP_PORT}/health")
        assert response.status_code == 200
        result = response.json()
        assert result.get('status') == 'healthy'

@pytest.mark.asyncio
async def test_orchestrator_health(agent_runner):
    """Test that orchestrator is responding"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"http://localhost:{ORCHESTRATOR_PORT}/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_data_loader_health(agent_runner):
    """Test that data loader is responding"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"http://localhost:{DATA_LOADER_PORT}/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_data_analyst_health(agent_runner):
    """Test that data analyst is responding"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"http://localhost:{DATA_ANALYST_PORT}/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_mcp_routing_simple(agent_runner):
    """Test basic MCP routing functionality"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test task routing through MCP
        task_request = {
            "task_id": "test-routing-123",
            "trace_id": "test-trace-123",
            "task_type": "loading",
            "parameters": {
                "query": "Test data loading capability",
                "description": "Simple routing test"
            },
            "data_handles": [],
            "priority": 5
        }
        
        response = await client.post(
            f"http://localhost:{MCP_PORT}/route",
            json=task_request,
            headers={"X-API-Key": "mcp-dev-key"}
        )
        
        assert response.status_code == 200
        result = response.json()
        # Adjust assertions based on actual response structure
        assert 'status' in result or 'result' in result

@pytest.mark.asyncio
async def test_orchestrator_workflow(agent_runner):
    """Test orchestrator coordination workflow"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Send a complex request to orchestrator
        request_data = {
            "task_id": "orchestrator-test-123",
            "message": {
                "content": "Load test data and perform basic analysis",
                "role": "user"
            }
        }
        
        response = await client.post(
            f"http://localhost:{ORCHESTRATOR_PORT}/execute",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        # The response should indicate successful coordination
        assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_agent_registration(agent_runner):
    """Test that agents are properly registered with MCP"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"http://localhost:{MCP_PORT}/agents",
            headers={"X-API-Key": "mcp-dev-key"}
        )
        
        assert response.status_code == 200
        agents = response.json()
        
        # Should have registered agents
        assert len(agents) > 0
        
        # Check that key agents are registered
        agent_names = [agent.get('name', '') for agent in agents]
        expected_agents = ['data_loader', 'data_analyst', 'orchestrator']
        
        for expected in expected_agents:
            assert any(expected in name.lower() for name in agent_names), f"Agent {expected} not found in registered agents"

if __name__ == "__main__":
    # Allow running the test directly for debugging
    pytest.main([__file__, "-v", "-s"]) 