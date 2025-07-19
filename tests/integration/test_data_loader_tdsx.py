import pytest
import asyncio
import subprocess
import time
import socket
import httpx
import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup test environment
from tests.setup_test_env import setup_test_environment
setup_test_environment()

# Ports from config
MCP_PORT = 10001
DATA_LOADER_PORT = 10006

class TdsxTestRunner:
    def __init__(self):
        self.processes = {}
        
    async def start_required_agents(self):
        """Start MCP server and Data Loader agent"""
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
            
            print("Required agents started successfully!")
            
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

@pytest.fixture(scope="module")
async def tdsx_runner():
    """Start required agents for TDSX testing"""
    runner = TdsxTestRunner()
    await runner.start_required_agents()
    yield runner
    await runner.cleanup()

@pytest.mark.asyncio
async def test_tdsx_file_exists():
    """Test that the AI_DS.tdsx file exists"""
    tdsx_path = PROJECT_ROOT / 'data-loader-agent' / 'data' / 'AI_DS.tdsx'
    assert tdsx_path.exists(), f"TDSX file not found at {tdsx_path}"
    print(f"✅ TDSX file found at {tdsx_path}")

@pytest.mark.asyncio
async def test_data_loader_health(tdsx_runner):
    """Test that data loader agent is responding"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"http://localhost:{DATA_LOADER_PORT}/health")
        assert response.status_code == 200
        print("✅ Data Loader agent health check passed")

@pytest.mark.asyncio
async def test_load_tdsx_via_mcp_routing(tdsx_runner):
    """Test loading TDSX file through MCP routing"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test task routing through MCP for TDSX loading
        task_request = {
            "task_id": "test-tdsx-load-123",
            "trace_id": "test-trace-tdsx-123",
            "task_type": "loading",
            "parameters": {
                "query": "Load AI_DS.tdsx file",
                "description": "Load the Tableau data source file for testing"
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
        print(f"MCP routing response: {result}")
        
        # Check that the task was processed
        assert 'status' in result or 'result' in result
        
        # If there's content in the result, check for TDSX processing
        if 'result' in result and 'content' in result['result']:
            content = result['result']['content']
            assert 'tdsx' in content.lower() or 'tableau' in content.lower()
            print(f"✅ TDSX loading processed: {content}")

@pytest.mark.asyncio
async def test_load_tdsx_direct_agent(tdsx_runner):
    """Test loading TDSX file directly through the data loader agent"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Send request directly to data loader agent
        request_data = {
            "task_id": "direct-tdsx-test-123",
            "query": "Load the AI_DS.tdsx file from the data directory",
            "session_id": "test-session"
        }
        
        response = await client.post(
            f"http://localhost:{DATA_LOADER_PORT}/invoke",
            json=request_data
        )
        
        print(f"Direct agent response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Direct agent response: {result}")
            # Check for successful processing
            assert isinstance(result, dict)
        else:
            # Print error details for debugging
            print(f"Error response: {response.text}")
            assert False, f"Direct agent call failed with status {response.status_code}"

@pytest.mark.asyncio
async def test_load_tdsx_with_full_path(tdsx_runner):
    """Test loading TDSX file with full file path"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get the full path to the TDSX file
        tdsx_path = str(PROJECT_ROOT / 'data-loader-agent' / 'data' / 'AI_DS.tdsx')
        
        task_request = {
            "task_id": "test-tdsx-fullpath-123",
            "trace_id": "test-trace-fullpath-123",
            "task_type": "loading",
            "parameters": {
                "query": f"Load TDSX file from {tdsx_path}",
                "description": "Load TDSX with full path"
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
        print(f"Full path TDSX loading response: {result}")

if __name__ == "__main__":
    # Allow running the test directly for debugging
    pytest.main([__file__, "-v", "-s"]) 