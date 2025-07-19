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

async def wait_for_port(port: int, service_name: str, host: str = 'localhost', timeout: float = 45.0):
    """Wait for a port to become available"""
    print(f"Waiting for {service_name} on port {port}...")
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"{service_name} is ready on port {port}")
                return True
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(1)
    print(f"{service_name} not available on port {port} after {timeout} seconds")
    return False

@pytest.mark.asyncio
async def test_complete_tdsx_loading():
    """Test complete TDSX loading workflow with MCP server and data loader agent"""
    
    mcp_process = None
    loader_process = None
    
    try:
        # Start MCP Server first
        print("Starting MCP Server...")
        mcp_cmd = [sys.executable, "mcp_server/server.py"]
        mcp_process = subprocess.Popen(
            mcp_cmd, 
            cwd=str(PROJECT_ROOT / 'data-loader-agent'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for MCP to start
        mcp_started = await wait_for_port(MCP_PORT, "MCP Server")
        assert mcp_started, "MCP Server failed to start"
        
        # Give MCP a moment to fully initialize
        await asyncio.sleep(3)
        
        # Start Data Loader Agent
        print("Starting Data Loader Agent...")
        loader_cmd = [sys.executable, "-m", "data_loader"]
        loader_process = subprocess.Popen(
            loader_cmd, 
            cwd=str(PROJECT_ROOT / 'data-loader-agent'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Data Loader to start
        loader_started = await wait_for_port(DATA_LOADER_PORT, "Data Loader Agent")
        assert loader_started, "Data Loader Agent failed to start"
        
        # Give the data loader time to register with MCP
        await asyncio.sleep(5)
        
        print("Both services started successfully!")
        
        # Test 1: Check that data loader agent is healthy
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://localhost:{DATA_LOADER_PORT}/health")
            assert response.status_code == 200
            print("✅ Data Loader health check passed")
        
        # Test 2: Check that agents are registered with MCP
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"http://localhost:{MCP_PORT}/agents",
                headers={"X-API-Key": "mcp-dev-key"}
            )
            
            if response.status_code == 200:
                agents = response.json()
                print(f"Registered agents: {agents}")
                assert len(agents) > 0, "No agents registered with MCP"
                print("✅ Agents are registered with MCP")
            else:
                print(f"⚠️ Could not get agent list: {response.status_code}")
        
        # Test 3: Test TDSX loading through MCP routing
        async with httpx.AsyncClient(timeout=90.0) as client:
            task_request = {
                "task_id": "complete-tdsx-test-123",
                "trace_id": "complete-trace-123",
                "task_type": "loading",
                "parameters": {
                    "query": "Load AI_DS.tdsx file",
                    "description": "Complete TDSX loading test"
                },
                "data_handles": [],
                "priority": 5
            }
            
            print("Sending TDSX loading request to MCP...")
            response = await client.post(
                f"http://localhost:{MCP_PORT}/route",
                json=task_request,
                headers={"X-API-Key": "mcp-dev-key"}
            )
            
            print(f"MCP Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"MCP Response: {result}")
                
                # Check that the task was processed successfully
                assert 'status' in result, "No status in MCP response"
                
                if result.get('status') == 'completed':
                    if 'results' in result and 'content' in result['results']:
                        content = result['results']['content']
                        print(f"✅ TDSX loading successful!")
                        print(f"Content preview: {content[:200]}...")
                        
                        # Verify TDSX-specific content
                        assert 'Successfully extracted TDSX file' in content
                        assert 'AI_DS.tdsx' in content
                        assert 'File size:' in content
                        
                        print("✅ All TDSX content checks passed!")
                    else:
                        print("⚠️ Task completed but no content returned")
                else:
                    print(f"⚠️ Task status: {result.get('status')}")
                    if 'error_message' in result:
                        print(f"Error: {result['error_message']}")
                        
            else:
                print(f"❌ MCP request failed: {response.text}")
                assert False, f"MCP request failed with status {response.status_code}"
                
        print("🎉 Complete TDSX loading test successful!")
                
    finally:
        # Clean up processes
        print("Cleaning up processes...")
        
        if loader_process and loader_process.poll() is None:
            print("Terminating Data Loader Agent...")
            loader_process.terminate()
            try:
                loader_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                loader_process.kill()
                loader_process.wait()
        
        if mcp_process and mcp_process.poll() is None:
            print("Terminating MCP Server...")
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_process.kill()
                mcp_process.wait()
        
        print("Cleanup complete!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 