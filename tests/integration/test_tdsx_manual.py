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

async def wait_for_port(port: int, service_name: str, host: str = 'localhost', timeout: float = 30.0):
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
async def test_tdsx_loading_through_mcp():
    """Test TDSX loading through MCP routing - manual startup"""
    
    # Start MCP Server
    print("Starting MCP Server...")
    mcp_cmd = [sys.executable, "mcp_server/server.py"]
    mcp_process = subprocess.Popen(
        mcp_cmd, 
        cwd=str(PROJECT_ROOT / 'data-loader-agent'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for MCP to start
        started = await wait_for_port(MCP_PORT, "MCP Server")
        assert started, "MCP Server failed to start"
        
        # Give it a moment to fully initialize
        await asyncio.sleep(2)
        
        # Test the MCP routing with TDSX request
        async with httpx.AsyncClient(timeout=60.0) as client:
            task_request = {
                "task_id": "manual-tdsx-test-123",
                "trace_id": "manual-trace-123",
                "task_type": "loading",
                "parameters": {
                    "query": "Load AI_DS.tdsx file",
                    "description": "Test TDSX loading via MCP"
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
                assert 'status' in result or 'result' in result
                
                # Check for TDSX-specific content if available
                if 'result' in result and isinstance(result['result'], dict):
                    content = result['result'].get('content', '')
                    if content:
                        print(f"Content received: {content}")
                        # Should mention TDSX or the file
                        assert 'tdsx' in content.lower() or 'ai_ds' in content.lower()
                        print("✅ TDSX loading through MCP successful!")
                    else:
                        print("⚠️ No content in MCP response")
                else:
                    print("⚠️ Unexpected MCP response format")
                    
            else:
                print(f"❌ MCP request failed: {response.text}")
                assert False, f"MCP request failed with status {response.status_code}"
                
    finally:
        # Clean up
        print("Terminating MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing MCP server...")
            mcp_process.kill()
            mcp_process.wait()
        
        # Print process output for debugging
        stdout, stderr = mcp_process.communicate()
        if stderr:
            print("MCP Server stderr:")
            print(stderr.decode()[:1000])  # First 1000 chars

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 