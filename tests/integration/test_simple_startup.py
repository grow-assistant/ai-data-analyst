import pytest
import asyncio
import subprocess
import time
import socket
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup test environment
from tests.setup_test_env import setup_test_environment
setup_test_environment()

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
async def test_mcp_server_startup():
    """Test that we can start the MCP server"""
    print("Testing MCP server startup...")
    
    # Start MCP Server
    mcp_cmd = [sys.executable, "mcp_server/server.py"]
    print(f"Starting MCP server with command: {' '.join(mcp_cmd)}")
    print(f"Working directory: {PROJECT_ROOT / 'data-loader-agent'}")
    
    process = subprocess.Popen(
        mcp_cmd, 
        cwd=str(PROJECT_ROOT / 'data-loader-agent'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for it to start
        started = await wait_for_port(MCP_PORT, "MCP Server")
        assert started, "MCP Server failed to start"
        
        print("MCP Server started successfully!")
        
    finally:
        # Clean up
        print("Terminating MCP server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing MCP server...")
            process.kill()
            process.wait()
        
        # Print output for debugging
        stdout, stderr = process.communicate()
        if stdout:
            print("STDOUT:", stdout.decode())
        if stderr:
            print("STDERR:", stderr.decode())

if __name__ == "__main__":
    # Allow running the test directly for debugging
    pytest.main([__file__, "-v", "-s"]) 