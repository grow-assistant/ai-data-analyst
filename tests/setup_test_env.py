#!/usr/bin/env python3
"""
Setup script for integration tests.
Sets required environment variables for testing.
"""

import os
import sys

def setup_test_environment():
    """Set up the test environment with required variables."""
    
    # Set required environment variables for testing
    test_env = {
        'GOOGLE_API_KEY': 'test-key-for-integration-tests',
        'MCP_API_KEY': 'mcp-dev-key', 
        'LOG_LEVEL': 'INFO',
        'ENABLE_SAFETY_CHECKS': 'true',
        'MCP_SERVER_URL': 'http://localhost:10001',
        'DATA_LOADER_PORT': '10006',
        'DATA_ANALYST_PORT': '10007',
        'ORCHESTRATOR_PORT': '10000',
        'MCP_PORT': '10001'
    }
    
    for key, value in test_env.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")
        else:
            print(f"Using existing {key}={os.environ[key]}")

if __name__ == "__main__":
    setup_test_environment()
    print("Test environment setup complete!") 