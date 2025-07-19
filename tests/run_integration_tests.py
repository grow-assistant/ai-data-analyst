#!/usr/bin/env python3
"""
Integration test runner for the ADK multi-agent system.
This script runs the integration tests that start actual agent processes.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_test_requirements():
    """Install test requirements if needed."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("Installing test requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
    else:
        print("Installing basic test dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "httpx"
        ])

def run_tests():
    """Run the integration tests."""
    test_dir = Path(__file__).parent
    
    # Change to project root for running tests
    project_root = test_dir.parent
    os.chdir(project_root)
    
    # Run pytest with specific options for integration tests
    pytest_args = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",  # Verbose output
        "-s",  # Don't capture output (so we can see print statements)
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--asyncio-mode=auto"  # Auto-detect async tests
    ]
    
    print("Running integration tests...")
    print(f"Command: {' '.join(pytest_args)}")
    print(f"Working directory: {os.getcwd()}")
    
    result = subprocess.run(pytest_args)
    return result.returncode

if __name__ == "__main__":
    # Install requirements first
    try:
        install_test_requirements()
    except subprocess.CalledProcessError as e:
        print(f"Failed to install test requirements: {e}")
        sys.exit(1)
    
    # Run the tests
    exit_code = run_tests()
    sys.exit(exit_code) 