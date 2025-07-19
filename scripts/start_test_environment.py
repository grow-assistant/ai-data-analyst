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

"""
Test Environment Startup Script

Starts all agents required for integration testing and waits for them to be healthy.
"""

import asyncio
import subprocess
import time
import httpx
import logging
from pathlib import Path
import sys
import signal
import os

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from common_utils.enhanced_logging import get_logger

logger = get_logger(__name__)

class TestEnvironmentManager:
    """Manages the test environment by starting and stopping agents."""
    
    def __init__(self):
        self.agent_processes = {}
        self.agent_configs = [
            {
                "name": "data_loader",
                "port": 10006,
                "module": "data_loader",
                "directory": "data-loader-agent"
            },
            {
                "name": "data_cleaning",
                "port": 10008,
                "module": "data_cleaning_agent",
                "directory": "data-cleaning-agent"
            },
            {
                "name": "data_enrichment",
                "port": 10009,
                "module": "data_enrichment_agent",
                "directory": "data-enrichment-agent"
            },
            {
                "name": "data_analyst",
                "port": 10007,
                "module": "data_analyst",
                "directory": "data-analyst-agent"
            },
            {
                "name": "presentation",
                "port": 10010,
                "module": "presentation_agent",
                "directory": "presentation-agent"
            },
            {
                "name": "rootcause_analyst",
                "port": 10011,
                "module": "rootcause_analyst",
                "directory": "rootcause-analyst-agent"
            },
            {
                "name": "schema_profiler",
                "port": 10012,
                "module": "schema_profiler",
                "directory": "schema-profiler-agent"
            }
        ]
        
    def start_agent(self, agent_config):
        """Start a single agent."""
        name = agent_config["name"]
        directory = agent_config["directory"]
        module = agent_config["module"]
        port = agent_config["port"]
        
        agent_path = parent_dir / directory
        
        if not agent_path.exists():
            logger.error(f"Agent directory not found: {agent_path}")
            return None
            
        logger.info(f"🚀 Starting {name} agent on port {port}")
        
        try:
            # Start the agent using Python module execution
            process = subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=str(agent_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.agent_processes[name] = process
            logger.info(f"✅ Started {name} agent (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start {name} agent: {e}")
            return None
    
    async def wait_for_agent_health(self, agent_config, max_attempts=30):
        """Wait for an agent to become healthy."""
        name = agent_config["name"]
        port = agent_config["port"]
        url = f"http://localhost:{port}/health"
        
        logger.info(f"🔍 Waiting for {name} agent to become healthy...")
        
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info(f"✅ {name} agent is healthy")
                        return True
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1}/{max_attempts} failed for {name}: {e}")
            
            await asyncio.sleep(2)
        
        logger.error(f"❌ {name} agent failed to become healthy after {max_attempts} attempts")
        return False
    
    async def start_all_agents(self):
        """Start all agents and wait for them to become healthy."""
        logger.info("🚀 Starting all agents for integration testing...")
        
        # Start all agents
        for agent_config in self.agent_configs:
            process = self.start_agent(agent_config)
            if not process:
                logger.error(f"Failed to start {agent_config['name']}")
                return False
        
        # Wait a bit for processes to initialize
        await asyncio.sleep(5)
        
        # Wait for all agents to become healthy
        logger.info("🔍 Waiting for all agents to become healthy...")
        
        health_checks = []
        for agent_config in self.agent_configs:
            health_checks.append(self.wait_for_agent_health(agent_config))
        
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        healthy_count = sum(1 for result in results if result is True)
        total_count = len(self.agent_configs)
        
        if healthy_count == total_count:
            logger.info(f"🎉 All {total_count} agents are healthy and ready!")
            return True
        else:
            logger.error(f"❌ Only {healthy_count}/{total_count} agents are healthy")
            return False
    
    def stop_all_agents(self):
        """Stop all running agents."""
        logger.info("🛑 Stopping all agents...")
        
        for name, process in self.agent_processes.items():
            try:
                logger.info(f"Stopping {name} agent (PID: {process.pid})")
                
                if os.name == 'nt':  # Windows
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:  # Unix/Linux
                    process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} agent stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {name} agent")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"Error stopping {name} agent: {e}")
        
        self.agent_processes.clear()
        logger.info("🛑 All agents stopped")
    
    async def run_integration_test(self):
        """Start agents and run integration test."""
        try:
            # Start agents
            if not await self.start_all_agents():
                logger.error("❌ Failed to start all agents. Cannot run integration test.")
                return False
            
            # Run the integration test
            logger.info("🧪 Running integration test...")
            
            test_script = parent_dir / "tests" / "integration" / "test_real_world_analysis.py"
            
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(parent_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("🎉 Integration test PASSED!")
                print(result.stdout)
                return True
            else:
                logger.error("💥 Integration test FAILED!")
                print("STDOUT:")
                print(result.stdout)
                print("STDERR:")
                print(result.stderr)
                return False
                
        finally:
            # Always stop agents
            self.stop_all_agents()

async def main():
    """Main function."""
    manager = TestEnvironmentManager()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal")
        manager.stop_all_agents()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = await manager.run_integration_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        manager.stop_all_agents()
        return 1
    except Exception as e:
        logger.exception(f"💥 Test environment failed: {e}")
        manager.stop_all_agents()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 