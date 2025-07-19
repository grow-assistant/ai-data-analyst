#!/usr/bin/env python3
"""
Orchestrator Agent - A2A Server Implementation
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import ORCHESTRATOR_AGENT_CARD
from .agent_executor import OrchestratorAgentExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "orchestrate_pipeline": self.orchestrate_pipeline,
            "process_tdsx_workflow": self.process_tdsx_workflow
        }
    
    async def orchestrate_pipeline(self, **kwargs):
        """A2A skill wrapper for pipeline orchestration."""
        return await self.executor.orchestrate_pipeline_skill(**kwargs)
    
    async def process_tdsx_workflow(self, **kwargs):
        """A2A skill wrapper for TDSX workflow processing."""
        return await self.executor.process_tdsx_workflow_skill(**kwargs)

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = OrchestratorAgentExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Orchestrator Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "orchestrator"}
    
    @app.get("/capabilities")
    async def capabilities():
        return {
            "skills": list(agent.skills.keys()),
            "description": "Multi-agent workflow orchestration"
        }
    
    @app.post("/")
    async def handle_request(request: dict):
        """Handle A2A JSON-RPC requests."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "orchestrate_pipeline":
                result = await agent.orchestrate_pipeline(**params)
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            elif method == "process_tdsx_workflow":
                result = await agent.process_tdsx_workflow(**params)
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": request_id
                }
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id")
            }
    
    return app

def main():
    """Main entry point for the orchestrator agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🎯 Starting Orchestrator Agent A2A Server")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Orchestrator Agent starting on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 