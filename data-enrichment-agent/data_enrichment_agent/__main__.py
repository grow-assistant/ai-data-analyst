#!/usr/bin/env python3
"""
Data Enrichment Agent - A2A Server Implementation
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import DATA_ENRICHMENT_AGENT_CARD
from .agent_executor import DataEnrichmentAgentExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "enrich_dataset": self.enrich_dataset
        }
    
    async def enrich_dataset(self, **kwargs):
        """A2A skill wrapper."""
        return await self.executor.enrich_dataset_skill(**kwargs)

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = DataEnrichmentAgentExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Data Enrichment Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "data_enrichment"}
    
    @app.post("/")
    async def handle_request(request: dict):
        """Handle A2A JSON-RPC requests."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "enrich_dataset":
                result = await agent.enrich_dataset(**params)
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
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id")
            }
    
    return app

def main():
    """Main entry point for the data enrichment agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🔄 Starting Data Enrichment Agent A2A Server")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Data Enrichment Agent starting on port 10009")
    uvicorn.run(app, host="0.0.0.0", port=10009)

if __name__ == "__main__":
    main()
