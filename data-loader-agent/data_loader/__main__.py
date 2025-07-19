#!/usr/bin/env python3
"""
Enhanced Data Loader Agent - A2A Server Implementation with Tableau Hyper API Support
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import DATA_LOADER_AGENT_CARD
from .agent_executor import EnhancedDataLoaderExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "load_data": self.load_data,
            "load_dataset": self.load_dataset  # Backward compatibility
        }
    
    async def load_data(self, **kwargs):
        """A2A skill wrapper for enhanced data loading."""
        return await self.executor.load_data_skill(**kwargs)
    
    async def load_dataset(self, **kwargs):
        """A2A skill wrapper for backward compatibility."""
        # Convert old parameters to new format
        file_path = kwargs.get('file_path')
        file_type = kwargs.get('file_type', 'auto')
        
        # Call new load_data_skill
        result = await self.executor.load_data_skill(file_path, file_type)
        
        # Convert response format for backward compatibility
        if result.get('status') == 'completed':
            return {
                "status": "completed",
                "data_handle_id": result["data_handle_id"],
                "metadata": result["metadata"],
                "message": f"Successfully loaded data using enhanced loader",
                "data_preview": []  # Legacy field
            }
        return result

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = EnhancedDataLoaderExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Enhanced Data Loader Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "data_loader", "version": "enhanced", "hyper_api_support": True}
    
    @app.post("/")
    async def handle_request(request: dict):
        """Handle A2A JSON-RPC requests."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method in agent.skills:
                result = await agent.skills[method](**params)
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
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
    """Main entry point for the enhanced data loader agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📁 Starting Enhanced Data Loader Agent A2A Server")
    logger.info("⚡ Tableau Hyper API support for blazing fast TDSX/Hyper file loading")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Enhanced Data Loader Agent starting on port 10006")
    uvicorn.run(app, host="0.0.0.0", port=10006)

if __name__ == "__main__":
    main() 