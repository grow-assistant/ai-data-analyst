#!/usr/bin/env python3
"""
Enhanced Presentation Agent - A2A Server Implementation with Google Gemini Executive Reporting
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import PRESENTATION_AGENT_CARD
from .agent_executor import EnhancedPresentationExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "create_executive_report": self.create_executive_report,
            "create_report": self.create_report  # Backward compatibility
        }
    
    async def create_executive_report(self, **kwargs):
        """A2A skill wrapper for executive report generation."""
        return await self.executor.create_executive_report_skill(**kwargs)
    
    async def create_report(self, **kwargs):
        """A2A skill wrapper for backward compatibility."""
        return await self.executor.create_report_skill(**kwargs)

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = EnhancedPresentationExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Enhanced Presentation Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "presentation", "version": "enhanced", "gemini_ai_support": True}
    
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
    """Main entry point for the enhanced presentation agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📄 Starting Enhanced Presentation Agent A2A Server")
    logger.info("🤖 Google Gemini AI executive reporting capabilities enabled")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Enhanced Presentation Agent starting on port 10010")
    uvicorn.run(app, host="0.0.0.0", port=10010)

if __name__ == "__main__":
    main()
