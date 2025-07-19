#!/usr/bin/env python3
"""
Enhanced Data Analyst Agent - A2A Server Implementation
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import DATA_ANALYST_AGENT_CARD
from .agent_executor import EnhancedDataAnalystExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "comprehensive_analysis": self.comprehensive_analysis,
            "analyze_dataset": self.analyze_dataset  # Backward compatibility
        }
    
    async def comprehensive_analysis(self, **kwargs):
        """A2A skill wrapper for comprehensive analysis."""
        return await self.executor.comprehensive_analysis_skill(**kwargs)
    
    async def analyze_dataset(self, **kwargs):
        """A2A skill wrapper for backward compatibility."""
        # Convert old parameters to new format
        analysis_config = {}
        if 'analysis_type' in kwargs:
            analysis_config['analysis_type'] = kwargs['analysis_type']
        
        # Call comprehensive analysis with config
        result = await self.executor.comprehensive_analysis_skill(
            kwargs.get('data_handle_id'), 
            analysis_config
        )
        
        # Return in old format for compatibility
        if result.get('status') == 'completed':
            return {
                "status": "completed",
                "analysis_data_handle_id": result["analysis_data_handle_id"],
                "results": {"summary": "Analysis completed using enhanced comprehensive analysis"}
            }
        return result

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = EnhancedDataAnalystExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Enhanced Data Analyst Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "data_analyst", "version": "enhanced"}
    
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
    """Main entry point for the enhanced data analyst agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📊 Starting Enhanced Data Analyst Agent A2A Server")
    logger.info("🔬 Analysis capabilities: comprehensive business intelligence suite")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Enhanced Data Analyst Agent starting on port 10007")
    uvicorn.run(app, host="0.0.0.0", port=10007)

if __name__ == "__main__":
    main() 