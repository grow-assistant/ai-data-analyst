#!/usr/bin/env python3
"""
Enhanced RootCause Analyst Agent (Why-Bot) - A2A Server Implementation
Features Google Gemini AI-powered hypothesis generation and comprehensive root cause analysis.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from .agent_executor import EnhancedRootCauseAnalystExecutor

logger = logging.getLogger(__name__)

class SimpleA2AAgent:
    """Simple A2A agent wrapper for our executor."""
    
    def __init__(self, executor):
        self.executor = executor
        self.skills = {
            "investigate_trend": self.investigate_trend,
            "root_cause_analysis": self.root_cause_analysis  # Alias for clarity
        }
    
    async def investigate_trend(self, **kwargs):
        """A2A skill wrapper for trend investigation and root cause analysis."""
        return await self.executor.investigate_trend_skill(**kwargs)
    
    async def root_cause_analysis(self, **kwargs):
        """A2A skill alias for root cause analysis."""
        return await self.executor.investigate_trend_skill(**kwargs)

def create_app():
    """Create the FastAPI application."""
    # Create our executor and wrap it
    executor = EnhancedRootCauseAnalystExecutor()
    agent = SimpleA2AAgent(executor)

    # Start a simple HTTP server for now
    from fastapi import FastAPI
    
    app = FastAPI(title="Enhanced RootCause Analyst Agent (Why-Bot)")
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy", 
            "agent": "rootcause_analyst", 
            "version": "enhanced_v2",
            "ai_capabilities": "google_gemini",
            "features": [
                "ai_hypothesis_generation",
                "statistical_testing",
                "variance_decomposition", 
                "causal_inference",
                "confidence_scoring",
                "escalation_logic"
            ]
        }
    
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
    """Main entry point for the enhanced rootcause analyst agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🔍 Starting Enhanced RootCause Analyst Agent (Why-Bot)")
    logger.info("🤖 Google Gemini AI-powered hypothesis generation")
    logger.info("📊 Comprehensive statistical testing and causal inference")
    logger.info("💡 Automated root cause discovery with confidence scoring")

    app = create_app()
    
    import uvicorn
    logger.info("🚀 Enhanced RootCause Analyst Agent starting on port 10011")
    uvicorn.run(app, host="0.0.0.0", port=10011)

if __name__ == "__main__":
    main() 