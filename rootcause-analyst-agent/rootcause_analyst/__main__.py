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

from common_utils import create_agent_server
from .agent_executor import EnhancedRootCauseAnalystExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the enhanced rootcause analyst agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🔍 Starting Enhanced RootCause Analyst Agent (Why-Bot)")
    logger.info("🤖 Google Gemini AI-powered hypothesis generation")
    logger.info("📊 Comprehensive statistical testing and causal inference")
    logger.info("💡 Automated root cause discovery with confidence scoring")

    # Create executor
    executor = EnhancedRootCauseAnalystExecutor()
    
    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="rootcause_analyst",
        title="Enhanced RootCause Analyst Agent (Why-Bot)",
        port=10011,
        agent_description="AI-powered root cause analysis with Google Gemini hypothesis generation",
        version="enhanced_v2.0",
        custom_health_data={
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
    )
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main() 