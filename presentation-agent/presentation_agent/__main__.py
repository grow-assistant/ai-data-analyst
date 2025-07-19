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

from common_utils import create_agent_server
from .agent_executor import EnhancedPresentationExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the enhanced presentation agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📄 Starting Enhanced Presentation Agent A2A Server")
    logger.info("🤖 Google Gemini AI executive reporting capabilities enabled")

    # Create executor
    executor = EnhancedPresentationExecutor()
    
    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="presentation",
        title="Enhanced Presentation Agent",
        port=10010,
        agent_description="AI-powered executive reporting with Google Gemini integration",
        version="enhanced_v2.0",
        custom_health_data={
            "gemini_ai_support": True,
            "report_types": ["executive", "summary", "dashboard"],
            "ai_features": ["narrative_generation", "insight_extraction", "executive_summary"]
        }
    )
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
