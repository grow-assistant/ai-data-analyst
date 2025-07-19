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

from common_utils import create_agent_server
from .agent_executor import EnhancedDataAnalystExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the enhanced data analyst agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📊 Starting Enhanced Data Analyst Agent A2A Server")
    logger.info("🔬 Analysis capabilities: comprehensive business intelligence suite")

    # Create executor
    executor = EnhancedDataAnalystExecutor()

    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="data_analyst",
        title="Enhanced Data Analyst Agent",
        port=10007,
        agent_description="Performs comprehensive business intelligence analysis.",
        version="enhanced_v2.0",
        custom_health_data={
            "analysis_suite": "comprehensive",
            "supported_analyses": [
                "trends", "impact", "outliers", "metrics",
                "contribution", "momentum", "narrative"
            ]
        }
    )

    # Run the server
    server.run()

if __name__ == "__main__":
    main() 