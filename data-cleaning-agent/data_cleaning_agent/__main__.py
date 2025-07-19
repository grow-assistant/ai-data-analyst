#!/usr/bin/env python3
"""
Data Cleaning Agent - A2A Server Implementation
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to make common_utils accessible
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.agent_cards import DATA_CLEANING_AGENT_CARD
from common_utils import create_agent_server
from .agent_executor import DataCleaningAgentExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the data cleaning agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🧹 Starting Data Cleaning Agent A2A Server")

    # Create executor
    executor = DataCleaningAgentExecutor()
    
    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="data_cleaning",
        title="Data Cleaning Agent",
        port=10008,
        agent_description="Data preprocessing and cleaning with advanced validation",
        version="v2.0",
        custom_health_data={
            "capabilities": ["missing_values", "outlier_removal", "data_standardization"]
        }
    )
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
