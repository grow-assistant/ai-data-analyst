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
from common_utils import create_agent_server
from .agent_executor import EnhancedDataLoaderExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the enhanced data loader agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("📁 Starting Enhanced Data Loader Agent A2A Server")
    logger.info("⚡ Tableau Hyper API support for blazing fast TDSX/Hyper file loading")

    # Create executor
    executor = EnhancedDataLoaderExecutor()
    
    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="data_loader",
        title="Enhanced Data Loader Agent",
        port=10006,
        agent_description="High-performance data loading with Tableau Hyper API support",
        version="enhanced_v2.0",
        custom_health_data={
            "hyper_api_support": True,
            "supported_formats": ["csv", "json", "parquet", "tdsx", "hyper"]
        }
    )
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main() 