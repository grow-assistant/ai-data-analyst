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
from common_utils import create_agent_server
from .agent_executor import DataEnrichmentAgentExecutor

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the data enrichment agent."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🔄 Starting Data Enrichment Agent A2A Server")

    # Create executor
    executor = DataEnrichmentAgentExecutor()
    
    # Create standardized agent server
    server = create_agent_server(
        executor=executor,
        agent_name="data_enrichment",
        title="Data Enrichment Agent",
        port=10009,
        agent_description="Data enrichment with external APIs and contextual data",
        version="v2.0",
        custom_health_data={
            "capabilities": ["external_data_fetch", "data_merging", "moving_averages"]
        }
    )
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()
