"""
Schema Profiler Agent Main Module
A2A Agent for automated schema profiling and data analysis.
"""

import logging
import os
from pathlib import Path
import sys

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils import create_agent_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Schema Profiler Agent."""
    try:
        logger.info("Starting Schema Profiler Agent A2A Server")
        
        # Initialize the agent executor
        from schema_profiler.agent_executor import SchemaProfilerAgentExecutor
        executor = SchemaProfilerAgentExecutor()
        
        # Get port from environment or use default
        port = int(os.environ.get('SCHEMA_PROFILER_PORT', 10012))
        
        # Create standardized agent server
        server = create_agent_server(
            executor=executor,
            agent_name="schema_profiler",
            title="Schema Profiler Agent",
            port=port,
            agent_description="AI-powered schema profiling and data analysis with configuration caching",
            version="enhanced_v2.0",
            custom_health_data={
                "ai_profiling": True,
                "capabilities": [
                    "profile_dataset", "ai_profile_dataset", 
                    "get_column_statistics", "compare_schemas",
                    "get_dataset_config", "list_all_configs"
                ],
                "cache_enabled": True
            }
        )
        
        # Run the server
        server.run()
        
    except Exception as e:
        logger.exception(f"Failed to start Schema Profiler Agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
