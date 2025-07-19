import logging
import sys
from pathlib import Path

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.mcp_server.tool_server import McpToolServer
from data_cleaning_agent.tools import CleaningTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Data Cleaning MCP Tool Server."""
    logger.info("🛠️ Starting Data Cleaning MCP Tool Server...")

    # Create the MCP Tool Server
    server = McpToolServer(
        title="Data Cleaning Tool Server",
        description="Hosts tools for cleaning and preprocessing data.",
        version="1.0.0"
    )

    # Register the CleaningTool
    cleaning_tool = CleaningTool()
    server.register_tool(cleaning_tool)

    # Run the server on a specific port for data cleaning tools
    # Note: This is a separate process from the agent itself.
    server.run(port=11008)

if __name__ == "__main__":
    main() 