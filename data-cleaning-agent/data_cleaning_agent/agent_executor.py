"""
Data Cleaning Agent Executor (MCP Client)
This module acts as an MCP client to invoke the remote cleaning tool.
"""

import logging
from typing import Dict, Any, List

from common_utils.agent_security import make_secure_agent_call
from common_utils.agent_config import get_agent_endpoint

logger = logging.getLogger(__name__)

class DataCleaningAgentExecutor:
    """
    Acts as an MCP client for the Data Cleaning agent.
    It invokes the remote CleaningTool on the MCP Tool Server.
    """
    def __init__(self):
        # The tool server's endpoint is configured in system_config.yaml
        self.tool_server_url = get_agent_endpoint("data_cleaning_tool_server") or "http://localhost:11008"
        logger.info(f"DataCleaningAgentExecutor initialized. Tool Server URL: {self.tool_server_url}")

    async def clean_dataset_skill(self, data_handle_id: str, operations: List[str] = None) -> Dict[str, Any]:
        """
        A2A skill that invokes the 'clean_dataset' tool on the MCP server.
        """
        logger.info(f"Invoking 'clean_dataset' tool for data handle: {data_handle_id}")
        
        payload = {
            "tool_name": "clean_dataset",
            "parameters": {
                "data_handle_id": data_handle_id,
                "operations": operations or ["remove_duplicates", "handle_missing_values"]
            }
        }

        try:
            # Using 'data_cleaning' as the agent name for making the secure call
            result = await make_secure_agent_call(
                "data_cleaning",
                f"{self.tool_server_url}/invoke",
                payload
            )
            return result
        except Exception as e:
            logger.exception(f"Error invoking cleaning tool for handle {data_handle_id}: {e}")
            return {"status": "error", "error": str(e)} 