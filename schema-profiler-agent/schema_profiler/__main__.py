"""
Schema Profiler Agent Main Module
A2A Agent for automated schema profiling and data analysis.
"""

import logging
import os
from pathlib import Path
import sys
import asyncio

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Schema Profiler Agent."""
    try:
        # Import config after logging is set up
        from common_utils.config import Settings
        
        logger.info("Starting Schema Profiler Agent A2A Server")
        
        # Initialize the agent executor
        from schema_profiler.agent_executor import SchemaProfilerAgentExecutor
        agent_executor = SchemaProfilerAgentExecutor()
        
        # Get port from environment or use default
        port = int(os.environ.get('SCHEMA_PROFILER_PORT', 10012))
        
        # Use fallback server implementation
        run_fallback_server(agent_executor, port)
        
    except Exception as e:
        logger.exception(f"Failed to start Schema Profiler Agent: {e}")
        sys.exit(1)

def run_fallback_server(agent_executor, port):
    """Fallback server implementation using FastAPI."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Schema Profiler Agent")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "agent": "schema_profiler"}
    
    @app.get("/capabilities")
    async def capabilities():
        return {
            "skills": [
                "profile_dataset",
                "ai_profile_dataset",
                "get_column_statistics", 
                "compare_schemas",
                "get_dataset_config",
                "list_all_configs"
            ],
            "description": "AI-powered schema profiling and data analysis agent with configuration caching"
        }
    
    @app.post("/")
    async def handle_jsonrpc(request: Request):
        """Handle JSON-RPC requests."""
        try:
            request_data = await request.json()
            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")
            
            if method == "profile_dataset":
                result = await agent_executor.profile_dataset_skill(
                    data_handle_id=params.get("data_handle_id"),
                    profile_type=params.get("profile_type", "comprehensive")
                )
            elif method == "ai_profile_dataset":
                result = await agent_executor.ai_profile_dataset_skill(
                    data_handle_id=params.get("data_handle_id"),
                    use_cache=params.get("use_cache", True),
                    force_ai=params.get("force_ai", False)
                )
            elif method == "get_column_statistics":
                result = await agent_executor.get_column_statistics_skill(
                    data_handle_id=params.get("data_handle_id"),
                    column_name=params.get("column_name")
                )
            elif method == "compare_schemas":
                result = await agent_executor.compare_schemas_skill(
                    data_handle_id1=params.get("data_handle_id1"),
                    data_handle_id2=params.get("data_handle_id2")
                )
            elif method == "get_dataset_config":
                result = await agent_executor.get_dataset_config_skill(
                    data_handle_id=params.get("data_handle_id"),
                    dataset_name=params.get("dataset_name")
                )
            elif method == "list_all_configs":
                result = await agent_executor.list_all_configs_skill()
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id
                })
            
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
            
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return JSONResponse({
                "jsonrpc": "2.0", 
                "error": {"code": -32603, "message": str(e)},
                "id": request_data.get("id") if 'request_data' in locals() else None
            })
    
    logger.info(f"Schema Profiler Agent starting on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
