"""A2A Client for Orchestrator Agent - Inter-Agent Communication"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory for common_utils access
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import A2A SDK
from a2a import A2AClient
from a2a.types import TaskStatus

from common_utils.circuit_breaker import get_circuit_breaker_manager, CircuitBreakerConfig, CircuitBreakerException
from common_utils.types import TaskRequest, TaskResponse, DataHandle
from common_utils.agent_cards import ALL_AGENT_CARDS

logger = logging.getLogger(__name__)

class OrchestratorA2AClient:
    """A2A Client for orchestrator to communicate with specialist agents."""
    
    def __init__(self):
        self.a2a_client = A2AClient()
        self.circuit_manager = get_circuit_breaker_manager()
        
        # Initialize security for inter-agent calls
        try:
            from common_utils.security import security_manager
            self.security_manager = security_manager
            self.orchestrator_api_key = self.security_manager.get_agent_api_key("orchestrator")
            if not self.orchestrator_api_key:
                self.orchestrator_api_key = self.security_manager.register_agent_api_key("orchestrator")
            logger.info("A2A Client security initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize A2A Client security: {e}")
            self.security_manager = None
            self.orchestrator_api_key = None
        
        # Load agent endpoints from configuration
        try:
            from common_utils import get_agent_endpoints
            configured_endpoints = get_agent_endpoints()
            # Map to A2A client naming convention
            self.agent_endpoints = {
                "data_loader_agent": configured_endpoints.get("data_loader", "http://localhost:10006"),
                "data_cleaning_agent": configured_endpoints.get("data_cleaning", "http://localhost:10008"), 
                "data_enrichment_agent": configured_endpoints.get("data_enrichment", "http://localhost:10009"),
                "data_analyst_agent": configured_endpoints.get("data_analyst", "http://localhost:10007"),
                "presentation_agent": configured_endpoints.get("presentation", "http://localhost:10010")
            }
            logger.info(f"A2A Client loaded agent endpoints from configuration: {self.agent_endpoints}")
        except Exception as e:
            logger.warning(f"Failed to load agent endpoints from config, using defaults: {e}")
            # Fallback to hardcoded values
            self.agent_endpoints = {
                "data_loader_agent": "http://localhost:10006",
                "data_cleaning_agent": "http://localhost:10008", 
                "data_enrichment_agent": "http://localhost:10009",
                "data_analyst_agent": "http://localhost:10007",
                "presentation_agent": "http://localhost:10010"
            }
        
        # Configure circuit breakers for each agent
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=45,
            success_threshold=2,
            timeout=60.0  # Longer timeout for data processing
        )
        
        for agent_name in self.agent_endpoints:
            self.circuit_manager.get_breaker(agent_name, self.circuit_config)
        
        logger.info("OrchestratorA2AClient initialized")
    
    async def send_task_to_agent(self, agent_name: str, skill_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a task to a specific agent using A2A protocol.
        
        Args:
            agent_name: Name of the target agent
            skill_id: ID of the skill to invoke
            parameters: Parameters for the skill
            
        Returns:
            Response from the agent
        """
        try:
            circuit_breaker = self.circuit_manager.get_breaker(agent_name, self.circuit_config)
            
            async def _send_request():
                agent_url = self.agent_endpoints.get(agent_name)
                if not agent_url:
                    raise ValueError(f"Unknown agent: {agent_name}")
                
                # Use A2A client to send task
                response = await self.a2a_client.send_task(
                    agent_url=agent_url,
                    skill_id=skill_id,
                    parameters=parameters,
                    timeout=60.0
                )
                
                return response
            
            # Execute with circuit breaker protection
            result = await circuit_breaker.call(_send_request)
            
            logger.info(f"Successfully sent task to {agent_name}.{skill_id}")
            return result
            
        except CircuitBreakerException as e:
            logger.error(f"Circuit breaker open for {agent_name}: {e}")
            return {"error": f"Agent {agent_name} temporarily unavailable"}
        except Exception as e:
            logger.error(f"Error sending task to {agent_name}.{skill_id}: {e}")
            return {"error": f"Failed to communicate with {agent_name}: {str(e)}"}
    
    async def load_dataset(self, file_path: str, load_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load a dataset using the data loader agent."""
        parameters = {
            "file_path": file_path,
            "load_options": load_options or {}
        }
        return await self.send_task_to_agent("data_loader_agent", "load_dataset", parameters)
    
    async def clean_dataset(self, data_handle_id: str, cleaning_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Clean a dataset using the data cleaning agent."""
        parameters = {
            "data_handle_id": data_handle_id,
            "cleaning_options": cleaning_options or {}
        }
        return await self.send_task_to_agent("data_cleaning_agent", "clean_dataset", parameters)
    
    async def enrich_dataset(self, data_handle_id: str, enrichment_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enrich a dataset using the data enrichment agent."""
        parameters = {
            "data_handle_id": data_handle_id,
            "enrichment_options": enrichment_options or {}
        }
        return await self.send_task_to_agent("data_enrichment_agent", "enrich_dataset", parameters)
    
    async def analyze_dataset(self, data_handle_id: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze a dataset using the data analyst agent."""
        parameters = {
            "data_handle_id": data_handle_id,
            "analysis_options": analysis_options or {}
        }
        return await self.send_task_to_agent("data_analyst_agent", "analyze_dataset", parameters)
    
    async def create_report(self, data_handle_id: str = None, analysis_results: Dict[str, Any] = None, 
                          report_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a report using the presentation agent."""
        parameters = {
            "report_options": report_options or {}
        }
        
        if data_handle_id:
            parameters["data_handle_id"] = data_handle_id
        elif analysis_results:
            parameters["analysis_results"] = analysis_results
        else:
            return {"error": "Either data_handle_id or analysis_results is required"}
        
        return await self.send_task_to_agent("presentation_agent", "create_report", parameters)
    
    async def execute_full_pipeline(self, file_path: str, pipeline_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the full data analysis pipeline from loading to presentation.
        
        Args:
            file_path: Path to the data file to analyze
            pipeline_options: Options for each stage of the pipeline
            
        Returns:
            Dictionary containing the final report and intermediate results
        """
        try:
            options = pipeline_options or {}
            trace_id = f"pipeline_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}"
            
            logger.info(f"[{trace_id}] Starting full pipeline for {file_path}")
            
            # Step 1: Load data
            logger.info(f"[{trace_id}] Step 1: Loading dataset")
            load_result = await self.load_dataset(file_path, options.get("load_options"))
            if "error" in load_result:
                return {"error": f"Data loading failed: {load_result['error']}", "stage": "loading"}
            
            data_handle = load_result.get("data_handle")
            if not data_handle:
                return {"error": "No data handle returned from loader", "stage": "loading"}
            
            data_handle_id = data_handle.get("handle_id")
            logger.info(f"[{trace_id}] Data loaded with handle: {data_handle_id}")
            
            # Step 2: Clean data
            logger.info(f"[{trace_id}] Step 2: Cleaning dataset")
            clean_result = await self.clean_dataset(data_handle_id, options.get("cleaning_options"))
            if "error" in clean_result:
                return {"error": f"Data cleaning failed: {clean_result['error']}", "stage": "cleaning"}
            
            cleaned_handle = clean_result.get("cleaned_data_handle")
            cleaned_handle_id = cleaned_handle.get("handle_id")
            logger.info(f"[{trace_id}] Data cleaned with handle: {cleaned_handle_id}")
            
            # Step 3: Enrich data
            logger.info(f"[{trace_id}] Step 3: Enriching dataset")
            enrich_result = await self.enrich_dataset(cleaned_handle_id, options.get("enrichment_options"))
            if "error" in enrich_result:
                return {"error": f"Data enrichment failed: {enrich_result['error']}", "stage": "enrichment"}
            
            enriched_handle = enrich_result.get("enriched_data_handle")
            enriched_handle_id = enriched_handle.get("handle_id")
            logger.info(f"[{trace_id}] Data enriched with handle: {enriched_handle_id}")
            
            # Step 4: Analyze data
            logger.info(f"[{trace_id}] Step 4: Analyzing dataset")
            analysis_result = await self.analyze_dataset(enriched_handle_id, options.get("analysis_options"))
            if "error" in analysis_result:
                return {"error": f"Data analysis failed: {analysis_result['error']}", "stage": "analysis"}
            
            logger.info(f"[{trace_id}] Analysis completed")
            
            # Step 5: Create presentation
            logger.info(f"[{trace_id}] Step 5: Creating report")
            report_result = await self.create_report(
                analysis_results=analysis_result.get("results"),
                report_options=options.get("report_options")
            )
            if "error" in report_result:
                return {"error": f"Report creation failed: {report_result['error']}", "stage": "presentation"}
            
            logger.info(f"[{trace_id}] Pipeline completed successfully")
            
            # Return comprehensive results
            return {
                "success": True,
                "trace_id": trace_id,
                "pipeline_results": {
                    "loading": load_result,
                    "cleaning": clean_result,
                    "enrichment": enrich_result,
                    "analysis": analysis_result,
                    "presentation": report_result
                },
                "final_report_path": report_result.get("report_path"),
                "data_handles": {
                    "raw": data_handle_id,
                    "cleaned": cleaned_handle_id,
                    "enriched": enriched_handle_id
                }
            }
            
        except Exception as e:
            logger.error(f"[{trace_id}] Pipeline execution failed: {e}")
            return {"error": f"Pipeline execution failed: {str(e)}", "stage": "orchestration"}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all managed agents."""
        status = {}
        
        for agent_name, endpoint in self.agent_endpoints.items():
            circuit_breaker = self.circuit_manager.get_breaker(agent_name, self.circuit_config)
            circuit_status = circuit_breaker.get_status()
            
            status[agent_name] = {
                "endpoint": endpoint,
                "circuit_breaker": circuit_status,
                "available": circuit_status["state"] != "OPEN"
            }
        
        return status
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self.a2a_client, 'close'):
            await self.a2a_client.close()
        logger.info("A2A client closed") 