"""
Orchestrator Agent Executor
This module contains the implementation of the Orchestrator Agent's skills.
"""

import logging
import httpx
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

logger = logging.getLogger(__name__)

class OrchestratorAgentExecutor:
    """
    Implements the logic for the orchestrator agent's skills.
    """
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        
        # Initialize security for inter-agent calls
        try:
            from common_utils.security import security_manager
            self.security_manager = security_manager
            self.orchestrator_api_key = self.security_manager.get_agent_api_key("orchestrator")
            if not self.orchestrator_api_key:
                # Register if not exists
                self.orchestrator_api_key = self.security_manager.register_agent_api_key("orchestrator")
            logger.info("Orchestrator security initialized with API key authentication")
        except Exception as e:
            logger.warning(f"Failed to initialize security manager: {e}")
            self.security_manager = None
            self.orchestrator_api_key = None
        
        # Load agent endpoints from configuration and discovery
        try:
            from common_utils import get_agent_endpoints, get_agent_registry
            self.agent_endpoints = get_agent_endpoints()
            
            # Try to also discover agents from registry
            try:
                registry = get_agent_registry()
                discovered_endpoints = registry.get_agent_endpoints()
                # Merge discovered endpoints with configured ones
                self.agent_endpoints.update(discovered_endpoints)
                logger.info(f"Loaded agent endpoints from config and discovery: {self.agent_endpoints}")
            except Exception as discovery_error:
                logger.warning(f"Agent discovery failed: {discovery_error}")
                logger.info(f"Using configured endpoints: {self.agent_endpoints}")
                
        except Exception as e:
            logger.warning(f"Failed to load agent endpoints from config, using defaults: {e}")
            # Fallback to hardcoded values
            self.agent_endpoints = {
                "data_loader": "http://localhost:10006",
                "data_cleaning": "http://localhost:10008", 
                "data_enrichment": "http://localhost:10009",
                "data_analyst": "http://localhost:10007",
                "presentation": "http://localhost:10010",
                "schema_profiler": "http://localhost:10012",
                "rootcause_analyst": "http://localhost:10011"
            }
        
        logger.info("OrchestratorAgentExecutor initialized.")

    async def refresh_agent_discovery(self) -> Dict[str, Any]:
        """Refresh agent discovery and return current agent status."""
        try:
            from common_utils import get_agent_registry
            registry = get_agent_registry()
            
            # Perform health checks on all agents
            health_results = await registry.health_check_all_agents()
            
            # Update our endpoints with discovered agents
            discovered_endpoints = registry.get_agent_endpoints()
            self.agent_endpoints.update(discovered_endpoints)
            
            # Clean up inactive agents
            cleaned_up = registry.cleanup_inactive_agents(max_age_hours=1)
            
            return {
                "status": "completed",
                "discovered_agents": len(discovered_endpoints),
                "health_results": health_results,
                "cleaned_up_agents": cleaned_up,
                "active_endpoints": self.agent_endpoints
            }
        except Exception as e:
            logger.exception(f"Error refreshing agent discovery: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def discover_agents_by_capability_skill(self, capability: str) -> Dict[str, Any]:
        """A2A skill to discover agents that have a specific capability."""
        try:
            from common_utils import discover_agents_by_capability
            
            matching_agents = await discover_agents_by_capability(capability)
            
            return {
                "status": "completed",
                "capability": capability,
                "matching_agents": len(matching_agents),
                "agents": [
                    {
                        "name": agent["card"]["name"],
                        "url": agent["card"]["url"],
                        "status": agent["status"],
                        "description": agent["card"]["description"]
                    }
                    for agent in matching_agents
                ]
            }
        except Exception as e:
            logger.exception(f"Error discovering agents by capability: {e}")
            return {
                "status": "error",
                "error": str(e),
                "capability": capability
            }

    async def process_tdsx_workflow_skill(self, file_path: str, profile_type: str = "comprehensive") -> Dict[str, Any]:
        """
        A2A skill to orchestrate file processing workflow.
        Loads any supported file type and creates a comprehensive schema profile.
        """
        logger.info(f"Executing process_tdsx_workflow_skill for file: {file_path}")
        
        workflow_results = {
            "workflow": "file_processing",
            "file_path": file_path,
            "status": "in_progress",
            "steps": [],
            "data_handles": {},
            "errors": []
        }

        try:
            # Step 1: Check agent availability
            logger.info("Step 1: Checking agent availability...")
            workflow_results["steps"].append("checking_agents")
            
            available_agents = await self._check_agent_health()
            required_agents = ["data_loader"]
            optional_agents = ["schema_profiler"]
            
            missing_required = [agent for agent in required_agents if agent not in available_agents]
            if missing_required:
                error_msg = f"Required agents not available: {missing_required}"
                workflow_results["errors"].append(error_msg)
                workflow_results["status"] = "failed"
                logger.error(error_msg)
                return workflow_results
            
            available_optional = [agent for agent in optional_agents if agent in available_agents]
            logger.info(f"Available agents: {list(available_agents.keys())}")

            # Step 2: Load file (auto-detect type)
            logger.info("Step 2: Loading file...")
            workflow_results["steps"].append("loading_file")
            
            # Resolve file path
            resolved_path = await self._resolve_file_path(file_path)
            if not resolved_path:
                error_msg = f"File not found: {file_path}"
                workflow_results["errors"].append(error_msg)
                workflow_results["status"] = "failed"
                return workflow_results

            # Auto-detect file type from extension
            file_extension = resolved_path.suffix.lower().lstrip('.')
            if file_extension in ['csv']:
                detected_type = 'csv'
            elif file_extension in ['tdsx']:
                detected_type = 'tdsx'
            elif file_extension in ['json']:
                detected_type = 'json'
            elif file_extension in ['xlsx', 'xls']:
                detected_type = 'xlsx'
            elif file_extension in ['tsv', 'txt']:
                detected_type = 'tsv'
            else:
                # Default to auto-detection by pandas
                detected_type = None
            
            logger.info(f"Detected file type: {detected_type or 'auto-detect'}")

            # Load the dataset with correct file type
            load_params = {
                "file_path": str(resolved_path)
            }
            if detected_type:
                load_params["file_type"] = detected_type
            
            load_result = await self._call_agent("data_loader", "load_dataset", load_params)
            
            if load_result.get("status") != "completed":
                error_msg = f"File loading failed: {load_result}"
                workflow_results["errors"].append(error_msg)
                workflow_results["status"] = "failed"
                return workflow_results
            
            data_handle_id = load_result["data_handle_id"]
            workflow_results["data_handles"]["loaded_data"] = data_handle_id
            workflow_results["metadata"] = load_result.get("metadata", {})
            workflow_results["detected_file_type"] = detected_type or "auto-detected"
            logger.info(f"✅ File loaded successfully: {data_handle_id}")

            # Step 3: Create schema profile (if schema profiler available)
            profile_result = None
            if "schema_profiler" in available_optional:
                logger.info("Step 3: Creating schema profile...")
                workflow_results["steps"].append("creating_profile")
                
                try:
                    profile_result = await self._call_agent(
                        "schema_profiler",
                        "profile_dataset",
                        {
                            "data_handle_id": data_handle_id,
                            "profile_type": profile_type
                        }
                    )
                    
                    if profile_result.get("status") == "completed":
                        workflow_results["data_handles"]["schema_profile"] = profile_result["profile_handle_id"]
                        workflow_results["profile_summary"] = profile_result.get("summary", {})
                        logger.info(f"✅ Schema profile created: {profile_result['profile_handle_id']}")
                    else:
                        error_msg = f"Schema profiling failed: {profile_result}"
                        workflow_results["errors"].append(error_msg)
                        logger.warning(error_msg)
                        
                except Exception as e:
                    error_msg = f"Schema profiling error: {str(e)}"
                    workflow_results["errors"].append(error_msg)
                    logger.warning(error_msg)
            else:
                logger.info("Step 3: Schema Profiler not available, skipping profiling")
                workflow_results["steps"].append("skipped_profiling")

            # Step 4: Compile final results
            logger.info("Step 4: Compiling results...")
            workflow_results["steps"].append("compiling_results")
            
            # Store comprehensive results in a data handle
            final_results = {
                "workflow_type": "file_processing",
                "source_file": str(resolved_path),
                "file_type": detected_type or "auto-detected",
                "file_metadata": workflow_results.get("metadata", {}),
                "data_handle_id": data_handle_id,
                "profile_data": profile_result.get("profile_data") if profile_result else None,
                "processing_summary": {
                    "steps_completed": workflow_results["steps"],
                    "agents_used": list(available_agents.keys()),
                    "has_schema_profile": profile_result is not None,
                    "data_quality_score": workflow_results.get("profile_summary", {}).get("data_quality_score"),
                    "total_rows": workflow_results.get("profile_summary", {}).get("total_rows"),
                    "total_columns": workflow_results.get("profile_summary", {}).get("total_columns")
                }
            }
            
            # Create data handle for final results
            results_handle = self.data_manager.create_handle(
                data=final_results,
                data_type="workflow_results",
                metadata={
                    "workflow_type": "file_processing",
                    "source_file": str(resolved_path),
                    "file_type": detected_type or "auto-detected",
                    "processing_timestamp": str(asyncio.get_event_loop().time())
                }
            )
            
            workflow_results["data_handles"]["final_results"] = results_handle.handle_id
            workflow_results["status"] = "completed"
            workflow_results["final_results_handle"] = results_handle.handle_id
            
            logger.info(f"✅ File processing workflow completed successfully: {results_handle.handle_id}")
            
            return workflow_results

        except Exception as e:
            error_msg = f"File processing workflow failed: {str(e)}"
            workflow_results["errors"].append(error_msg)
            workflow_results["status"] = "failed"
            logger.exception(error_msg)
            return workflow_results

    async def _check_agent_health(self) -> Dict[str, str]:
        """Check which agents are healthy and available concurrently."""
        
        async def check(agent_name: str, endpoint: str):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{endpoint}/health")
                    if response.status_code == 200:
                        logger.debug(f"Agent {agent_name} is healthy")
                        return agent_name, endpoint
            except Exception as e:
                logger.debug(f"Agent {agent_name} not reachable: {e}")
            return None

        tasks = [check(name, endpoint) for name, endpoint in self.agent_endpoints.items()]
        results = await asyncio.gather(*tasks)
        
        healthy_agents = {name: endpoint for name, endpoint in results if name}
        logger.info(f"Health check complete. Active agents: {list(healthy_agents.keys())}")
        return healthy_agents

    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve file path to absolute path."""
        try:
            path = Path(file_path)
            
            # If absolute path and exists, return it
            if path.is_absolute() and path.exists():
                return path
            
            # Try relative to current working directory
            cwd_path = Path.cwd() / path
            if cwd_path.exists():
                return cwd_path
            
            # Try relative to test_data directory
            test_data_path = Path(__file__).parent.parent.parent / "test_data" / path.name
            if test_data_path.exists():
                return test_data_path
            
            # Try relative to data-loader-agent/data directory
            loader_data_path = Path(__file__).parent.parent.parent / "data-loader-agent" / "data" / path.name
            if loader_data_path.exists():
                return loader_data_path
            
            # If original path doesn't exist, still return it for better error messages
            if path.is_absolute():
                return path
            else:
                return Path.cwd() / path
                
        except Exception as e:
            logger.error(f"Error resolving file path {file_path}: {e}")
            return Path(file_path)

    def _detect_file_type(self, path: Path) -> str:
        """Detect file type from extension."""
        suffix = path.suffix.lower()
        
        type_map = {
            '.hyper': 'hyper',
            '.tdsx': 'tdsx', 
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.json': 'json',
            '.xlsx': 'xlsx',
            '.xls': 'xls',
            '.txt': 'csv'
        }
        
        return type_map.get(suffix, 'auto')

    async def _call_agent_with_retry(self, agent_url: str, method: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Call agent with retry logic."""
        for attempt in range(max_retries):
            try:
                response = await self._make_agent_call(agent_url, method, params)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Agent call failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(1)  # Wait before retry

    async def _make_agent_call(self, agent_url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make actual HTTP call to agent with security headers."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        # Prepare security headers
        headers = {"Content-Type": "application/json"}
        if self.orchestrator_api_key:
            headers["X-API-Key"] = self.orchestrator_api_key
        
        timeout = httpx.Timeout(600.0)  # 10 minutes for large files
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(agent_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                raise ValueError(f"Agent error: {result['error']}")
                
            return result.get("result", {})

    async def orchestrate_pipeline_skill(self, file_path: str, pipeline_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced orchestrate full multi-agent pipeline: Load → Clean → Enrich → Analyze → Present
        Now uses comprehensive analysis and Google Gemini executive reporting.
        """
        logger.info(f"🎯 Starting enhanced multi-agent pipeline for: {file_path}")
        
        try:
            config = pipeline_config or {}
            results = {
                "pipeline_id": str(uuid.uuid4()),
                "file_path": file_path,
                "started_at": datetime.now().isoformat(),
                "stages": {}
            }
            
            # Resolve file path
            resolved_path = self._resolve_file_path(file_path)
            file_type = self._detect_file_type(resolved_path)
            
            logger.info(f"📁 Processing file: {resolved_path} (type: {file_type})")
            
            # Stage 1: Enhanced Data Loading with Hyper API support
            logger.info("🔄 Stage 1: Enhanced Data Loading")
            load_response = await self._call_agent_with_retry(
                "http://localhost:10006", 
                "load_data",  # Updated to use new enhanced skill
                {
                    "file_path": str(resolved_path),
                    "file_type": file_type
                }
            )
            
            if load_response.get("status") != "completed":
                raise ValueError(f"Data loading failed: {load_response.get('error', 'Unknown error')}")
            
            loaded_handle = load_response["data_handle_id"]
            results["stages"]["data_loading"] = {
                "status": "completed",
                "data_handle_id": loaded_handle,
                "metadata": load_response.get("metadata", {}),
                "fast_loading_used": load_response.get("metadata", {}).get("fast_loading", False)
            }
            
            logger.info(f"✅ Data loaded successfully. Handle: {loaded_handle}")
            if results["stages"]["data_loading"]["fast_loading_used"]:
                logger.info("⚡ Tableau Hyper API fast loading was used")
            
            # Stage 2: Data Cleaning
            logger.info("🔄 Stage 2: Data Cleaning")
            clean_response = await self._call_agent_with_retry(
                "http://localhost:10008", 
                "clean_data", 
                {"data_handle_id": loaded_handle}
            )
            
            if clean_response.get("status") != "completed":
                logger.warning(f"Data cleaning failed: {clean_response.get('error', 'Unknown error')}")
                cleaned_handle = loaded_handle  # Use original data
            else:
                cleaned_handle = clean_response["data_handle_id"]
                logger.info(f"✅ Data cleaned successfully. Handle: {cleaned_handle}")
            
            results["stages"]["data_cleaning"] = {
                "status": clean_response.get("status", "skipped"),
                "data_handle_id": cleaned_handle
            }
            
            # Stage 3: Data Enrichment
            logger.info("🔄 Stage 3: Data Enrichment")
            enrich_response = await self._call_agent_with_retry(
                "http://localhost:10009", 
                "enrich_data", 
                {"data_handle_id": cleaned_handle}
            )
            
            if enrich_response.get("status") != "completed":
                logger.warning(f"Data enrichment failed: {enrich_response.get('error', 'Unknown error')}")
                enriched_handle = cleaned_handle  # Use cleaned data
            else:
                enriched_handle = enrich_response["data_handle_id"]
                logger.info(f"✅ Data enriched successfully. Handle: {enriched_handle}")
            
            results["stages"]["data_enrichment"] = {
                "status": enrich_response.get("status", "skipped"),
                "data_handle_id": enriched_handle
            }
            
            # Stage 4: Enhanced Comprehensive Data Analysis
            logger.info("🔄 Stage 4: Enhanced Comprehensive Analysis")
            analysis_response = await self._call_agent_with_retry(
                "http://localhost:10007", 
                "comprehensive_analysis",  # Updated to use new comprehensive analysis
                {
                    "data_handle_id": enriched_handle,
                    "analysis_config": config.get("analysis_config", {})
                }
            )
            
            if analysis_response.get("status") != "completed":
                raise ValueError(f"Comprehensive analysis failed: {analysis_response.get('error', 'Unknown error')}")
            
            analysis_handle = analysis_response["analysis_data_handle_id"]
            results["stages"]["comprehensive_analysis"] = {
                "status": "completed",
                "analysis_handle_id": analysis_handle,
                "summary": analysis_response.get("summary", {}),
                "analysis_modules_used": analysis_response.get("summary", {}).get("analysis_modules_executed", 0)
            }
            
            logger.info(f"✅ Comprehensive analysis completed. Handle: {analysis_handle}")
            logger.info(f"🔬 Analysis modules executed: {analysis_response.get('summary', {}).get('analysis_modules_executed', 0)}")
            
            # Stage 5: Root Cause Analysis with Why-Bot
            logger.info("🔄 Stage 5: Root Cause Analysis (Why-Bot)")
            rootcause_response = await self._call_agent_with_retry(
                "http://localhost:10011", 
                "investigate_trend",  # RootCause Analyst main skill
                {
                    "analysis_handle_id": analysis_handle,
                    "trend_config": config.get("rootcause_config", {})
                }
            )
            
            if rootcause_response.get("status") != "completed":
                logger.warning(f"Root cause analysis failed: {rootcause_response.get('error', 'Unknown error')}")
                investigation_handle = None
                requires_escalation = True
            else:
                investigation_handle = rootcause_response["investigation_handle_id"]
                requires_escalation = rootcause_response.get("summary", {}).get("requires_escalation", False)
                logger.info(f"✅ Root cause analysis completed. Handle: {investigation_handle}")
                if requires_escalation:
                    logger.warning("⚠️ Root cause analysis flagged for escalation - low confidence")
            
            results["stages"]["root_cause_analysis"] = {
                "status": rootcause_response.get("status", "failed"),
                "investigation_handle_id": investigation_handle,
                "summary": rootcause_response.get("summary", {}),
                "requires_escalation": requires_escalation,
                "ai_enhanced": rootcause_response.get("summary", {}).get("ai_enhanced", False)
            }
            
            # Stage 6: Enhanced Executive Report Generation with Google Gemini
            logger.info("🔄 Stage 6: Google Gemini Executive Report Generation")
            presentation_response = await self._call_agent_with_retry(
                "http://localhost:10010", 
                "create_executive_report",  # Updated to use new executive reporting
                {
                    "analysis_handle_id": analysis_handle,
                    "report_config": {
                        **config.get("report_config", {}),
                        "rootcause_handle_id": investigation_handle,
                        "include_rootcause_analysis": investigation_handle is not None
                    }
                }
            )
            
            if presentation_response.get("status") != "completed":
                raise ValueError(f"Executive report generation failed: {presentation_response.get('error', 'Unknown error')}")
            
            final_report_handle = presentation_response["report_data_handle_id"]
            results["stages"]["executive_reporting"] = {
                "status": "completed",
                "report_handle_id": final_report_handle,
                "report_type": "executive",
                "ai_powered": presentation_response.get("ai_powered", False),
                "summary": presentation_response.get("summary", {}),
                "includes_rootcause": investigation_handle is not None
            }
            
            logger.info(f"✅ Executive report generated successfully. Handle: {final_report_handle}")
            if presentation_response.get("ai_powered"):
                logger.info("🤖 Google Gemini AI-powered insights included")
            
            # Pipeline completion
            results.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "final_report_handle_id": final_report_handle,
                "investigation_handle_id": investigation_handle,
                "pipeline_summary": {
                    "total_stages": len(results["stages"]),
                    "successful_stages": len([s for s in results["stages"].values() if s["status"] == "completed"]),
                    "ai_enhanced": True,
                    "hyper_api_used": results["stages"]["data_loading"]["fast_loading_used"],
                    "gemini_ai_used": results["stages"]["executive_reporting"]["ai_powered"],
                    "rootcause_analysis_completed": investigation_handle is not None,
                    "requires_escalation": requires_escalation
                }
            })
            
            # Create comprehensive completion message
            completion_msg = f"""
🎯 Enhanced Multi-Agent Processing Complete with Root Cause Analysis:
  ✅ Data Loader Agent - File loaded with {'Hyper API fast loading' if results['stages']['data_loading']['fast_loading_used'] else 'standard loading'}
  ✅ Data Cleaning Agent - Data cleaned and standardized
  ✅ Data Enrichment Agent - Data enhanced with additional features
  ✅ Data Analyst Agent - Comprehensive analysis performed ({results['stages']['comprehensive_analysis']['analysis_modules_used']} modules)
  {'✅' if investigation_handle else '⚠️'} RootCause Analyst Agent (Why-Bot) - {'Root cause investigation completed' if investigation_handle else 'Root cause analysis failed'}
  ✅ Presentation Agent - Executive report generated with {'Google Gemini AI insights' if results['stages']['executive_reporting']['ai_powered'] else 'standard reporting'}

🔧 Enhanced Capabilities Used:
  {'⚡ Tableau Hyper API for fast file loading' if results['stages']['data_loading']['fast_loading_used'] else '📁 Standard file loading'}
  🔬 Comprehensive business intelligence analysis suite
  {'🤖 Google Gemini AI-powered root cause analysis' if investigation_handle and results['stages']['root_cause_analysis']['ai_enhanced'] else '📊 Statistical root cause analysis'}
  {'🤖 Google Gemini AI-powered executive insights' if results['stages']['executive_reporting']['ai_powered'] else '📊 Standard reporting'}

📊 Final Results:
  - Executive Report Handle: {final_report_handle}
  - Analysis Modules Used: {results['stages']['comprehensive_analysis']['analysis_modules_used']}
  - Root Cause Investigation: {investigation_handle or 'Not completed'}
  {'- ⚠️ ESCALATION REQUIRED: Low confidence in root cause analysis' if requires_escalation else '- ✅ High confidence root cause analysis'}
  - Insights Generated: {results['stages']['executive_reporting']['summary'].get('insights_count', 'N/A')}
  - Recommendations: {results['stages']['executive_reporting']['summary'].get('recommendations_count', 'N/A')}

🔧 Next Steps:
  - Extract executive report: python extract_report.py {final_report_handle}
  - Access comprehensive analysis: {analysis_handle}
  {'- Extract root cause investigation: python extract_report.py ' + investigation_handle if investigation_handle else '- Root cause analysis needs manual review'}
  - Review AI-powered business insights and strategic recommendations
  {'- ESCALATE: Manual expert review recommended due to low confidence' if requires_escalation else '- COMPLETE: Automated analysis pipeline successful'}
            """
            
            print(completion_msg)
            logger.info("🎉 Enhanced multi-agent pipeline with root cause analysis completed successfully")
            
            return results
            
        except Exception as e:
            logger.exception(f"Enhanced pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "partial_results": results if 'results' in locals() else {}
            }

    async def _call_agent(self, agent_name: str, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific agent's skill via JSON-RPC.
        """
        endpoint = self.agent_endpoints[agent_name]
        
        request_payload = {
            "jsonrpc": "2.0",
            "method": skill_name,
            "params": params,
            "id": f"orchestrator_{agent_name}_{skill_name}",
        }

        # Prepare security headers
        headers = {"Content-Type": "application/json"}
        if self.orchestrator_api_key:
            headers["X-API-Key"] = self.orchestrator_api_key

        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=request_payload, headers=headers, timeout=300.0)  # 5 minutes for large files
            response.raise_for_status()
            response_data = response.json()
            
            if "error" in response_data:
                raise ValueError(f"Agent {agent_name} returned error: {response_data['error']}")
            
            return response_data.get("result", {})

    async def orchestrate_parallel_analysis_skill(self, data_handle_id: str) -> Dict[str, Any]:
        """
        Orchestrates multiple independent analyses in parallel to improve performance.
        """
        logger.info(f"🎯 Starting parallel analysis for data handle: {data_handle_id}")

        results = {
            "pipeline_id": str(uuid.uuid4()),
            "data_handle_id": data_handle_id,
            "started_at": datetime.now().isoformat(),
            "analyses": {}
        }

        try:
            # Define a set of independent analyses to run in parallel
            analysis_tasks = {
                "comprehensive_analysis": {
                    "agent": "data_analyst",
                    "skill": "comprehensive_analysis",
                    "params": {"data_handle_id": data_handle_id}
                },
                "root_cause_analysis": {
                    "agent": "rootcause_analyst",
                    "skill": "investigate_trend",
                    "params": {"analysis_handle_id": None} # Depends on comprehensive analysis
                },
                "schema_profile": {
                    "agent": "schema_profiler",
                    "skill": "ai_profile_dataset",
                    "params": {"data_handle_id": data_handle_id}
                }
            }

            # First, run the comprehensive analysis to get its handle
            comp_analysis_response = await self._call_agent(
                analysis_tasks["comprehensive_analysis"]["agent"],
                analysis_tasks["comprehensive_analysis"]["skill"],
                analysis_tasks["comprehensive_analysis"]["params"]
            )

            if comp_analysis_response.get("status") != "completed":
                raise ValueError("Comprehensive analysis failed, halting parallel execution.")

            analysis_handle = comp_analysis_response.get("analysis_data_handle_id")
            results["analyses"]["comprehensive_analysis"] = {
                "status": "completed",
                "handle": analysis_handle,
            }
            # Update the params for root cause analysis
            analysis_tasks["root_cause_analysis"]["params"]["analysis_handle_id"] = analysis_handle

            # Now, create tasks for the remaining parallelizable analyses
            concurrent_tasks = [
                self._call_agent(
                    details["agent"],
                    details["skill"],
                    details["params"]
                ) for name, details in analysis_tasks.items() if name != "comprehensive_analysis"
            ]
            
            # Execute tasks concurrently
            task_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

            # Process results
            parallel_task_names = [name for name in analysis_tasks if name != "comprehensive_analysis"]
            for name, result in zip(parallel_task_names, task_results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel analysis task '{name}' failed: {result}")
                    results["analyses"][name] = {"status": "error", "error": str(result)}
                else:
                    results["analyses"][name] = {"status": "completed", "result": result}
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now().isoformat()
            
            return results

        except Exception as e:
            logger.exception(f"Parallel analysis pipeline failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results 