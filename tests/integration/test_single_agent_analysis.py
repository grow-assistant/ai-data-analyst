#!/usr/bin/env python3

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Single Agent Analysis Test

Tests individual agents with real analysis scenarios to validate production readiness
and provide detailed critique of actual outputs.
"""

import asyncio
import logging
import json
import time
import subprocess
import signal
import os
from pathlib import Path
from datetime import datetime
import httpx
import pandas as pd

# Framework imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common_utils.enhanced_logging import get_logger, correlated_operation

logger = get_logger(__name__)

class SingleAgentAnalysisTest:
    """
    Test individual agents with focused analysis scenarios.
    """
    
    def __init__(self):
        self.test_data_path = Path(__file__).parent.parent.parent / "test_data"
        self.results_log = []
        self.active_processes = {}
        self.parent_dir = Path(__file__).parent.parent.parent
        
    async def test_data_loader_agent(self):
        """Test the data loader agent with real data."""
        logger.info("📁 TESTING DATA LOADER AGENT")
        
        # Start the data loader agent
        agent_process = await self.start_agent("data_loader", "data-loader-agent", 10006)
        
        if not agent_process:
            return {"status": "failed", "error": "Could not start data loader agent"}
        
        try:
            # Wait for agent to be ready
            await asyncio.sleep(5)
            
            # Test health endpoint
            health_result = await self.test_agent_health("data_loader", 10006)
            
            if not health_result["healthy"]:
                return {"status": "failed", "error": "Agent health check failed", "health": health_result}
            
            # Test data loading functionality
            load_result = await self.test_data_loading(10006)
            
            return {
                "status": "completed" if load_result["success"] else "failed",
                "health": health_result,
                "load_test": load_result,
                "agent": "data_loader",
                "critique": self.critique_data_loader_performance(health_result, load_result)
            }
            
        finally:
            await self.stop_agent("data_loader")
    
    async def test_data_analyst_agent(self):
        """Test the data analyst agent with analysis scenarios."""
        logger.info("📊 TESTING DATA ANALYST AGENT")
        
        # Start the data analyst agent
        agent_process = await self.start_agent("data_analyst", "data-analyst-agent", 10007)
        
        if not agent_process:
            return {"status": "failed", "error": "Could not start data analyst agent"}
        
        try:
            # Wait for agent to be ready
            await asyncio.sleep(5)
            
            # Test health endpoint
            health_result = await self.test_agent_health("data_analyst", 10007)
            
            if not health_result["healthy"]:
                return {"status": "failed", "error": "Agent health check failed", "health": health_result}
            
            # Test analysis functionality
            analysis_results = await self.test_analysis_scenarios(10007)
            
            return {
                "status": "completed" if any(r["success"] for r in analysis_results) else "failed",
                "health": health_result,
                "analysis_tests": analysis_results,
                "agent": "data_analyst",
                "critique": self.critique_analyst_performance(health_result, analysis_results)
            }
            
        finally:
            await self.stop_agent("data_analyst")
    
    async def test_orchestrator_integration(self):
        """Test orchestrator with multiple agents."""
        logger.info("🎼 TESTING ORCHESTRATOR INTEGRATION")
        
        # Start key agents
        agents_to_start = [
            ("data_loader", "data-loader-agent", 10006),
            ("data_analyst", "data-analyst-agent", 10007),
            ("presentation", "presentation-agent", 10010)
        ]
        
        started_agents = []
        
        try:
            # Start agents sequentially
            for agent_name, agent_dir, port in agents_to_start:
                process = await self.start_agent(agent_name, agent_dir, port)
                if process:
                    started_agents.append(agent_name)
                    await asyncio.sleep(3)  # Wait between starts
            
            logger.info(f"✅ Started {len(started_agents)} agents: {started_agents}")
            
            # Wait for all agents to be ready
            await asyncio.sleep(10)
            
            # Test orchestrated pipeline
            pipeline_result = await self.test_orchestrated_pipeline()
            
            return {
                "status": "completed" if pipeline_result["success"] else "failed",
                "started_agents": started_agents,
                "pipeline_test": pipeline_result,
                "critique": self.critique_orchestrator_performance(started_agents, pipeline_result)
            }
            
        finally:
            # Stop all started agents
            for agent_name in started_agents:
                await self.stop_agent(agent_name)
    
    async def start_agent(self, agent_name: str, agent_dir: str, port: int):
        """Start a single agent."""
        agent_path = self.parent_dir / agent_dir
        
        if not agent_path.exists():
            logger.error(f"Agent directory not found: {agent_path}")
            return None
            
        logger.info(f"🚀 Starting {agent_name} agent on port {port}")
        
        try:
            # Determine the module name based on the agent
            module_map = {
                "data_loader": "data_loader",
                "data_analyst": "data_analyst", 
                "presentation": "presentation_agent",
                "data_cleaning": "data_cleaning_agent",
                "data_enrichment": "data_enrichment_agent",
                "rootcause_analyst": "rootcause_analyst",
                "schema_profiler": "schema_profiler"
            }
            
            module_name = module_map.get(agent_name, agent_name)
            
            process = subprocess.Popen(
                [sys.executable, "-m", module_name],
                cwd=str(agent_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.active_processes[agent_name] = process
            logger.info(f"✅ Started {agent_name} agent (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start {agent_name} agent: {e}")
            return None
    
    async def stop_agent(self, agent_name: str):
        """Stop a single agent."""
        if agent_name in self.active_processes:
            process = self.active_processes[agent_name]
            try:
                logger.info(f"🛑 Stopping {agent_name} agent")
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ {agent_name} agent stopped")
            except Exception as e:
                logger.error(f"Error stopping {agent_name}: {e}")
                process.kill()
            finally:
                del self.active_processes[agent_name]
    
    async def test_agent_health(self, agent_name: str, port: int) -> dict:
        """Test agent health endpoint."""
        url = f"http://localhost:{port}/health"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    health_data = response.json()
                    logger.info(f"✅ {agent_name} health check passed")
                    return {
                        "healthy": True,
                        "status_code": response.status_code,
                        "health_data": health_data,
                        "response_time": response.elapsed.total_seconds()
                    }
                else:
                    logger.error(f"❌ {agent_name} health check failed: {response.status_code}")
                    return {
                        "healthy": False,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"❌ {agent_name} health check exception: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def test_data_loading(self, port: int) -> dict:
        """Test data loading functionality."""
        url = f"http://localhost:{port}/invoke"
        
        payload = {
            "method": "load_data",
            "params": {
                "file_path": str(self.test_data_path / "sales_data_small.csv"),
                "config": {"format": "csv", "validate": True}
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Data loading test passed")
                    return {
                        "success": True,
                        "response": result,
                        "data_info": self.extract_data_info(result)
                    }
                else:
                    logger.error(f"❌ Data loading test failed: {response.status_code}")
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                    
        except Exception as e:
            logger.error(f"❌ Data loading test exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_analysis_scenarios(self, port: int) -> list:
        """Test various analysis scenarios."""
        url = f"http://localhost:{port}/invoke"
        
        scenarios = [
            {
                "name": "revenue_analysis",
                "method": "analyze_metrics",
                "params": {
                    "data_handle": "test_data",
                    "metrics": ["revenue"],
                    "analysis_type": "descriptive"
                }
            },
            {
                "name": "trend_analysis", 
                "method": "analyze_trends",
                "params": {
                    "data_handle": "test_data",
                    "time_column": "date",
                    "value_column": "revenue"
                }
            },
            {
                "name": "outlier_detection",
                "method": "detect_outliers",
                "params": {
                    "data_handle": "test_data",
                    "columns": ["revenue", "units_sold"]
                }
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json={
                        "method": scenario["method"],
                        "params": scenario["params"]
                    })
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ {scenario['name']} analysis passed")
                        results.append({
                            "scenario": scenario["name"],
                            "success": True,
                            "response": result,
                            "insights": self.extract_analysis_insights(result)
                        })
                    else:
                        logger.error(f"❌ {scenario['name']} analysis failed: {response.status_code}")
                        results.append({
                            "scenario": scenario["name"],
                            "success": False,
                            "error": f"HTTP {response.status_code}"
                        })
                        
            except Exception as e:
                logger.error(f"❌ {scenario['name']} analysis exception: {e}")
                results.append({
                    "scenario": scenario["name"], 
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def test_orchestrated_pipeline(self) -> dict:
        """Test orchestrated analysis pipeline."""
        # Import orchestrator here to avoid circular imports
        try:
            from orchestrator_agent.agent_executor import OrchestratorAgentExecutor
            orchestrator = OrchestratorAgentExecutor()
            
            # Run a simple analysis pipeline
            result = await orchestrator.orchestrate_pipeline_skill(
                file_path=str(self.test_data_path / "sales_data_small.csv"),
                pipeline_config={
                    "analysis_focus": "revenue_analysis",
                    "metrics": ["revenue", "units_sold"],
                    "quick_analysis": True
                }
            )
            
            logger.info("✅ Orchestrated pipeline test completed")
            return {
                "success": True,
                "result": result,
                "pipeline_insights": self.extract_pipeline_insights(result)
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrated pipeline test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_data_info(self, result: dict) -> dict:
        """Extract data information from loading result."""
        return {
            "status": result.get("status", "unknown"),
            "data_handle": result.get("data_handle_id", "none"),
            "row_count": result.get("metadata", {}).get("row_count", 0),
            "column_count": result.get("metadata", {}).get("column_count", 0),
            "file_size": result.get("metadata", {}).get("file_size", 0)
        }
    
    def extract_analysis_insights(self, result: dict) -> dict:
        """Extract insights from analysis result."""
        return {
            "analysis_type": result.get("analysis_type", "unknown"),
            "insights_count": len(result.get("insights", [])),
            "has_visualizations": "visualizations" in result,
            "confidence_score": result.get("confidence", 0)
        }
    
    def extract_pipeline_insights(self, result: dict) -> dict:
        """Extract insights from pipeline result."""
        return {
            "pipeline_status": result.get("status", "unknown"),
            "stages_completed": len([s for s in result.get("stages", {}).values() if s.get("status") == "completed"]),
            "total_stages": len(result.get("stages", {})),
            "has_final_report": bool(result.get("final_report_handle_id")),
            "session_id": result.get("session_id")
        }
    
    def critique_data_loader_performance(self, health: dict, load_test: dict) -> dict:
        """Provide detailed critique of data loader performance."""
        critique = {
            "overall_rating": "excellent",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Health check analysis
        if health["healthy"]:
            critique["strengths"].append("Agent startup and health endpoints working correctly")
            if health.get("response_time", 1) < 0.5:
                critique["strengths"].append("Fast health check response time")
        else:
            critique["weaknesses"].append("Health check failures indicate startup issues")
            critique["overall_rating"] = "poor"
        
        # Data loading analysis
        if load_test["success"]:
            data_info = load_test["data_info"]
            if data_info["row_count"] > 0:
                critique["strengths"].append(f"Successfully loaded {data_info['row_count']} rows of data")
            if data_info["column_count"] > 0:
                critique["strengths"].append(f"Correctly identified {data_info['column_count']} columns")
        else:
            critique["weaknesses"].append("Data loading functionality failed")
            critique["overall_rating"] = "poor"
        
        # Recommendations
        if critique["overall_rating"] == "excellent":
            critique["recommendations"].append("Data loader is production-ready")
        else:
            critique["recommendations"].append("Address startup and loading issues before production")
        
        return critique
    
    def critique_analyst_performance(self, health: dict, analysis_tests: list) -> dict:
        """Provide detailed critique of data analyst performance."""
        critique = {
            "overall_rating": "good",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        successful_tests = [t for t in analysis_tests if t["success"]]
        success_rate = len(successful_tests) / len(analysis_tests) if analysis_tests else 0
        
        if success_rate >= 0.8:
            critique["strengths"].append(f"High analysis success rate: {success_rate:.1%}")
        elif success_rate >= 0.5:
            critique["weaknesses"].append(f"Moderate analysis success rate: {success_rate:.1%}")
            critique["overall_rating"] = "fair"
        else:
            critique["weaknesses"].append(f"Low analysis success rate: {success_rate:.1%}")
            critique["overall_rating"] = "poor"
        
        # Analyze specific capabilities
        scenario_names = [t["scenario"] for t in successful_tests]
        if "revenue_analysis" in scenario_names:
            critique["strengths"].append("Revenue analysis capability working")
        if "trend_analysis" in scenario_names:
            critique["strengths"].append("Trend analysis capability working")
        if "outlier_detection" in scenario_names:
            critique["strengths"].append("Outlier detection capability working")
        
        critique["recommendations"].append("Consider expanding analysis capabilities")
        
        return critique
    
    def critique_orchestrator_performance(self, started_agents: list, pipeline_result: dict) -> dict:
        """Provide detailed critique of orchestrator performance."""
        critique = {
            "overall_rating": "good",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Agent coordination analysis
        if len(started_agents) >= 2:
            critique["strengths"].append(f"Successfully coordinated {len(started_agents)} agents")
        else:
            critique["weaknesses"].append("Limited agent coordination capability")
        
        # Pipeline execution analysis
        if pipeline_result["success"]:
            insights = pipeline_result["pipeline_insights"]
            completion_rate = insights["stages_completed"] / insights["total_stages"] if insights["total_stages"] > 0 else 0
            
            if completion_rate >= 0.8:
                critique["strengths"].append(f"High pipeline completion rate: {completion_rate:.1%}")
            else:
                critique["weaknesses"].append(f"Incomplete pipeline execution: {completion_rate:.1%}")
                critique["overall_rating"] = "fair"
        else:
            critique["weaknesses"].append("Pipeline execution failed")
            critique["overall_rating"] = "poor"
        
        critique["recommendations"].append("Test with full agent ecosystem for complete evaluation")
        
        return critique
    
    async def run_comprehensive_single_agent_tests(self):
        """Run all single agent tests and provide comprehensive analysis."""
        logger.info("🧪 STARTING COMPREHENSIVE SINGLE AGENT TESTS")
        
        test_results = []
        
        # Test individual agents
        tests = [
            ("Data Loader", self.test_data_loader_agent),
            ("Data Analyst", self.test_data_analyst_agent),
            ("Orchestrator Integration", self.test_orchestrator_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 TESTING: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                result["test_name"] = test_name
                test_results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                test_results.append({
                    "test_name": test_name,
                    "status": "crashed",
                    "error": str(e)
                })
        
        # Comprehensive analysis
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 COMPREHENSIVE SINGLE AGENT TEST ANALYSIS")
        logger.info(f"{'='*80}")
        
        analysis = self.analyze_comprehensive_results(test_results)
        
        # Save results
        results_file = Path(__file__).parent / f"single_agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({"test_results": test_results, "analysis": analysis}, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to: {results_file}")
        
        return analysis
    
    def analyze_comprehensive_results(self, test_results: list) -> dict:
        """Analyze comprehensive test results and provide final critique."""
        successful_tests = [r for r in test_results if r.get("status") == "completed"]
        
        analysis = {
            "summary": {
                "total_tests": len(test_results),
                "successful_tests": len(successful_tests),
                "success_rate": len(successful_tests) / len(test_results) if test_results else 0
            },
            "agent_readiness": {},
            "system_critique": {
                "production_readiness": "needs_assessment",
                "key_strengths": [],
                "critical_issues": [],
                "recommendations": []
            }
        }
        
        # Analyze individual agent results
        for result in test_results:
            if "critique" in result:
                agent_name = result.get("agent", result.get("test_name", "unknown"))
                analysis["agent_readiness"][agent_name] = result["critique"]
        
        # Overall system critique
        success_rate = analysis["summary"]["success_rate"]
        
        if success_rate >= 0.8:
            analysis["system_critique"]["production_readiness"] = "ready"
            analysis["system_critique"]["key_strengths"].append("High agent reliability and functionality")
        elif success_rate >= 0.6:
            analysis["system_critique"]["production_readiness"] = "nearly_ready"
            analysis["system_critique"]["key_strengths"].append("Core functionality working")
            analysis["system_critique"]["critical_issues"].append("Some agent reliability issues")
        else:
            analysis["system_critique"]["production_readiness"] = "not_ready"
            analysis["system_critique"]["critical_issues"].append("Significant agent failures")
        
        # Generate recommendations
        if success_rate < 1.0:
            analysis["system_critique"]["recommendations"].append("Address failing agent tests before production deployment")
        
        analysis["system_critique"]["recommendations"].extend([
            "Implement comprehensive monitoring and alerting",
            "Add automated health checks in production",
            "Consider load testing with larger datasets"
        ])
        
        # Log final critique
        logger.info(f"🎯 FINAL SYSTEM CRITIQUE:")
        logger.info(f"   📊 Success Rate: {success_rate:.1%}")
        logger.info(f"   🚀 Production Readiness: {analysis['system_critique']['production_readiness']}")
        logger.info(f"   💪 Key Strengths: {len(analysis['system_critique']['key_strengths'])}")
        logger.info(f"   ⚠️ Critical Issues: {len(analysis['system_critique']['critical_issues'])}")
        
        return analysis


# Standalone execution
async def main():
    """Main function for standalone execution."""
    test_suite = SingleAgentAnalysisTest()
    results = await test_suite.run_comprehensive_single_agent_tests()
    
    success_rate = results["summary"]["success_rate"]
    if success_rate >= 0.7:
        logger.info("🎉 SINGLE AGENT TESTS PASSED!")
        return 0
    else:
        logger.error("💥 SINGLE AGENT TESTS NEED IMPROVEMENT!")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 