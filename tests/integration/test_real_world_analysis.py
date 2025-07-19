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
Real-World Analysis Integration Test Suite

Tests the complete AI Data Analyst Multi-Agent Framework with 3 typical
business analysis scenarios using sales data.
"""

import asyncio
import logging
import json
import time
from pathlib import Path
import pytest
from datetime import datetime

# Framework imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common_utils.observability import get_observability_manager, trace_operation
from common_utils.session_manager import get_session_manager
from common_utils.enhanced_logging import get_logger, correlated_operation

# Import orchestrator
orchestrator_path = Path(__file__).parent.parent.parent / "orchestrator-agent"
sys.path.insert(0, str(orchestrator_path))
from orchestrator_agent.agent_executor import OrchestratorAgentExecutor

logger = get_logger(__name__)

class RealWorldAnalysisTest:
    """
    Integration test suite for real-world business analysis scenarios.
    """
    
    def __init__(self):
        self.orchestrator = OrchestratorAgentExecutor()
        self.session_manager = get_session_manager()
        self.observability = get_observability_manager()
        self.test_data_path = Path(__file__).parent.parent.parent / "test_data"
        self.results_log = []
        
    async def setup(self):
        """Setup test environment."""
        logger.info("🚀 Setting up Real-World Analysis Integration Test")
        
        # Verify test data exists
        sales_data_file = self.test_data_path / "sales_data_small.csv"
        if not sales_data_file.exists():
            raise FileNotFoundError(f"Test data not found: {sales_data_file}")
        
        logger.info(f"✅ Test data verified: {sales_data_file}")
        
    async def run_analysis_scenario(self, scenario_name: str, file_path: str, 
                                  pipeline_config: dict = None) -> dict:
        """Run a complete analysis scenario and capture results."""
        logger.info(f"🎯 Starting analysis scenario: {scenario_name}")
        
        with correlated_operation(f"analysis_scenario_{scenario_name}"):
            start_time = time.time()
            
            try:
                # Run the orchestrated pipeline
                result = await self.orchestrator.orchestrate_pipeline_skill(
                    file_path=file_path,
                    pipeline_config=pipeline_config
                )
                
                duration = time.time() - start_time
                
                # Log results
                scenario_result = {
                    "scenario": scenario_name,
                    "status": result.get("status", "unknown"),
                    "duration_seconds": duration,
                    "session_id": result.get("session_id"),
                    "stages_completed": len([s for s in result.get("stages", {}).values() if s.get("status") == "completed"]),
                    "total_stages": len(result.get("stages", {})),
                    "final_report_handle": result.get("final_report_handle_id"),
                    "investigation_handle": result.get("investigation_handle_id"),
                    "pipeline_summary": result.get("pipeline_summary", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_log.append(scenario_result)
                
                if result.get("status") == "completed":
                    logger.info(f"✅ Scenario {scenario_name} completed successfully in {duration:.2f}s")
                else:
                    logger.error(f"❌ Scenario {scenario_name} failed: {result.get('error', 'Unknown error')}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"💥 Scenario {scenario_name} crashed: {e}")
                
                scenario_result = {
                    "scenario": scenario_name,
                    "status": "crashed",
                    "duration_seconds": duration,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_log.append(scenario_result)
                raise

    async def scenario_1_sales_performance_analysis(self):
        """
        Scenario 1: Sales Performance Analysis
        
        Business Question: "Analyze our sales performance across regions and products.
        Which regions and products are driving the most revenue? Are there any 
        performance outliers we should investigate?"
        """
        logger.info("📊 SCENARIO 1: Sales Performance Analysis")
        
        config = {
            "analysis_config": {
                "focus_areas": ["regional_performance", "product_analysis", "outlier_detection"],
                "metrics": ["revenue", "units_sold", "average_order_value"],
                "dimensions": ["region", "product", "sales_rep"]
            },
            "report_config": {
                "report_type": "performance_dashboard",
                "include_charts": True,
                "executive_summary": True
            }
        }
        
        return await self.run_analysis_scenario(
            scenario_name="sales_performance_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            pipeline_config=config
        )

    async def scenario_2_trend_analysis(self):
        """
        Scenario 2: Sales Trend Analysis
        
        Business Question: "What are the trends in our sales data over time?
        Are we seeing growth or decline? What seasonal patterns exist?
        Can we predict future performance?"
        """
        logger.info("📈 SCENARIO 2: Sales Trend Analysis")
        
        config = {
            "analysis_config": {
                "focus_areas": ["time_series_analysis", "seasonal_patterns", "growth_trends"],
                "metrics": ["revenue", "units_sold"],
                "time_dimension": "date",
                "trend_analysis": True
            },
            "rootcause_config": {
                "investigate_trends": True,
                "trend_threshold": 0.1
            },
            "report_config": {
                "report_type": "trend_analysis",
                "include_forecasts": True,
                "time_series_charts": True
            }
        }
        
        return await self.run_analysis_scenario(
            scenario_name="sales_trend_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            pipeline_config=config
        )

    async def scenario_3_discount_impact_investigation(self):
        """
        Scenario 3: Discount Impact Investigation
        
        Business Question: "How are our discount strategies affecting profitability?
        Are high discounts driving volume but hurting margins? What's the optimal
        discount strategy for each region/product?"
        """
        logger.info("💰 SCENARIO 3: Discount Impact Investigation")
        
        config = {
            "analysis_config": {
                "focus_areas": ["discount_analysis", "profitability_impact", "correlation_analysis"],
                "metrics": ["revenue", "units_sold", "discount_rate"],
                "correlations": [["discount_rate", "revenue"], ["discount_rate", "units_sold"]],
                "segmentation": ["region", "product"]
            },
            "rootcause_config": {
                "investigate_correlations": True,
                "focus_metric": "revenue",
                "driver_candidates": ["discount_rate", "region", "product"]
            },
            "report_config": {
                "report_type": "impact_analysis",
                "include_recommendations": True,
                "correlation_charts": True
            }
        }
        
        return await self.run_analysis_scenario(
            scenario_name="discount_impact_investigation",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            pipeline_config=config
        )

    async def analyze_test_results(self):
        """Analyze and report on test results."""
        logger.info("📋 ANALYZING TEST RESULTS")
        
        total_scenarios = len(self.results_log)
        successful_scenarios = len([r for r in self.results_log if r["status"] == "completed"])
        failed_scenarios = len([r for r in self.results_log if r["status"] in ["error", "crashed"]])
        
        total_duration = sum(r.get("duration_seconds", 0) for r in self.results_log)
        avg_duration = total_duration / total_scenarios if total_scenarios > 0 else 0
        
        logger.info(f"🎯 TEST SUMMARY:")
        logger.info(f"   Total scenarios: {total_scenarios}")
        logger.info(f"   ✅ Successful: {successful_scenarios}")
        logger.info(f"   ❌ Failed: {failed_scenarios}")
        logger.info(f"   ⏱️ Total duration: {total_duration:.2f}s")
        logger.info(f"   📊 Average duration: {avg_duration:.2f}s")
        logger.info(f"   🎉 Success rate: {(successful_scenarios/total_scenarios*100):.1f}%")
        
        # Detailed results
        for result in self.results_log:
            status_icon = "✅" if result["status"] == "completed" else "❌"
            logger.info(f"{status_icon} {result['scenario']}: {result['status']} ({result.get('duration_seconds', 0):.2f}s)")
            
            if result["status"] == "completed":
                logger.info(f"     📊 Stages: {result.get('stages_completed', 0)}/{result.get('total_stages', 0)}")
                if result.get("final_report_handle"):
                    logger.info(f"     📋 Report: {result['final_report_handle']}")
                if result.get("investigation_handle"):
                    logger.info(f"     🔍 Investigation: {result['investigation_handle']}")
            else:
                logger.error(f"     💥 Error: {result.get('error', 'Unknown')}")
        
        return {
            "total_scenarios": total_scenarios,
            "successful_scenarios": successful_scenarios,
            "failed_scenarios": failed_scenarios,
            "success_rate": successful_scenarios/total_scenarios if total_scenarios > 0 else 0,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "detailed_results": self.results_log
        }

    async def run_all_scenarios(self):
        """Run all analysis scenarios in sequence."""
        logger.info("🚀 STARTING REAL-WORLD ANALYSIS INTEGRATION TEST")
        
        await self.setup()
        
        scenarios = [
            self.scenario_1_sales_performance_analysis,
            self.scenario_2_trend_analysis,
            self.scenario_3_discount_impact_investigation
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 EXECUTING SCENARIO {i}/{len(scenarios)}")
            logger.info(f"{'='*60}")
            
            try:
                result = await scenario()
                
                # Brief pause between scenarios
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Scenario {i} failed, continuing with next scenario...")
                continue
        
        # Analyze results
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 FINAL TEST ANALYSIS")
        logger.info(f"{'='*60}")
        
        final_results = await self.analyze_test_results()
        
        # Save results to file
        results_file = Path(__file__).parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        return final_results


# Pytest integration
@pytest.mark.asyncio
async def test_real_world_analysis_scenarios():
    """Pytest wrapper for the real-world analysis test."""
    test_suite = RealWorldAnalysisTest()
    results = await test_suite.run_all_scenarios()
    
    # Assert that at least 2 out of 3 scenarios succeed
    assert results["success_rate"] >= 0.66, f"Success rate too low: {results['success_rate']:.2f}"
    assert results["successful_scenarios"] >= 2, f"Not enough successful scenarios: {results['successful_scenarios']}"


# Standalone execution
async def main():
    """Main function for standalone execution."""
    test_suite = RealWorldAnalysisTest()
    results = await test_suite.run_all_scenarios()
    
    if results["success_rate"] >= 0.66:
        logger.info("🎉 INTEGRATION TEST PASSED!")
        return 0
    else:
        logger.error("💥 INTEGRATION TEST FAILED!")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 