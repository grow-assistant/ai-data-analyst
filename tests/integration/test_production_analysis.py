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
Production Analysis Integration Test

Tests the complete AI Data Analyst Multi-Agent Framework with 9 different
business analysis scenarios using real datasets and fully running agents.
"""

import asyncio
import logging
import json
import time
import subprocess
import signal
import os
from pathlib import Path
import pytest
from datetime import datetime
import httpx
import shutil

# Framework imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common_utils.enhanced_logging import get_logger, correlated_operation
from common_utils.session_manager import get_session_manager
from common_utils.observability import get_observability_manager

# Import orchestrator
orchestrator_path = Path(__file__).parent.parent.parent / "orchestrator-agent"
sys.path.insert(0, str(orchestrator_path))
from orchestrator_agent.agent_executor import OrchestratorAgentExecutor

logger = get_logger(__name__)

class ProductionAnalysisTest:
    """
    Production-level integration test for all business analysis scenarios.
    """
    
    def __init__(self):
        self.orchestrator = OrchestratorAgentExecutor()
        self.session_manager = get_session_manager()
        self.observability = get_observability_manager()
        self.test_data_path = Path(__file__).parent.parent.parent / "test_data"
        self.results_log = []
        self.agent_processes = {}
        
        # Agent configuration for startup
        self.agent_configs = [
            {"name": "data_loader", "port": 10006, "module": "data_loader", "directory": "data-loader-agent"},
            {"name": "data_cleaning", "port": 10008, "module": "data_cleaning_agent", "directory": "data-cleaning-agent"},
            {"name": "data_enrichment", "port": 10009, "module": "data_enrichment_agent", "directory": "data-enrichment-agent"},
            {"name": "data_analyst", "port": 10007, "module": "data_analyst", "directory": "data-analyst-agent"},
            {"name": "presentation", "port": 10010, "module": "presentation_agent", "directory": "presentation-agent"},
            {"name": "rootcause_analyst", "port": 10011, "module": "rootcause_analyst", "directory": "rootcause-analyst-agent"},
            {"name": "schema_profiler", "port": 10012, "module": "schema_profiler", "directory": "schema-profiler-agent"}
        ]
        
    async def setup_test_environment(self):
        """Setup the complete test environment with all agents running."""
        logger.info("🏗️ Setting up Production Test Environment")
        
        # Verify test data exists
        sales_data_file = self.test_data_path / "sales_data_small.csv"
        if not sales_data_file.exists():
            raise FileNotFoundError(f"Test data not found: {sales_data_file}")
        
        logger.info(f"✅ Test data verified: {sales_data_file}")
        
        # Start all agents
        await self.start_all_agents()
        
        # Wait for agents to be ready
        await self.wait_for_agents_ready()
        
    async def start_all_agents(self):
        """Start all required agents."""
        logger.info("🚀 Starting all agents for production testing...")
        
        parent_dir = Path(__file__).parent.parent.parent
        
        for agent_config in self.agent_configs:
            name = agent_config["name"]
            directory = agent_config["directory"]
            module = agent_config["module"]
            port = agent_config["port"]
            
            agent_path = parent_dir / directory
            
            if not agent_path.exists():
                logger.warning(f"Agent directory not found: {agent_path}")
                continue
                
            logger.info(f"🚀 Starting {name} agent on port {port}")
            
            try:
                # Start the agent using Python module execution
                process = subprocess.Popen(
                    [sys.executable, "-m", module],
                    cwd=str(agent_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )
                
                self.agent_processes[name] = process
                logger.info(f"✅ Started {name} agent (PID: {process.pid})")
                
            except Exception as e:
                logger.error(f"❌ Failed to start {name} agent: {e}")
    
    async def wait_for_agents_ready(self, max_attempts=30):
        """Wait for all agents to become healthy and ready."""
        logger.info("⏳ Waiting for all agents to become ready...")
        
        # Give agents time to start up
        await asyncio.sleep(10)
        
        ready_agents = []
        failed_agents = []
        
        for agent_config in self.agent_configs:
            name = agent_config["name"]
            port = agent_config["port"]
            url = f"http://localhost:{port}/health"
            
            logger.info(f"🔍 Checking {name} agent health...")
            
            for attempt in range(max_attempts):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(url)
                        if response.status_code == 200:
                            ready_agents.append(name)
                            logger.info(f"✅ {name} agent is ready")
                            break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        failed_agents.append(name)
                        logger.error(f"❌ {name} agent failed to become ready: {e}")
                    await asyncio.sleep(2)
        
        logger.info(f"📊 Agent Status: {len(ready_agents)}/{len(self.agent_configs)} ready")
        
        if len(ready_agents) < 4:  # Need at least core agents
            raise RuntimeError(f"Not enough agents ready. Only {len(ready_agents)} out of {len(self.agent_configs)}")
        
        return ready_agents, failed_agents
    
    async def run_analysis_scenario(self, scenario_name: str, file_path: str, 
                                  analysis_request: str, pipeline_config: dict = None) -> dict:
        """Run a complete analysis scenario and capture results."""
        logger.info(f"🎯 Starting analysis scenario: {scenario_name}")
        logger.info(f"📝 Analysis request: {analysis_request}")
        
        with correlated_operation(f"production_scenario_{scenario_name}"):
            start_time = time.time()
            
            try:
                # Run the orchestrated pipeline
                result = await self.orchestrator.orchestrate_pipeline_skill(
                    file_path=file_path,
                    pipeline_config=pipeline_config,
                    analysis_request=analysis_request
                )
                
                duration = time.time() - start_time
                
                # Extract key metrics from result
                scenario_result = {
                    "scenario": scenario_name,
                    "analysis_request": analysis_request,
                    "status": result.get("status", "unknown"),
                    "duration_seconds": duration,
                    "session_id": result.get("session_id"),
                    "stages_completed": len([s for s in result.get("stages", {}).values() if s.get("status") == "completed"]),
                    "total_stages": len(result.get("stages", {})),
                    "final_report_handle": result.get("final_report_handle_id"),
                    "investigation_handle": result.get("investigation_handle_id"),
                    "pipeline_summary": result.get("pipeline_summary", {}),
                    "data_insights": self.extract_insights_from_result(result),
                    "agent_performance": self.extract_agent_performance(result),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_log.append(scenario_result)
                
                if result.get("status") == "completed":
                    logger.info(f"✅ Scenario {scenario_name} completed successfully in {duration:.2f}s")
                    # Log key insights
                    insights = scenario_result["data_insights"]
                    logger.info(f"📊 Key insights: {insights.get('summary', 'No summary available')}")
                else:
                    logger.error(f"❌ Scenario {scenario_name} failed: {result.get('error', 'Unknown error')}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"💥 Scenario {scenario_name} crashed: {e}")
                
                scenario_result = {
                    "scenario": scenario_name,
                    "analysis_request": analysis_request,
                    "status": "crashed",
                    "duration_seconds": duration,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_log.append(scenario_result)
                raise

    def extract_insights_from_result(self, result: dict) -> dict:
        """Extract business insights from analysis result."""
        insights = {
            "summary": "Analysis completed",
            "metrics_found": [],
            "trends_identified": [],
            "anomalies": [],
            "recommendations": []
        }
        
        # Try to extract insights from different parts of the result
        if "analysis_summary" in result:
            insights["summary"] = result["analysis_summary"]
        
        if "findings" in result:
            findings = result["findings"]
            if isinstance(findings, dict):
                insights["metrics_found"] = findings.get("metrics", [])
                insights["trends_identified"] = findings.get("trends", [])
                insights["anomalies"] = findings.get("anomalies", [])
        
        return insights
    
    def extract_agent_performance(self, result: dict) -> dict:
        """Extract agent performance metrics from result."""
        performance = {
            "agents_used": [],
            "processing_times": {},
            "data_quality_score": None,
            "analysis_confidence": None
        }
        
        # Extract from stages
        stages = result.get("stages", {})
        for stage_name, stage_data in stages.items():
            if stage_data.get("status") == "completed":
                performance["agents_used"].append(stage_name)
                if "duration" in stage_data:
                    performance["processing_times"][stage_name] = stage_data["duration"]
        
        return performance

    # 9 Different Analysis Scenarios
    
    async def scenario_1_sales_performance_analysis(self):
        """Revenue Performance Analysis - Core business metrics."""
        analysis_request = """
        Analyze our sales performance across all dimensions. I need to understand:
        1. Total revenue and growth trends over time
        2. Performance by region - which regions are driving the most revenue?
        3. Product performance - which products are our top performers?
        4. Sales rep effectiveness - are there performance outliers?
        5. Seasonal patterns in the data
        Please provide actionable insights and recommendations.
        """
        
        config = {
            "analysis_focus": "revenue_performance",
            "metrics": ["revenue", "units_sold", "growth_rate"],
            "dimensions": ["region", "product", "sales_rep", "date"],
            "analysis_types": ["trends", "segmentation", "ranking"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="sales_performance_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_2_discount_impact_analysis(self):
        """Discount Strategy Analysis - Pricing optimization."""
        analysis_request = """
        Analyze the impact of our discount strategies on profitability and volume:
        1. How do different discount levels affect sales volume and revenue?
        2. What's the optimal discount rate for each product/region?
        3. Are we over-discounting in any segments?
        4. Correlation between discount rates and customer acquisition
        5. ROI analysis of discount campaigns
        Provide recommendations for discount optimization.
        """
        
        config = {
            "analysis_focus": "discount_impact",
            "metrics": ["discount_rate", "revenue", "units_sold", "profit_margin"],
            "dimensions": ["product", "region", "customer_segment"],
            "analysis_types": ["correlation", "optimization", "elasticity"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="discount_impact_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_3_customer_segmentation_analysis(self):
        """Customer Behavior Analysis - Segmentation and targeting."""
        analysis_request = """
        Perform customer segmentation analysis to understand our customer base:
        1. Identify distinct customer segments based on purchasing behavior
        2. Calculate customer lifetime value by segment
        3. Purchase frequency and recency patterns
        4. Geographic concentration of high-value customers
        5. Product affinity by customer segment
        Recommend targeted marketing strategies for each segment.
        """
        
        config = {
            "analysis_focus": "customer_segmentation",
            "metrics": ["customer_value", "purchase_frequency", "recency"],
            "dimensions": ["customer_id", "region", "product"],
            "analysis_types": ["clustering", "behavioral", "value_based"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="customer_segmentation_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_4_time_series_forecasting(self):
        """Predictive Analytics - Sales forecasting."""
        analysis_request = """
        Generate sales forecasts and predictive insights:
        1. 3-month revenue forecast by region and product
        2. Identify seasonal trends and cyclical patterns
        3. Predict which products will have highest growth
        4. Risk assessment for revenue targets
        5. External factors that might impact forecasts
        Provide confidence intervals and scenario planning.
        """
        
        config = {
            "analysis_focus": "forecasting",
            "metrics": ["revenue", "units_sold"],
            "time_dimension": "date",
            "forecast_horizon": 90,
            "analysis_types": ["time_series", "predictive", "trend_analysis"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="time_series_forecasting",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_5_anomaly_detection(self):
        """Anomaly Detection - Identifying unusual patterns."""
        analysis_request = """
        Detect anomalies and unusual patterns in our sales data:
        1. Identify unusual sales spikes or drops
        2. Detect outlier transactions that need investigation
        3. Find regional performance anomalies
        4. Sales rep performance outliers (both positive and negative)
        5. Product performance anomalies
        Provide root cause analysis for significant anomalies found.
        """
        
        config = {
            "analysis_focus": "anomaly_detection",
            "metrics": ["revenue", "units_sold", "discount_rate"],
            "dimensions": ["region", "product", "sales_rep", "date"],
            "analysis_types": ["outlier_detection", "statistical_anomalies", "pattern_breaks"],
            "rootcause_config": {"investigate_anomalies": True}
        }
        
        return await self.run_analysis_scenario(
            scenario_name="anomaly_detection",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_6_competitive_analysis(self):
        """Market Position Analysis - Competitive benchmarking."""
        analysis_request = """
        Analyze our market position and competitive landscape:
        1. Market share by region and product category
        2. Performance vs. industry benchmarks
        3. Price positioning relative to market standards
        4. Growth rate comparison with market trends
        5. Identify competitive threats and opportunities
        Recommend strategic actions to improve market position.
        """
        
        config = {
            "analysis_focus": "competitive_analysis",
            "metrics": ["market_share", "price_position", "growth_rate"],
            "dimensions": ["region", "product", "competitor"],
            "analysis_types": ["benchmarking", "positioning", "competitive"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="competitive_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_7_profitability_analysis(self):
        """Profitability Deep Dive - Margin analysis."""
        analysis_request = """
        Conduct comprehensive profitability analysis:
        1. Profit margins by product, region, and sales channel
        2. Cost structure analysis and cost drivers
        3. Identify most and least profitable segments
        4. Break-even analysis for different scenarios
        5. Impact of discounting on overall profitability
        Recommend actions to improve overall profitability.
        """
        
        config = {
            "analysis_focus": "profitability",
            "metrics": ["profit_margin", "cost_ratio", "break_even"],
            "dimensions": ["product", "region", "sales_channel"],
            "analysis_types": ["margin_analysis", "cost_analysis", "break_even"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="profitability_analysis",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_8_inventory_optimization(self):
        """Inventory Management - Stock optimization."""
        analysis_request = """
        Optimize inventory management based on sales patterns:
        1. Inventory turnover rates by product and region
        2. Demand forecasting for inventory planning
        3. Identify slow-moving and fast-moving inventory
        4. Seasonal inventory requirements
        5. Safety stock recommendations
        Provide inventory optimization recommendations.
        """
        
        config = {
            "analysis_focus": "inventory_optimization",
            "metrics": ["inventory_turnover", "demand_forecast", "stock_levels"],
            "dimensions": ["product", "region", "date"],
            "analysis_types": ["turnover_analysis", "demand_planning", "optimization"]
        }
        
        return await self.run_analysis_scenario(
            scenario_name="inventory_optimization",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def scenario_9_executive_dashboard(self):
        """Executive Summary - High-level KPIs and insights."""
        analysis_request = """
        Create an executive dashboard with key business insights:
        1. Overall business health and performance summary
        2. Key performance indicators and trend analysis
        3. Critical issues requiring immediate attention
        4. Growth opportunities and strategic recommendations
        5. Risk assessment and mitigation strategies
        Provide a concise executive summary suitable for C-level presentation.
        """
        
        config = {
            "analysis_focus": "executive_summary",
            "metrics": ["total_revenue", "growth_rate", "market_share", "profitability"],
            "dimensions": ["overall", "strategic"],
            "analysis_types": ["summary", "strategic", "executive"],
            "report_format": "executive_dashboard"
        }
        
        return await self.run_analysis_scenario(
            scenario_name="executive_dashboard",
            file_path=str(self.test_data_path / "sales_data_small.csv"),
            analysis_request=analysis_request,
            pipeline_config=config
        )

    async def analyze_test_results(self):
        """Analyze and critique the test results."""
        logger.info("📋 ANALYZING PRODUCTION TEST RESULTS")
        
        total_scenarios = len(self.results_log)
        successful_scenarios = len([r for r in self.results_log if r["status"] == "completed"])
        failed_scenarios = len([r for r in self.results_log if r["status"] in ["error", "crashed"]])
        
        total_duration = sum(r.get("duration_seconds", 0) for r in self.results_log)
        avg_duration = total_duration / total_scenarios if total_scenarios > 0 else 0
        
        # Detailed analysis
        analysis_critique = {
            "overall_performance": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "failed_scenarios": failed_scenarios,
                "success_rate": successful_scenarios/total_scenarios if total_scenarios > 0 else 0,
                "total_duration": total_duration,
                "average_duration": avg_duration
            },
            "agent_performance": self.analyze_agent_performance(),
            "data_quality": self.analyze_data_quality(),
            "business_insights": self.analyze_business_insights(),
            "system_reliability": self.analyze_system_reliability(),
            "recommendations": self.generate_recommendations()
        }
        
        # Log detailed critique
        logger.info(f"🎯 PRODUCTION TEST CRITIQUE:")
        logger.info(f"   📊 Success Rate: {(successful_scenarios/total_scenarios*100):.1f}%")
        logger.info(f"   ⏱️ Average Analysis Time: {avg_duration:.2f}s")
        logger.info(f"   🔍 Data Quality: {analysis_critique['data_quality']['overall_score']:.1f}/10")
        logger.info(f"   🤖 Agent Reliability: {analysis_critique['system_reliability']['agent_uptime']:.1f}%")
        
        return analysis_critique

    def analyze_agent_performance(self) -> dict:
        """Analyze individual agent performance."""
        agent_stats = {}
        total_calls = 0
        successful_calls = 0
        
        for result in self.results_log:
            if result["status"] == "completed":
                agent_perf = result.get("agent_performance", {})
                for agent in agent_perf.get("agents_used", []):
                    if agent not in agent_stats:
                        agent_stats[agent] = {"calls": 0, "successes": 0, "avg_time": 0}
                    agent_stats[agent]["calls"] += 1
                    agent_stats[agent]["successes"] += 1
                    total_calls += 1
                    successful_calls += 1
        
        return {
            "agent_statistics": agent_stats,
            "overall_success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "most_reliable_agent": max(agent_stats.keys(), key=lambda x: agent_stats[x]["successes"]) if agent_stats else None
        }

    def analyze_data_quality(self) -> dict:
        """Analyze data quality and processing effectiveness."""
        successful_results = [r for r in self.results_log if r["status"] == "completed"]
        
        data_completeness = len(successful_results) / len(self.results_log) if self.results_log else 0
        
        return {
            "overall_score": data_completeness * 10,
            "processing_success_rate": data_completeness,
            "data_coverage": "comprehensive" if data_completeness > 0.8 else "partial",
            "insights_quality": "high" if len(successful_results) > 6 else "medium"
        }

    def analyze_business_insights(self) -> dict:
        """Analyze the quality and usefulness of business insights generated."""
        insights_found = []
        
        for result in self.results_log:
            if result["status"] == "completed":
                insights = result.get("data_insights", {})
                if insights.get("summary") != "Analysis completed":
                    insights_found.append(insights)
        
        return {
            "total_insights": len(insights_found),
            "insight_categories": ["revenue", "profitability", "customer", "competitive"],
            "actionable_recommendations": len(insights_found) * 2,  # Estimate
            "business_value": "high" if len(insights_found) > 5 else "medium"
        }

    def analyze_system_reliability(self) -> dict:
        """Analyze system reliability and stability."""
        total_stages = sum(r.get("total_stages", 0) for r in self.results_log)
        completed_stages = sum(r.get("stages_completed", 0) for r in self.results_log)
        
        avg_duration = sum(r.get("duration_seconds", 0) for r in self.results_log) / len(self.results_log) if self.results_log else 0
        
        return {
            "agent_uptime": (completed_stages / total_stages * 100) if total_stages > 0 else 0,
            "error_rate": len([r for r in self.results_log if r["status"] != "completed"]) / len(self.results_log) * 100 if self.results_log else 0,
            "system_stability": "stable" if completed_stages / total_stages > 0.8 else "needs_improvement",
            "scalability": "good" if avg_duration < 60 else "needs_optimization"
        }

    def generate_recommendations(self) -> list:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        success_rate = len([r for r in self.results_log if r["status"] == "completed"]) / len(self.results_log)
        
        if success_rate < 0.8:
            recommendations.append("Improve agent error handling and retry mechanisms")
        
        avg_duration = sum(r.get("duration_seconds", 0) for r in self.results_log) / len(self.results_log)
        if avg_duration > 30:
            recommendations.append("Optimize agent response times for better user experience")
        
        recommendations.append("Implement caching for frequently requested analysis types")
        recommendations.append("Add more sophisticated business intelligence algorithms")
        recommendations.append("Enhance natural language understanding for analysis requests")
        
        return recommendations

    def stop_all_agents(self):
        """Stop all running agents."""
        logger.info("🛑 Stopping all agents...")
        
        for name, process in self.agent_processes.items():
            try:
                logger.info(f"Stopping {name} agent (PID: {process.pid})")
                
                if os.name == 'nt':  # Windows
                    process.terminate()
                else:  # Unix/Linux
                    process.terminate()
                
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} agent stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {name} agent")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"Error stopping {name} agent: {e}")
        
        self.agent_processes.clear()
        logger.info("🛑 All agents stopped")

    async def run_all_production_scenarios(self):
        """Run all 9 production analysis scenarios."""
        logger.info("🚀 STARTING PRODUCTION ANALYSIS INTEGRATION TEST")
        
        try:
            # Setup environment
            await self.setup_test_environment()
            
            # All 9 analysis scenarios
            scenarios = [
                self.scenario_1_sales_performance_analysis,
                self.scenario_2_discount_impact_analysis,
                self.scenario_3_customer_segmentation_analysis,
                self.scenario_4_time_series_forecasting,
                self.scenario_5_anomaly_detection,
                self.scenario_6_competitive_analysis,
                self.scenario_7_profitability_analysis,
                self.scenario_8_inventory_optimization,
                self.scenario_9_executive_dashboard
            ]
            
            for i, scenario in enumerate(scenarios, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"🎯 EXECUTING PRODUCTION SCENARIO {i}/{len(scenarios)}: {scenario.__name__}")
                logger.info(f"{'='*80}")
                
                try:
                    result = await scenario()
                    
                    # Brief pause between scenarios
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Scenario {i} failed: {e}")
                    continue
            
            # Comprehensive analysis and critique
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 COMPREHENSIVE PRODUCTION ANALYSIS & CRITIQUE")
            logger.info(f"{'='*80}")
            
            critique = await self.analyze_test_results()
            
            # Save detailed results
            results_file = Path(__file__).parent / f"production_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({"test_results": self.results_log, "critique": critique}, f, indent=2, default=str)
            
            logger.info(f"📄 Detailed production test results saved to: {results_file}")
            
            return critique
                
        finally:
            # Always stop agents
            self.stop_all_agents()


# Pytest integration
@pytest.mark.asyncio
async def test_production_analysis_scenarios():
    """Pytest wrapper for the production analysis test."""
    test_suite = ProductionAnalysisTest()
    results = await test_suite.run_all_production_scenarios()
    
    # Assert reasonable success rate for production system
    assert results["overall_performance"]["success_rate"] >= 0.6, f"Production success rate too low: {results['overall_performance']['success_rate']:.2f}"


# Standalone execution
async def main():
    """Main function for standalone execution."""
    test_suite = ProductionAnalysisTest()
    results = await test_suite.run_all_production_scenarios()
    
    success_rate = results["overall_performance"]["success_rate"]
    if success_rate >= 0.7:
        logger.info("🎉 PRODUCTION INTEGRATION TEST PASSED!")
        return 0
    else:
        logger.error("💥 PRODUCTION INTEGRATION TEST NEEDS IMPROVEMENT!")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 