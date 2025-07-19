#!/usr/bin/env python3
"""
Comprehensive Integration Test Framework
Tests multiple analysis types across all datasets and improves the system iteratively
"""

import asyncio
import httpx
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from common_utils.output_manager import output_manager

class FrameworkIntegrationTester:
    """Comprehensive integration tester for the multi-agent framework"""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:10000"
        self.test_results = []
        self.current_test = None
        
        # Define test scenarios
        self.analysis_types = [
            {
                "name": "Trend Analysis",
                "query": "Analyze trends and patterns in the data, identify key drivers of performance",
                "config": {"analysis_depth": "comprehensive", "include_root_cause": True}
            },
            {
                "name": "Performance Analysis", 
                "query": "Find top and bottom performers, analyze what makes them successful or problematic",
                "config": {"analysis_depth": "detailed", "include_root_cause": False}
            },
            {
                "name": "Outlier Detection",
                "query": "Detect anomalies and outliers, investigate potential data quality issues",
                "config": {"analysis_depth": "basic", "include_root_cause": True}
            }
        ]
        
        # Define datasets to test
        self.datasets = [
            "Superstore.csv",
            # Add other datasets as they become available
            # "M5.csv",
            # "AI_DS.tdsx"
        ]
    
    async def check_system_health(self) -> Dict[str, bool]:
        """Check if all agents are healthy"""
        agents = {
            "orchestrator": 10000,
            "data_loader": 10006,
            "data_analyst": 10007, 
            "data_cleaning": 10008,
            "data_enrichment": 10009,
            "presentation": 10010,
            "rootcause_analyst": 10011,
            "schema_profiler": 10012
        }
        
        health_status = {}
        print("🏥 Checking system health...")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for agent_name, port in agents.items():
                try:
                    response = await client.get(f"http://localhost:{port}/health")
                    is_healthy = response.status_code == 200
                    health_status[agent_name] = is_healthy
                    status_icon = "✅" if is_healthy else "❌"
                    print(f"   {status_icon} {agent_name}: port {port}")
                except Exception as e:
                    health_status[agent_name] = False
                    print(f"   ❌ {agent_name}: port {port} - {str(e)}")
        
        all_healthy = all(health_status.values())
        print(f"\n🎯 Overall system health: {'✅ HEALTHY' if all_healthy else '❌ ISSUES DETECTED'}")
        return health_status
    
    async def run_analysis(self, dataset: str, analysis_type: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single analysis and return results"""
        print(f"\n🔄 Running {analysis_type['name']} on {dataset}")
        print(f"   Query: {analysis_type['query']}")
        
        # Create session for this test
        session_id = f"test_{dataset.replace('.', '_')}_{analysis_type['name'].replace(' ', '_').lower()}_{int(time.time())}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "orchestrate_pipeline",
            "params": {
                "file_path": f"data/{dataset}",
                "pipeline_config": {
                    "analysis_config": {
                        "query": analysis_type['query'],
                        **analysis_type['config']
                    },
                    "session_id": session_id
                }
            },
            "id": 1
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
                print(f"   📡 Sending request to orchestrator...")
                response = await client.post(self.orchestrator_url, json=payload)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "duration": time.time() - start_time
                    }
                
                result = response.json()
                duration = time.time() - start_time
                
                if "error" in result:
                    return {
                        "status": "error", 
                        "error": result["error"],
                        "duration": duration
                    }
                
                # Extract the actual result
                analysis_result = result.get("result", {})
                analysis_result["duration"] = duration
                analysis_result["session_id"] = session_id
                
                return analysis_result
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Exception: {str(e)}",
                "traceback": traceback.format_exc(),
                "duration": time.time() - start_time
            }
    
    def analyze_results(self, dataset: str, analysis_type: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results and identify issues"""
        analysis = {
            "dataset": dataset,
            "analysis_type": analysis_type["name"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "issues": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # Check basic success criteria
        status = results.get("status", "unknown")
        duration = results.get("duration", 0)
        
        analysis["metrics"]["duration_seconds"] = duration
        analysis["metrics"]["status"] = status
        
        if status == "error":
            analysis["issues"].append(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            if "traceback" in results:
                analysis["issues"].append(f"Traceback: {results['traceback']}")
        elif status == "success":
            analysis["success"] = True
            
            # Check for pipeline stages
            stages_completed = results.get("stages_completed", [])
            analysis["metrics"]["stages_completed"] = len(stages_completed)
            
            if len(stages_completed) == 0:
                analysis["issues"].append("No pipeline stages completed")
            elif len(stages_completed) < 3:
                analysis["issues"].append(f"Only {len(stages_completed)} stages completed (expected 3+)")
            
            # Check for data quality
            data_quality = results.get("data_quality_score", 0)
            analysis["metrics"]["data_quality_score"] = data_quality
            
            if data_quality < 0.7:
                analysis["issues"].append(f"Low data quality score: {data_quality}")
            
            # Check for insights
            insights = results.get("insights", [])
            analysis["metrics"]["insights_count"] = len(insights)
            
            if len(insights) == 0:
                analysis["issues"].append("No insights generated")
            
        else:
            analysis["issues"].append(f"Unknown status: {status}")
        
        # Performance analysis
        if duration > 120:  # 2 minutes
            analysis["issues"].append(f"Slow execution: {duration:.1f} seconds")
        elif duration > 60:  # 1 minute
            analysis["recommendations"].append(f"Consider optimization - took {duration:.1f} seconds")
        
        # Generate recommendations based on issues
        if "Pipeline failed" in str(analysis["issues"]):
            analysis["recommendations"].append("Check agent connectivity and logs")
            analysis["recommendations"].append("Verify orchestrator is properly routing requests")
        
        if "No pipeline stages completed" in str(analysis["issues"]):
            analysis["recommendations"].append("Debug orchestrator pipeline execution logic")
            analysis["recommendations"].append("Check individual agent functionality")
        
        return analysis
    
    def suggest_improvements(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Suggest specific code improvements based on test results"""
        improvements = []
        
        # Analyze patterns across all tests
        total_tests = len(analyses)
        failed_tests = sum(1 for a in analyses if not a["success"])
        
        if failed_tests == total_tests:
            improvements.append("CRITICAL: All tests failing - check basic system connectivity")
            improvements.append("Verify all agents are starting properly")
            improvements.append("Check orchestrator JSON-RPC endpoint implementation")
        
        elif failed_tests > total_tests / 2:
            improvements.append("MAJOR: High failure rate - investigate common failure patterns")
        
        # Check for specific issues
        common_issues = {}
        for analysis in analyses:
            for issue in analysis["issues"]:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        for issue, count in common_issues.items():
            if count > 1:
                improvements.append(f"RECURRING ISSUE ({count}x): {issue}")
        
        # Performance improvements
        avg_duration = sum(a["metrics"].get("duration_seconds", 0) for a in analyses) / total_tests if total_tests > 0 else 0
        if avg_duration > 60:
            improvements.append(f"PERFORMANCE: Average execution time {avg_duration:.1f}s - consider optimization")
        
        return improvements
    
    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Framework Integration Test")
        print("=" * 70)
        
        # Check system health first
        health_status = await self.check_system_health()
        if not all(health_status.values()):
            print("\n❌ System health check failed. Please fix agent connectivity before proceeding.")
            return False
        
        all_analyses = []
        
        # Test each dataset with each analysis type
        for dataset in self.datasets:
            print(f"\n📊 Testing Dataset: {dataset}")
            print("-" * 50)
            
            for analysis_type in self.analysis_types:
                # Run the analysis
                results = await self.run_analysis(dataset, analysis_type)
                
                # Analyze results
                analysis = self.analyze_results(dataset, analysis_type, results)
                all_analyses.append(analysis)
                
                # Report results
                status_icon = "✅" if analysis["success"] else "❌"
                duration = analysis["metrics"]["duration_seconds"]
                print(f"   {status_icon} {analysis_type['name']}: {duration:.1f}s")
                
                if analysis["issues"]:
                    for issue in analysis["issues"]:
                        print(f"      ⚠️  {issue}")
                
                # If this test failed, suggest immediate improvements
                if not analysis["success"]:
                    print(f"      💡 Recommendations:")
                    for rec in analysis["recommendations"]:
                        print(f"         - {rec}")
                    
                    # For now, continue with next test rather than stopping
                    # In a full implementation, you might want to fix issues here
                    print(f"      ⏭️  Continuing to next test...")
                
                print()  # Empty line for readability
        
        # Generate overall report
        self.generate_final_report(all_analyses)
        
        return all_analyses
    
    def generate_final_report(self, analyses: List[Dict[str, Any]]):
        """Generate final test report with recommendations"""
        print("\n" + "=" * 70)
        print("📋 FINAL TEST REPORT")
        print("=" * 70)
        
        total_tests = len(analyses)
        successful_tests = sum(1 for a in analyses if a["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Successful: {successful_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if successful_tests > 0:
            avg_duration = sum(a["metrics"]["duration_seconds"] for a in analyses if a["success"]) / successful_tests
            print(f"   ⏱️  Average Duration: {avg_duration:.1f}s")
        
        # Show detailed results by category
        print(f"\n📋 Results by Analysis Type:")
        for analysis_type in self.analysis_types:
            type_analyses = [a for a in analyses if a["analysis_type"] == analysis_type["name"]]
            type_success = sum(1 for a in type_analyses if a["success"])
            print(f"   {analysis_type['name']}: {type_success}/{len(type_analyses)} successful")
        
        print(f"\n📋 Results by Dataset:")
        for dataset in self.datasets:
            dataset_analyses = [a for a in analyses if a["dataset"] == dataset]
            dataset_success = sum(1 for a in dataset_analyses if a["success"])
            print(f"   {dataset}: {dataset_success}/{len(dataset_analyses)} successful")
        
        # Generate improvement suggestions
        improvements = self.suggest_improvements(analyses)
        if improvements:
            print(f"\n💡 Recommended Improvements:")
            for i, improvement in enumerate(improvements, 1):
                print(f"   {i}. {improvement}")
        
        # Save detailed results
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": (successful_tests/total_tests)*100 if total_tests > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": analyses,
            "improvements": improvements
        }
        
        report_file = f"test_results_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {report_file}")
        
        if successful_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Framework is working correctly.")
        elif successful_tests > 0:
            print(f"\n⚠️  PARTIAL SUCCESS: {failed_tests} tests need attention.")
        else:
            print(f"\n🚨 ALL TESTS FAILED: Framework requires significant fixes.")

async def main():
    """Main test runner"""
    tester = FrameworkIntegrationTester()
    await tester.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main()) 