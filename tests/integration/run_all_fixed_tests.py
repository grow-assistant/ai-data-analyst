#!/usr/bin/env python3
"""
Fixed Test Runner - Comprehensive testing with robust error handling
This script runs all tests with improved error handling and fallback mechanisms.
"""

import asyncio
import logging
import time
import json
import sys
import httpx
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustTestRunner:
    """Robust test runner with comprehensive error handling."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.agent_health = {}
        
    async def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met before running tests."""
        logger.info("🔍 Checking test prerequisites...")
        
        # Check agent health
        agent_endpoints = {
            'orchestrator': 'http://localhost:8000',
            'data_loader': 'http://localhost:10006',
            'data_cleaning': 'http://localhost:10008',
            'data_enrichment': 'http://localhost:10009',
            'data_analyst': 'http://localhost:10007',
            'presentation': 'http://localhost:10010'
        }
        
        for agent_name, url in agent_endpoints.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{url}/health")
                    self.agent_health[agent_name] = response.status_code == 200
                    if self.agent_health[agent_name]:
                        logger.info(f"✅ {agent_name} is healthy")
                    else:
                        logger.warning(f"⚠️ {agent_name} not healthy: {response.status_code}")
            except Exception as e:
                self.agent_health[agent_name] = False
                logger.warning(f"⚠️ {agent_name} not reachable: {e}")
        
        # Check dependencies
        dependencies = {
            'pandas': False,
            'numpy': False,
            'httpx': False,
            'psutil': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
                logger.info(f"✅ {dep} available")
            except ImportError:
                logger.warning(f"⚠️ {dep} not available")
        
        return {
            'agents_healthy': sum(self.agent_health.values()),
            'total_agents': len(self.agent_health),
            'dependencies': dependencies
        }
    
    async def run_phase_1_individual_agents(self) -> Tuple[int, int]:
        """Run Phase 1: Individual Agent Testing with robust error handling."""
        logger.info("📋 Phase 1: Individual Agent Testing")
        
        passed = 0
        failed = 0
        
        # Import and run existing individual agent tests
        try:
            from test_individual_agents import AgentTester
            
            tester = AgentTester()
            results = await tester.run_all_tests()
            
            # Count results
            for agent_result in results.get("results", []):
                tests = agent_result.get("tests", [])
                agent_passed = sum(1 for test in tests if test.get("passed", False))
                agent_failed = len(tests) - agent_passed
                passed += agent_passed
                failed += agent_failed
            
            logger.info(f"📊 Phase 1 Results: {passed} passed, {failed} failed")
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 1 testing issue: {e}")
            logger.warning("⚠️ This may be expected if agents are not running")
            failed += 1
        
        return passed, failed
    
    async def run_phase_2_a2a_communication(self) -> Tuple[int, int]:
        """Run Phase 2: A2A Communication Testing."""
        logger.info("📋 Phase 2: A2A Communication Testing")
        
        passed = 0
        failed = 0
        
        try:
            from test_a2a_communication import run_all_a2a_tests
            test_passed, test_failed = await run_all_a2a_tests()
            passed += test_passed
            failed += test_failed
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 2 testing issue: {e}")
            failed += 1
        
        return passed, failed
    
    async def run_phase_3_integration(self) -> Tuple[int, int]:
        """Run Phase 3: Integration Testing with improved error recovery."""
        logger.info("📋 Phase 3: Integration Testing")
        
        passed = 0
        failed = 0
        
        try:
            from test_full_pipeline import IntegrationTester
            
            tester = IntegrationTester()
            await tester.run_all_integration_tests()
            
            # Count results from the improved error recovery
            for test_name, result in tester.test_results.items():
                if isinstance(result, dict):
                    # Check for successful completion or graceful error handling
                    if result.get('success', False) or result.get('error_recovery_success', False):
                        passed += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            
            logger.info(f"📊 Phase 3 Results: {passed} passed, {failed} failed")
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 3 testing issue: {e}")
            failed += 1
        
        return passed, failed
    
    async def run_phase_4_workflows(self) -> Tuple[int, int]:
        """Run Phase 4: End-to-End Workflow Testing with fallbacks."""
        logger.info("📋 Phase 4: End-to-End Workflow Testing")
        
        passed = 0
        failed = 0
        
        # Phase 4.1: Scheduled Workflows
        try:
            from test_scheduled_workflows import run_scheduled_workflow_tests
            p1, f1 = await run_scheduled_workflow_tests()
            passed += p1
            failed += f1
        except Exception as e:
            logger.warning(f"⚠️ Phase 4.1 issue: {e}")
            failed += 1
        
        # Phase 4.2: Production Scenarios
        try:
            from test_production_scenarios import run_production_scenario_tests
            p2, f2 = await run_production_scenario_tests()
            passed += p2
            failed += f2
        except Exception as e:
            logger.warning(f"⚠️ Phase 4.2 issue: {e}")
            failed += 1
        
        return passed, failed
    
    async def run_phase_5_security_observability(self) -> Tuple[int, int]:
        """Run Phase 5: Security and Observability Testing with fallbacks."""
        logger.info("📋 Phase 5: Security and Observability Testing")
        
        passed = 0
        failed = 0
        
        # Phase 5.1: Security
        try:
            from test_security_features import run_security_feature_tests
            p1, f1 = await run_security_feature_tests()
            passed += p1
            failed += f1
        except Exception as e:
            logger.warning(f"⚠️ Phase 5.1 issue: {e}")
            failed += 1
        
        # Phase 5.2: Observability
        try:
            from test_observability_features import run_observability_feature_tests
            p2, f2 = await run_observability_feature_tests()
            passed += p2
            failed += f2
        except Exception as e:
            logger.warning(f"⚠️ Phase 5.2 issue: {e}")
            failed += 1
        
        return passed, failed
    
    async def run_phase_6_schema_profiler(self) -> Tuple[int, int]:
        """Run Phase 6: Schema Profiler Testing."""
        logger.info("📋 Phase 6: Schema Profiler Testing")
        
        passed = 0
        failed = 0
        
        try:
            from test_schema_profiler import run_schema_profiler_tests
            p, f = await run_schema_profiler_tests()
            passed += p
            failed += f
        except Exception as e:
            logger.warning(f"⚠️ Phase 6 issue: {e}")
            failed += 1
        
        return passed, failed
    
    async def run_all_tests_robust(self) -> Dict[str, Any]:
        """Run all tests with comprehensive error handling."""
        logger.info("🚀 Starting Robust Multi-Agent Framework Testing")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Check prerequisites
        prereqs = await self.check_prerequisites()
        logger.info(f"📊 Prerequisites: {prereqs['agents_healthy']}/{prereqs['total_agents']} agents healthy")
        
        # Initialize results
        phase_results = {}
        
        # Run all phases with error handling
        phases = [
            ("Phase 1: Individual Agents", self.run_phase_1_individual_agents),
            ("Phase 2: A2A Communication", self.run_phase_2_a2a_communication),
            ("Phase 3: Integration", self.run_phase_3_integration),
            ("Phase 4: Workflows", self.run_phase_4_workflows),
            ("Phase 5: Security & Observability", self.run_phase_5_security_observability),
            ("Phase 6: Schema Profiler", self.run_phase_6_schema_profiler),
        ]
        
        total_passed = 0
        total_failed = 0
        
        for phase_name, phase_func in phases:
            logger.info(f"\n🔄 Running {phase_name}...")
            try:
                passed, failed = await phase_func()
                phase_results[phase_name] = {
                    "passed": passed,
                    "failed": failed,
                    "success_rate": (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0,
                    "status": "COMPLETED"
                }
                total_passed += passed
                total_failed += failed
                logger.info(f"✅ {phase_name}: {passed} passed, {failed} failed")
            except Exception as e:
                logger.error(f"❌ {phase_name} failed to run: {e}")
                phase_results[phase_name] = {
                    "passed": 0,
                    "failed": 1,
                    "success_rate": 0,
                    "status": "FAILED",
                    "error": str(e)
                }
                total_failed += 1
        
        self.end_time = time.time()
        
        # Generate comprehensive summary
        return {
            "test_suite": "Robust Multi-Agent Framework Testing",
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": self.end_time - self.start_time,
            "prerequisites": prereqs,
            "overall_statistics": {
                "total_tests": total_passed + total_failed,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "success_rate": (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
            },
            "phase_results": phase_results,
            "agent_health": self.agent_health
        }
    
    def print_comprehensive_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPREHENSIVE TESTING SUMMARY")
        logger.info("=" * 80)
        
        # Prerequisites
        prereqs = summary["prerequisites"]
        logger.info(f"\n🔍 PREREQUISITES:")
        logger.info(f"   Agents Healthy: {prereqs['agents_healthy']}/{prereqs['total_agents']}")
        
        # Overall results
        overall = summary["overall_statistics"]
        logger.info(f"\n📊 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {overall['total_tests']}")
        logger.info(f"   ✅ Passed: {overall['total_passed']}")
        logger.info(f"   ❌ Failed: {overall['total_failed']}")
        logger.info(f"   📈 Success Rate: {overall['success_rate']:.1f}%")
        logger.info(f"   ⏱️ Duration: {summary['duration_seconds']:.2f} seconds")
        
        # Phase breakdown
        logger.info(f"\n🔍 PHASE BREAKDOWN:")
        for phase_name, results in summary["phase_results"].items():
            status_icon = "✅" if results["status"] == "COMPLETED" else "❌"
            logger.info(f"   {status_icon} {phase_name}: {results['passed']}/{results['passed'] + results['failed']} ({results['success_rate']:.1f}%)")
            if "error" in results:
                logger.info(f"      Error: {results['error']}")
        
        # Agent health
        logger.info(f"\n🏥 AGENT HEALTH:")
        for agent_name, healthy in summary["agent_health"].items():
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            logger.info(f"   {agent_name}: {status}")
        
        # Framework assessment
        success_rate = overall['success_rate']
        if success_rate >= 80:
            logger.info(f"\n🎉 FRAMEWORK STATUS: ✅ EXCELLENT ({success_rate:.1f}%)")
            logger.info("   The Multi-Agent A2A Framework is performing excellently!")
        elif success_rate >= 60:
            logger.info(f"\n⚠️ FRAMEWORK STATUS: 🟨 GOOD ({success_rate:.1f}%)")
            logger.info("   The framework is working well with some areas for improvement.")
        elif success_rate >= 40:
            logger.info(f"\n🔧 FRAMEWORK STATUS: 🟠 NEEDS ATTENTION ({success_rate:.1f}%)")
            logger.info("   The framework has basic functionality but needs improvements.")
        else:
            logger.info(f"\n🚨 FRAMEWORK STATUS: 🔴 NEEDS SIGNIFICANT WORK ({success_rate:.1f}%)")
            logger.info("   The framework requires significant attention.")
        
        logger.info("\n" + "=" * 80)

async def main():
    """Main robust test runner."""
    runner = RobustTestRunner()
    
    try:
        # Run all tests with robust error handling
        summary = await runner.run_all_tests_robust()
        
        # Print comprehensive summary
        runner.print_comprehensive_summary(summary)
        
        # Save detailed results
        results_file = f"robust_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        # Provide actionable recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        if summary["prerequisites"]["agents_healthy"] < summary["prerequisites"]["total_agents"]:
            logger.info("   🔄 Start missing agents with: ./start_all_agents.ps1")
        
        unhealthy_agents = [name for name, healthy in summary["agent_health"].items() if not healthy]
        if unhealthy_agents:
            logger.info(f"   🏥 Check these agents: {', '.join(unhealthy_agents)}")
        
        if summary["overall_statistics"]["success_rate"] < 100:
            logger.info("   📈 Some tests failed - this may be expected in development mode")
            logger.info("   🔍 Check the detailed results for specific issues")
        
        return summary
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Testing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Testing failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 