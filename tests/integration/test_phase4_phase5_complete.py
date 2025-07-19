#!/usr/bin/env python3
"""
Complete Phase 4 & Phase 5 Test Runner
Runs all remaining tests to complete the Multi-Agent A2A Framework testing plan.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple

# Import test modules
from test_scheduled_workflows import run_scheduled_workflow_tests
from test_production_scenarios import run_production_scenario_tests
from test_security_features import run_security_feature_tests
from test_observability_features import run_observability_feature_tests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4Phase5TestRunner:
    """Complete test runner for Phase 4 and Phase 5 testing."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 4 and Phase 5 tests."""
        logger.info("🚀 Starting Complete Phase 4 & Phase 5 Testing Suite")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Phase 4: End-to-End Workflow Testing
        logger.info("\n📋 PHASE 4: END-TO-END WORKFLOW TESTING")
        logger.info("=" * 50)
        
        # Phase 4.1: Scheduled Workflow Testing
        logger.info("\n🕐 Phase 4.1: Scheduled Workflow Testing")
        logger.info("-" * 40)
        
        try:
            phase_4_1_passed, phase_4_1_failed = await run_scheduled_workflow_tests()
            self.results["phase_4_1_scheduled_workflows"] = {
                "passed": phase_4_1_passed,
                "failed": phase_4_1_failed,
                "success_rate": (phase_4_1_passed / (phase_4_1_passed + phase_4_1_failed)) * 100 if (phase_4_1_passed + phase_4_1_failed) > 0 else 0,
                "status": "COMPLETED"
            }
            logger.info(f"📊 Phase 4.1 Results: {phase_4_1_passed} passed, {phase_4_1_failed} failed")
        except Exception as e:
            logger.error(f"❌ Phase 4.1 failed to run: {e}")
            self.results["phase_4_1_scheduled_workflows"] = {
                "passed": 0,
                "failed": 1,
                "success_rate": 0,
                "status": "FAILED",
                "error": str(e)
            }
        
        # Phase 4.2: Production Scenario Testing
        logger.info("\n🏭 Phase 4.2: Production Scenario Testing")
        logger.info("-" * 40)
        
        try:
            phase_4_2_passed, phase_4_2_failed = await run_production_scenario_tests()
            self.results["phase_4_2_production_scenarios"] = {
                "passed": phase_4_2_passed,
                "failed": phase_4_2_failed,
                "success_rate": (phase_4_2_passed / (phase_4_2_passed + phase_4_2_failed)) * 100 if (phase_4_2_passed + phase_4_2_failed) > 0 else 0,
                "status": "COMPLETED"
            }
            logger.info(f"📊 Phase 4.2 Results: {phase_4_2_passed} passed, {phase_4_2_failed} failed")
        except Exception as e:
            logger.error(f"❌ Phase 4.2 failed to run: {e}")
            self.results["phase_4_2_production_scenarios"] = {
                "passed": 0,
                "failed": 1,
                "success_rate": 0,
                "status": "FAILED",
                "error": str(e)
            }
        
        # Phase 5: Security and Observability Testing
        logger.info("\n🔐 PHASE 5: SECURITY AND OBSERVABILITY TESTING")
        logger.info("=" * 50)
        
        # Phase 5.1: Security Testing
        logger.info("\n🛡️ Phase 5.1: Security Testing")
        logger.info("-" * 40)
        
        try:
            phase_5_1_passed, phase_5_1_failed = await run_security_feature_tests()
            self.results["phase_5_1_security"] = {
                "passed": phase_5_1_passed,
                "failed": phase_5_1_failed,
                "success_rate": (phase_5_1_passed / (phase_5_1_passed + phase_5_1_failed)) * 100 if (phase_5_1_passed + phase_5_1_failed) > 0 else 0,
                "status": "COMPLETED"
            }
            logger.info(f"📊 Phase 5.1 Results: {phase_5_1_passed} passed, {phase_5_1_failed} failed")
        except Exception as e:
            logger.error(f"❌ Phase 5.1 failed to run: {e}")
            self.results["phase_5_1_security"] = {
                "passed": 0,
                "failed": 1,
                "success_rate": 0,
                "status": "FAILED",
                "error": str(e)
            }
        
        # Phase 5.2: Observability Testing
        logger.info("\n📊 Phase 5.2: Observability Testing")
        logger.info("-" * 40)
        
        try:
            phase_5_2_passed, phase_5_2_failed = await run_observability_feature_tests()
            self.results["phase_5_2_observability"] = {
                "passed": phase_5_2_passed,
                "failed": phase_5_2_failed,
                "success_rate": (phase_5_2_passed / (phase_5_2_passed + phase_5_2_failed)) * 100 if (phase_5_2_passed + phase_5_2_failed) > 0 else 0,
                "status": "COMPLETED"
            }
            logger.info(f"📊 Phase 5.2 Results: {phase_5_2_passed} passed, {phase_5_2_failed} failed")
        except Exception as e:
            logger.error(f"❌ Phase 5.2 failed to run: {e}")
            self.results["phase_5_2_observability"] = {
                "passed": 0,
                "failed": 1,
                "success_rate": 0,
                "status": "FAILED",
                "error": str(e)
            }
        
        self.end_time = time.time()
        
        # Generate comprehensive summary
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_duration = self.end_time - self.start_time
        
        # Calculate overall statistics
        total_passed = sum(phase["passed"] for phase in self.results.values())
        total_failed = sum(phase["failed"] for phase in self.results.values())
        total_tests = total_passed + total_failed
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Phase-wise statistics
        phase_4_passed = (
            self.results["phase_4_1_scheduled_workflows"]["passed"] + 
            self.results["phase_4_2_production_scenarios"]["passed"]
        )
        phase_4_failed = (
            self.results["phase_4_1_scheduled_workflows"]["failed"] + 
            self.results["phase_4_2_production_scenarios"]["failed"]
        )
        phase_4_success_rate = (phase_4_passed / (phase_4_passed + phase_4_failed) * 100) if (phase_4_passed + phase_4_failed) > 0 else 0
        
        phase_5_passed = (
            self.results["phase_5_1_security"]["passed"] + 
            self.results["phase_5_2_observability"]["passed"]
        )
        phase_5_failed = (
            self.results["phase_5_1_security"]["failed"] + 
            self.results["phase_5_2_observability"]["failed"]
        )
        phase_5_success_rate = (phase_5_passed / (phase_5_passed + phase_5_failed) * 100) if (phase_5_passed + phase_5_failed) > 0 else 0
        
        summary = {
            "test_suite": "Phase 4 & Phase 5 Complete Testing",
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": total_duration,
            "overall_statistics": {
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "success_rate": overall_success_rate
            },
            "phase_4_statistics": {
                "total_tests": phase_4_passed + phase_4_failed,
                "passed": phase_4_passed,
                "failed": phase_4_failed,
                "success_rate": phase_4_success_rate,
                "status": "COMPLETED" if phase_4_passed > 0 else "FAILED"
            },
            "phase_5_statistics": {
                "total_tests": phase_5_passed + phase_5_failed,
                "passed": phase_5_passed,
                "failed": phase_5_failed,
                "success_rate": phase_5_success_rate,
                "status": "COMPLETED" if phase_5_passed > 0 else "FAILED"
            },
            "detailed_results": self.results
        }
        
        return summary
    
    def print_final_summary(self, summary: Dict[str, Any]):
        """Print comprehensive final summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 4 & PHASE 5 TESTING COMPLETE!")
        logger.info("=" * 80)
        
        # Overall statistics
        overall = summary["overall_statistics"]
        logger.info(f"\n📊 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {overall['total_tests']}")
        logger.info(f"   ✅ Passed: {overall['total_passed']}")
        logger.info(f"   ❌ Failed: {overall['total_failed']}")
        logger.info(f"   📈 Success Rate: {overall['success_rate']:.1f}%")
        logger.info(f"   ⏱️ Duration: {summary['duration_seconds']:.2f} seconds")
        
        # Phase 4 statistics
        phase_4 = summary["phase_4_statistics"]
        logger.info(f"\n📋 PHASE 4 - END-TO-END WORKFLOW TESTING:")
        logger.info(f"   Status: {'✅ ' + phase_4['status'] if phase_4['status'] == 'COMPLETED' else '❌ ' + phase_4['status']}")
        logger.info(f"   Tests: {phase_4['total_tests']}")
        logger.info(f"   Passed: {phase_4['passed']}")
        logger.info(f"   Failed: {phase_4['failed']}")
        logger.info(f"   Success Rate: {phase_4['success_rate']:.1f}%")
        
        # Phase 5 statistics
        phase_5 = summary["phase_5_statistics"]
        logger.info(f"\n🔐 PHASE 5 - SECURITY AND OBSERVABILITY TESTING:")
        logger.info(f"   Status: {'✅ ' + phase_5['status'] if phase_5['status'] == 'COMPLETED' else '❌ ' + phase_5['status']}")
        logger.info(f"   Tests: {phase_5['total_tests']}")
        logger.info(f"   Passed: {phase_5['passed']}")
        logger.info(f"   Failed: {phase_5['failed']}")
        logger.info(f"   Success Rate: {phase_5['success_rate']:.1f}%")
        
        # Detailed breakdown
        logger.info(f"\n🔍 DETAILED BREAKDOWN:")
        for phase_name, results in summary["detailed_results"].items():
            status_icon = "✅" if results["status"] == "COMPLETED" else "❌"
            logger.info(f"   {status_icon} {phase_name}: {results['passed']}/{results['passed'] + results['failed']} ({results['success_rate']:.1f}%)")
        
        # Framework status
        if overall['success_rate'] >= 80:
            logger.info(f"\n🎊 FRAMEWORK STATUS: ✅ PRODUCTION READY")
            logger.info(f"   The Multi-Agent A2A Framework has achieved {overall['success_rate']:.1f}% test success rate!")
        elif overall['success_rate'] >= 60:
            logger.info(f"\n⚠️ FRAMEWORK STATUS: 🟨 MOSTLY READY")
            logger.info(f"   The framework shows {overall['success_rate']:.1f}% success rate with some areas for improvement.")
        else:
            logger.info(f"\n🔧 FRAMEWORK STATUS: 🔴 NEEDS WORK")
            logger.info(f"   The framework needs attention with {overall['success_rate']:.1f}% success rate.")
        
        logger.info("\n" + "=" * 80)

async def main():
    """Main test runner."""
    test_runner = Phase4Phase5TestRunner()
    
    try:
        # Run all tests
        summary = await test_runner.run_all_tests()
        
        # Print final summary
        test_runner.print_final_summary(summary)
        
        # Save detailed results
        results_file = f"phase_4_5_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        return summary
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Testing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Testing failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 