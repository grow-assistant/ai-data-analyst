#!/usr/bin/env python3
"""
Individual Agent Testing Script for A2A Multi-Agent Framework

This script tests each agent individually to ensure they can:
1. Start as A2A servers
2. Register their capabilities correctly 
3. Execute their core skills
4. Handle data properly
"""

import asyncio
import sys
import os
import logging
import tempfile
import pandas as pd
from pathlib import Path
import json
import time
import httpx
from typing import Dict, Any

# Add parent directory for common_utils access
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Correctly import from the a2a-sdk
from a2a.client.client import A2AClient
from a2a.types import SendMessageRequest, Message, TextPart, MessageSendParams

# Import common utilities
from common_utils.types import DataHandle
from common_utils.data_handle_manager import get_data_handle_manager

logger = logging.getLogger(__name__)

class AgentTester:
    """Test runner for individual agents."""
    
    def __init__(self):
        self.httpx_client = httpx.AsyncClient()
        # We'll create the A2AClient when needed with specific URLs
        self.data_manager = get_data_handle_manager()
        self.test_results = {}
        
        # Agent endpoints
        self.agents = {
            "data_loader_agent": "http://localhost:10006",
            "data_cleaning_agent": "http://localhost:10008", 
            "data_enrichment_agent": "http://localhost:10009",
            "data_analyst_agent": "http://localhost:10007",
            "presentation_agent": "http://localhost:10010",
            "orchestrator_agent": "http://localhost:8000"
        }
        
    async def test_agent_health(self, agent_name: str, url: str) -> bool:
        """Test if an agent is running and healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    health_data = response.json()
                    logger.info(f"✅ {agent_name} is healthy: {health_data}")
                    return True
                else:
                    logger.error(f"❌ {agent_name} health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ {agent_name} not reachable: {e}")
            return False
    
    async def test_agent_capabilities(self, agent_name: str, url: str) -> Dict[str, Any]:
        """Test agent capabilities and skills."""
        try:
            async with httpx.AsyncClient() as client:
                # Try to get agent card/capabilities
                response = await client.get(f"{url}/capabilities", timeout=5.0)
                if response.status_code == 200:
                    capabilities = response.json()
                    logger.info(f"✅ {agent_name} capabilities: {capabilities.get('skills', [])}")
                    return capabilities
                else:
                    logger.warning(f"⚠️ {agent_name} capabilities endpoint not available")
                    return {}
        except Exception as e:
            logger.warning(f"⚠️ Could not get {agent_name} capabilities: {e}")
            return {}
    
    def create_test_data(self) -> str:
        """Create sample test data for testing."""
        # Create a temporary CSV file with sample data
        test_data = {
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'revenue': [1000 + i*10 + (i%7)*50 for i in range(100)],
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Product_A', 'Product_B'] * 50,
            'sales_rep': [f'Rep_{i%10}' for i in range(100)]
        }
        
        df = pd.DataFrame(test_data)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        logger.info(f"📊 Created test data file: {temp_file.name}")
        return temp_file.name
    
    async def test_data_loader_agent(self):
        """Test the Data Loader Agent."""
        print("🔄 Testing Data Loader Agent...")
        url = "http://localhost:10006"
        
        results = {
            "agent": "Data Loader Agent",
            "tests": [],
            "data_handle": None
        }
        
        try:
            # Test health check
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{url}/health", timeout=10.0)
                health_success = health_response.status_code == 200
                results["tests"].append({"test": "health_check", "passed": health_success})
                
                if not health_success:
                    logger.error(f"❌ Health check failed: {health_response.status_code}")
                    return results
                
                logger.info("✅ Health check passed")
            
            # Test skill execution
            test_file = self.create_test_data()
            
            request_payload = {
                "jsonrpc": "2.0",
                "method": "load_dataset",
                "params": {"file_path": test_file},
                "id": "1",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=request_payload, timeout=30.0)
            
            response.raise_for_status()
            response_data = response.json()
            
            load_success = "result" in response_data and response_data["result"]["status"] == "completed"
            results["tests"].append({"test": "load_dataset_skill", "passed": load_success})
            
            if load_success:
                data_handle = response_data["result"]
                results["data_handle"] = data_handle
                logger.info(f"✅ Data loaded successfully, handle: {data_handle['data_handle_id']}")
            else:
                error_message = response_data.get("error", "Unknown error")
                logger.error(f"❌ Data loading failed: {error_message}")
                
            # Cleanup
            if test_file and os.path.exists(test_file):
                os.remove(test_file)
                
        except Exception as e:
            logger.exception(f"❌ Error testing data loader: {e}")
            results["tests"].append({
                "test": "load_dataset_skill", 
                "passed": False,
                "error": str(e)
            })
        
        return results
    
    async def test_data_cleaning_agent(self, input_data_handle: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test the data cleaning agent."""
        agent_name = "data_cleaning_agent"
        url = self.agents[agent_name]
        results = {"agent": agent_name, "tests": []}
        
        logger.info(f"🧪 Testing {agent_name}")
        
        # Test 1: Health check
        health_ok = await self.test_agent_health(agent_name, url)
        results["tests"].append({"test": "health_check", "passed": health_ok})
        
        if not health_ok:
            return results
        
        # Test 2: Capabilities
        capabilities = await self.test_agent_capabilities(agent_name, url)
        results["tests"].append({"test": "capabilities", "passed": len(capabilities) > 0})
        
        # Test 3: Clean dataset skill
        if input_data_handle:
            try:
                request_payload = {
                    "jsonrpc": "2.0",
                    "method": "clean_dataset",
                    "params": {"data_handle_id": input_data_handle.get("data_handle_id")},
                    "id": "1",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=request_payload, timeout=30.0)
                
                response.raise_for_status()
                response_data = response.json()
                
                clean_success = "result" in response_data and response_data["result"]["status"] == "completed"
                results["tests"].append({"test": "clean_dataset_skill", "passed": clean_success})
                
                if clean_success:
                    cleaned_handle = response_data["result"]
                    results["cleaned_data_handle"] = cleaned_handle
                    logger.info(f"✅ Data cleaned successfully")
                else:
                    error_message = response_data.get("error", "Unknown error")
                    logger.error(f"❌ Data cleaning failed: {error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error testing data cleaning: {e}")
                results["tests"].append({"test": "clean_dataset_skill", "passed": False, "error": str(e)})
        else:
            results["tests"].append({"test": "clean_dataset_skill", "passed": False, "error": "No input data handle"})
        
        return results
    
    async def test_data_enrichment_agent(self, input_data_handle: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test the data enrichment agent.""" 
        agent_name = "data_enrichment_agent"
        url = self.agents[agent_name]
        results = {"agent": agent_name, "tests": []}
        
        logger.info(f"🧪 Testing {agent_name}")
        
        # Test 1: Health check
        health_ok = await self.test_agent_health(agent_name, url)
        results["tests"].append({"test": "health_check", "passed": health_ok})
        
        if not health_ok:
            return results
        
        # Test 2: Capabilities
        capabilities = await self.test_agent_capabilities(agent_name, url)
        results["tests"].append({"test": "capabilities", "passed": len(capabilities) > 0})
        
        # Test 3: Enrich dataset skill
        if input_data_handle:
            try:
                request_payload = {
                    "jsonrpc": "2.0",
                    "method": "enrich_dataset",
                    "params": {"data_handle_id": input_data_handle.get("data_handle_id")},
                    "id": "1",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=request_payload, timeout=30.0)
                
                response.raise_for_status()
                response_data = response.json()
                
                enrich_success = "result" in response_data and response_data["result"]["status"] == "completed"
                results["tests"].append({"test": "enrich_dataset_skill", "passed": enrich_success})
                
                if enrich_success:
                    enriched_handle = response_data["result"]
                    results["enriched_data_handle"] = enriched_handle
                    logger.info(f"✅ Data enriched successfully")
                else:
                    error_message = response_data.get("error", "Unknown error")
                    logger.error(f"❌ Data enrichment failed: {error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error testing data enrichment: {e}")
                results["tests"].append({"test": "enrich_dataset_skill", "passed": False, "error": str(e)})
        else:
            results["tests"].append({"test": "enrich_dataset_skill", "passed": False, "error": "No input data handle"})
        
        return results
    
    async def test_data_analyst_agent(self, input_data_handle: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test the data analyst agent."""
        agent_name = "data_analyst_agent"
        url = self.agents[agent_name]
        results = {"agent": agent_name, "tests": []}
        
        logger.info(f"🧪 Testing {agent_name}")
        
        # Test 1: Health check
        health_ok = await self.test_agent_health(agent_name, url)
        results["tests"].append({"test": "health_check", "passed": health_ok})
        
        if not health_ok:
            return results
        
        # Test 2: Capabilities
        capabilities = await self.test_agent_capabilities(agent_name, url)
        results["tests"].append({"test": "capabilities", "passed": len(capabilities) > 0})
        
        # Test 3: Analyze dataset skill
        if input_data_handle:
            try:
                request_payload = {
                    "jsonrpc": "2.0",
                    "method": "analyze_dataset",
                    "params": {"data_handle_id": input_data_handle.get("data_handle_id"), "analysis_type": "summary"},
                    "id": "1",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=request_payload, timeout=60.0)
                
                response.raise_for_status()
                response_data = response.json()
                
                analyze_success = "result" in response_data and response_data["result"]["status"] == "completed"
                results["tests"].append({"test": "analyze_dataset_skill", "passed": analyze_success})
                
                if analyze_success:
                    analysis_handle = response_data["result"]
                    results["analysis_data_handle"] = analysis_handle
                    logger.info(f"✅ Data analyzed successfully")
                else:
                    error_message = response_data.get("error", "Unknown error")
                    logger.error(f"❌ Data analysis failed: {error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error testing data analysis: {e}")
                results["tests"].append({"test": "analyze_dataset_skill", "passed": False, "error": str(e)})
        else:
            results["tests"].append({"test": "analyze_dataset_skill", "passed": False, "error": "No input data handle"})
        
        return results
    
    async def test_presentation_agent(self, input_data_handle: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test the presentation agent."""
        agent_name = "presentation_agent"
        url = self.agents[agent_name]
        results = {"agent": agent_name, "tests": []}
        
        logger.info(f"🧪 Testing {agent_name}")
        
        # Test 1: Health check
        health_ok = await self.test_agent_health(agent_name, url)
        results["tests"].append({"test": "health_check", "passed": health_ok})
        
        if not health_ok:
            return results
        
        # Test 2: Capabilities
        capabilities = await self.test_agent_capabilities(agent_name, url)
        results["tests"].append({"test": "capabilities", "passed": len(capabilities) > 0})
        
        # Test 3: Create report skill
        if input_data_handle:
            try:
                request_payload = {
                    "jsonrpc": "2.0",
                    "method": "create_report",
                    "params": {
                        "data_handle_id": input_data_handle.get("data_handle_id"),
                        "report_type": "summary"
                    },
                    "id": "1",
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=request_payload, timeout=60.0)
                
                response.raise_for_status()
                response_data = response.json()
                
                report_success = "result" in response_data and response_data["result"]["status"] == "completed"
                results["tests"].append({"test": "create_report_skill", "passed": report_success})
                
                if report_success:
                    report_handle = response_data["result"]
                    results["report_handle"] = report_handle
                    logger.info(f"✅ Report created successfully")
                else:
                    error_message = response_data.get("error", "Unknown error")
                    logger.error(f"❌ Report creation failed: {error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error testing presentation: {e}")
                results["tests"].append({"test": "create_report_skill", "passed": False, "error": str(e)})
        else:
            results["tests"].append({"test": "create_report_skill", "passed": False, "error": "No input data handle"})
        
        return results
    
    async def test_orchestrator_agent(self) -> Dict[str, Any]:
        """Test the orchestrator agent."""
        agent_name = "orchestrator_agent"
        url = self.agents[agent_name]
        results = {"agent": agent_name, "tests": []}
        
        logger.info(f"🧪 Testing {agent_name}")
        
        # Test 1: Health check
        health_ok = await self.test_agent_health(agent_name, url)
        results["tests"].append({"test": "health_check", "passed": health_ok})
        
        if not health_ok:
            return results
        
        # Test 2: API endpoints (orchestrator has REST API)
        try:
            async with httpx.AsyncClient() as client:
                # Test API endpoints
                response = await client.get(f"{url}/health", timeout=5.0)
                api_ok = response.status_code == 200
                results["tests"].append({"test": "api_endpoints", "passed": api_ok})
                
        except Exception as e:
            logger.error(f"❌ Error testing orchestrator API: {e}")
            results["tests"].append({"test": "api_endpoints", "passed": False, "error": str(e)})
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all individual agent tests."""
        logger.info("🚀 Starting individual agent tests...")
        
        all_results = {
            "test_suite": "individual_agents",
            "timestamp": time.time(),
            "results": []
        }
        
        # Test orchestrator first (it's different - REST API not A2A server)
        orchestrator_results = await self.test_orchestrator_agent()
        all_results["results"].append(orchestrator_results)
        
        # Test data loader and get data handle for downstream tests
        loader_results = await self.test_data_loader_agent()
        all_results["results"].append(loader_results)
        
        data_handle = loader_results.get("data_handle")
        
        # Test data cleaning with output from loader
        cleaning_results = await self.test_data_cleaning_agent(data_handle)
        all_results["results"].append(cleaning_results)
        
        cleaned_handle = cleaning_results.get("cleaned_data_handle", data_handle)
        
        # Test data enrichment with cleaned data
        enrichment_results = await self.test_data_enrichment_agent(cleaned_handle)
        all_results["results"].append(enrichment_results)
        
        enriched_handle = enrichment_results.get("enriched_data_handle", cleaned_handle)
        
        # Test data analysis with enriched data
        analysis_results = await self.test_data_analyst_agent(enriched_handle)
        all_results["results"].append(analysis_results)
        
        analysis_handle = analysis_results.get("analysis_data_handle", enriched_handle)
        
        # Test presentation with analysis results
        presentation_results = await self.test_presentation_agent(analysis_handle)
        all_results["results"].append(presentation_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results."""
        print("\n" + "="*60)
        print("🧪 INDIVIDUAL AGENT TEST SUMMARY")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for agent_result in results["results"]:
            agent_name = agent_result["agent"]
            tests = agent_result.get("tests", [])
            
            agent_passed = sum(1 for test in tests if test["passed"])
            agent_total = len(tests)
            
            total_tests += agent_total
            passed_tests += agent_passed
            
            status = "✅ PASS" if agent_passed == agent_total else "❌ FAIL"
            print(f"\n{agent_name}: {status} ({agent_passed}/{agent_total})")
            
            for test in tests:
                test_status = "✅" if test["passed"] else "❌"
                test_name = test["test"]
                print(f"  {test_status} {test_name}")
                if not test["passed"] and "error" in test:
                    print(f"     Error: {test['error']}")
        
        print(f"\n{'='*60}")
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed")
        
        print("="*60)

async def main():
    """Main test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Individual Agent Testing Suite")
    print("This script tests each agent individually before integration testing.")
    print("\nMake sure all agents are running on their designated ports:")
    print("- Data Loader Agent: http://localhost:10006")
    print("- Data Cleaning Agent: http://localhost:10008") 
    print("- Data Enrichment Agent: http://localhost:10009")
    print("- Data Analysis Agent: http://localhost:10007")
    print("- Presentation Agent: http://localhost:10010")
    print("- Orchestrator Agent: http://localhost:8000")
    print("\nStarting tests...")
    
    tester = AgentTester()
    results = await tester.run_all_tests()
    
    # Print summary
    tester.print_summary(results)
    
    # Save results to file
    results_file = f"test_results_individual_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 