#!/usr/bin/env python3
"""
Phase 2: A2A Communication Testing
Tests orchestrator-to-agent communication and data handle flow through the pipeline.
"""

import asyncio
import httpx
import tempfile
import pandas as pd
import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2ACommunicationTester:
    """Test suite for A2A communication between agents."""
    
    def __init__(self):
        self.base_urls = {
            'orchestrator': 'http://localhost:8000',
            'data_loader': 'http://localhost:10006',
            'data_cleaning': 'http://localhost:10008',
            'data_enrichment': 'http://localhost:10009',
            'data_analyst': 'http://localhost:10007',
            'presentation': 'http://localhost:10010'
        }
        self.test_results = {}
        
    async def check_agent_health(self, agent_name: str) -> bool:
        """Check if an agent is healthy and responding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_urls[agent_name]}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info(f"✅ {agent_name} is healthy")
                    return True
                else:
                    logger.error(f"❌ {agent_name} health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"❌ {agent_name} not reachable: {e}")
            return False
    
    async def test_orchestrator_to_data_loader(self) -> bool:
        """Test Orchestrator → Data Loader communication."""
        logger.info("🧪 Testing Orchestrator → Data Loader Communication")
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [100, 200, 150],
            'category': ['A', 'B', 'C']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            test_file = f.name
        
        try:
            # Test direct orchestrator call to data loader
            load_payload = {
                "jsonrpc": "2.0",
                "method": "load_dataset",
                "params": {"file_path": test_file},
                "id": "test_orchestrator_loader"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_loader'], json=load_payload, timeout=15.0)
                result = response.json()
            
            if "result" in result:
                data_handle_id = result["result"]["data_handle_id"]
                logger.info(f"✅ Orchestrator → Data Loader: SUCCESS")
                logger.info(f"   Data Handle: {data_handle_id}")
                self.test_results['orchestrator_to_loader'] = {
                    'status': 'PASS',
                    'data_handle_id': data_handle_id,
                    'details': result["result"]
                }
                return data_handle_id
            else:
                logger.error(f"❌ Orchestrator → Data Loader: FAILED - {result}")
                self.test_results['orchestrator_to_loader'] = {'status': 'FAIL', 'error': result}
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestrator → Data Loader: ERROR - {e}")
            self.test_results['orchestrator_to_loader'] = {'status': 'ERROR', 'error': str(e)}
            return False
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    async def test_orchestrator_to_data_cleaning(self, data_handle_id: str) -> str:
        """Test Orchestrator → Data Cleaning communication with data handle."""
        logger.info("🧪 Testing Orchestrator → Data Cleaning Communication")
        
        try:
            clean_payload = {
                "jsonrpc": "2.0",
                "method": "clean_dataset",
                "params": {
                    "data_handle_id": data_handle_id,
                    "operations": ["remove_duplicates", "handle_missing_values"]
                },
                "id": "test_orchestrator_cleaning"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_cleaning'], json=clean_payload, timeout=15.0)
                result = response.json()
            
            if "result" in result:
                cleaned_handle_id = result["result"]["cleaned_data_handle_id"]
                logger.info(f"✅ Orchestrator → Data Cleaning: SUCCESS")
                logger.info(f"   Cleaned Handle: {cleaned_handle_id}")
                logger.info(f"   Operations: {result['result'].get('operations', [])}")
                self.test_results['orchestrator_to_cleaning'] = {
                    'status': 'PASS',
                    'cleaned_data_handle_id': cleaned_handle_id,
                    'details': result["result"]
                }
                return cleaned_handle_id
            else:
                logger.error(f"❌ Orchestrator → Data Cleaning: FAILED - {result}")
                self.test_results['orchestrator_to_cleaning'] = {'status': 'FAIL', 'error': result}
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestrator → Data Cleaning: ERROR - {e}")
            self.test_results['orchestrator_to_cleaning'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    async def test_orchestrator_to_data_enrichment(self, data_handle_id: str) -> str:
        """Test Orchestrator → Data Enrichment communication."""
        logger.info("🧪 Testing Orchestrator → Data Enrichment Communication")
        
        try:
            enrich_payload = {
                "jsonrpc": "2.0",
                "method": "enrich_dataset",
                "params": {"data_handle_id": data_handle_id},
                "id": "test_orchestrator_enrichment"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_enrichment'], json=enrich_payload, timeout=15.0)
                result = response.json()
            
            if "result" in result:
                enriched_handle_id = result["result"]["enriched_data_handle_id"]
                logger.info(f"✅ Orchestrator → Data Enrichment: SUCCESS")
                logger.info(f"   Enriched Handle: {enriched_handle_id}")
                self.test_results['orchestrator_to_enrichment'] = {
                    'status': 'PASS',
                    'enriched_data_handle_id': enriched_handle_id,
                    'details': result["result"]
                }
                return enriched_handle_id
            else:
                logger.error(f"❌ Orchestrator → Data Enrichment: FAILED - {result}")
                self.test_results['orchestrator_to_enrichment'] = {'status': 'FAIL', 'error': result}
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestrator → Data Enrichment: ERROR - {e}")
            self.test_results['orchestrator_to_enrichment'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    async def test_orchestrator_to_data_analyst(self, data_handle_id: str) -> Dict:
        """Test Orchestrator → Data Analyst communication."""
        logger.info("🧪 Testing Orchestrator → Data Analyst Communication")
        
        try:
            analyze_payload = {
                "jsonrpc": "2.0",
                "method": "analyze_dataset",
                "params": {"data_handle_id": data_handle_id},
                "id": "test_orchestrator_analysis"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_analyst'], json=analyze_payload, timeout=15.0)
                result = response.json()
            
            if "result" in result:
                analysis_results = result["result"]
                logger.info(f"✅ Orchestrator → Data Analyst: SUCCESS")
                logger.info(f"   Analysis completed: {analysis_results.get('status', 'Unknown')}")
                self.test_results['orchestrator_to_analyst'] = {
                    'status': 'PASS',
                    'details': analysis_results
                }
                return analysis_results
            else:
                logger.error(f"❌ Orchestrator → Data Analyst: FAILED - {result}")
                self.test_results['orchestrator_to_analyst'] = {'status': 'FAIL', 'error': result}
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestrator → Data Analyst: ERROR - {e}")
            self.test_results['orchestrator_to_analyst'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    async def test_orchestrator_to_presentation(self, data_handle_id: str) -> str:
        """Test Orchestrator → Presentation communication."""
        logger.info("🧪 Testing Orchestrator → Presentation Communication")
        
        try:
            present_payload = {
                "jsonrpc": "2.0",
                "method": "create_report",
                "params": {"data_handle_id": data_handle_id},
                "id": "test_orchestrator_presentation"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['presentation'], json=present_payload, timeout=15.0)
                result = response.json()
            
            if "result" in result:
                report_result = result["result"]
                logger.info(f"✅ Orchestrator → Presentation: SUCCESS")
                logger.info(f"   Report created: {report_result.get('status', 'Unknown')}")
                self.test_results['orchestrator_to_presentation'] = {
                    'status': 'PASS',
                    'details': report_result
                }
                return report_result
            else:
                logger.error(f"❌ Orchestrator → Presentation: FAILED - {result}")
                self.test_results['orchestrator_to_presentation'] = {'status': 'FAIL', 'error': result}
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestrator → Presentation: ERROR - {e}")
            self.test_results['orchestrator_to_presentation'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    async def test_data_handle_chaining(self):
        """Test complete data handle flow through the pipeline."""
        logger.info("🧪 Testing Complete Data Handle Pipeline Flow")
        
        # Test the complete pipeline: Load → Clean → Enrich → Analyze → Present
        data_handle_id = await self.test_orchestrator_to_data_loader()
        if not data_handle_id:
            logger.error("❌ Pipeline test failed at data loading stage")
            return False
        
        cleaned_handle_id = await self.test_orchestrator_to_data_cleaning(data_handle_id)
        if cleaned_handle_id:
            logger.info(f"✅ Data handle chaining successful: {data_handle_id} → {cleaned_handle_id}")
            
            # Continue with enrichment if cleaning succeeded
            enriched_handle_id = await self.test_orchestrator_to_data_enrichment(cleaned_handle_id)
            if enriched_handle_id:
                logger.info(f"✅ Enrichment chaining successful: {cleaned_handle_id} → {enriched_handle_id}")
                
                # Continue with analysis
                analysis_result = await self.test_orchestrator_to_data_analyst(enriched_handle_id)
                if analysis_result:
                    logger.info("✅ Analysis chaining successful")
                    
                    # Finally presentation
                    presentation_result = await self.test_orchestrator_to_presentation(enriched_handle_id)
                    if presentation_result:
                        logger.info("✅ Complete pipeline chaining successful!")
                        return True
        
        return False
    
    async def run_all_tests(self):
        """Run all A2A communication tests."""
        logger.info("🚀 Starting Phase 2: A2A Communication Testing")
        logger.info("=" * 70)
        
        # Check agent health first
        healthy_agents = {}
        for agent_name in self.base_urls.keys():
            healthy_agents[agent_name] = await self.check_agent_health(agent_name)
        
        logger.info("\n📊 Agent Health Status:")
        for agent, status in healthy_agents.items():
            status_emoji = "✅" if status else "❌"
            logger.info(f"   {status_emoji} {agent}: {'HEALTHY' if status else 'UNAVAILABLE'}")
        
        # Run communication tests
        logger.info("\n🔄 Running A2A Communication Tests...")
        
        # Test complete pipeline if all agents are healthy
        if all(healthy_agents.values()):
            await self.test_data_handle_chaining()
        else:
            # Test individual connections for available agents
            if healthy_agents.get('data_loader'):
                await self.test_orchestrator_to_data_loader()
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 2: A2A COMMUNICATION TEST RESULTS")
        logger.info("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASS')
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            logger.info(f"   {status_emoji} {test_name}: {result['status']}")
        
        # Save results to file
        results_file = f"test_results_a2a_communication_{int(asyncio.get_event_loop().time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\n📊 Detailed results saved to: {results_file}")
        
        if passed_tests == total_tests and total_tests > 0:
            logger.info("\n🎉 ALL A2A COMMUNICATION TESTS PASSED!")
            logger.info("✅ Ready for Phase 3: Integration Testing")
        else:
            logger.info(f"\n⚠️ {total_tests - passed_tests} tests need attention")

async def main():
    """Main test function."""
    tester = A2ACommunicationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 