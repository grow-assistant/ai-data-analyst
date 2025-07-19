"""
Integration tests for RootCause Analyst Agent (Why-Bot)
Tests all agent skills and A2A communication patterns.
"""

import asyncio
import httpx
import json
import pytest
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRootCauseAnalyst:
    """Test suite for RootCause Analyst Agent."""
    
    BASE_URL = "http://localhost:10011"
    
    async def setup(self):
        """Setup test environment."""
        # Verify agent is running
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.BASE_URL}/health")
                assert response.status_code == 200
                logger.info("✅ RootCause Analyst agent is healthy")
            except Exception as e:
                raise Exception(f"RootCause Analyst agent not available: {e}")
    
    async def test_health_endpoint(self):
        """Test agent health endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["agent"] == "rootcause_analyst"
            assert "version" in data
            
            logger.info("✅ Health endpoint working correctly")
    
    async def test_capabilities_endpoint(self):
        """Test agent capabilities endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/capabilities")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["agent_type"] == "rootcause_analyst"
            assert "skills" in data
            
            # Verify all expected skills are present
            skill_names = [skill["name"] for skill in data["skills"]]
            expected_skills = [
                "investigate_trend",
                "analyze_causal_factors", 
                "generate_hypotheses",
                "explain_variance",
                "create_causal_brief"
            ]
            
            for skill in expected_skills:
                assert skill in skill_names, f"Missing skill: {skill}"
            
            logger.info(f"✅ All {len(expected_skills)} skills available")
    
    async def test_generate_hypotheses_skill(self):
        """Test hypothesis generation skill."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "generate_hypotheses",
                "params": {
                    "data_handle_id": "test_handle_123",
                    "trend_description": "Revenue decreased by 15% in Q2 2024",
                    "business_context": "E-commerce subscription business"
                },
                "id": "test_hypotheses_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == "test_hypotheses_1"
            assert "result" in data
            
            result = data["result"]
            assert result["status"] == "completed"
            assert "hypotheses" in result
            assert len(result["hypotheses"]) >= 3  # Should generate multiple hypotheses
            
            # Verify hypothesis structure
            for hypothesis in result["hypotheses"]:
                assert "hypothesis_id" in hypothesis
                assert "hypothesis" in hypothesis
                assert "generated_by" in hypothesis
            
            logger.info(f"✅ Generated {len(result['hypotheses'])} hypotheses successfully")
    
    async def test_explain_variance_skill(self):
        """Test variance explanation skill."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "explain_variance",
                "params": {
                    "data_handle_id": "test_handle_123",
                    "target_metric": "revenue"
                },
                "id": "test_variance_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["jsonrpc"] == "2.0"
            assert "result" in data
            
            result = data["result"]
            assert result["status"] == "completed"
            assert "variance_analysis" in result
            
            variance_analysis = result["variance_analysis"]
            assert "target_metric" in variance_analysis
            assert variance_analysis["target_metric"] == "revenue"
            assert "sample_size" in variance_analysis
            assert "correlations" in variance_analysis
            
            logger.info("✅ Variance explanation completed successfully")
    
    async def test_investigate_trend_skill(self):
        """Test main trend investigation skill."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "investigate_trend",
                "params": {
                    "trend_id": "revenue_drop_test",
                    "dataset_id": "test_data",
                    "data_handle_id": "test_handle_123",
                    "affected_metric": "revenue",
                    "direction": "down",
                    "magnitude_pct": 15.2,
                    "time_window": ["2024-04-01", "2024-06-30"],
                    "business_context": "Q2 revenue analysis"
                },
                "id": "test_investigation_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["jsonrpc"] == "2.0"
            assert "result" in data
            
            result = data["result"]
            assert result["status"] == "completed"
            assert result["trend_id"] == "revenue_drop_test"
            
            # Verify comprehensive analysis structure
            assert "investigation_id" in result
            assert "execution_time" in result
            assert "trend_summary" in result
            assert "hypotheses" in result
            assert "variance_analysis" in result
            assert "segment_analysis" in result
            assert "causal_analysis" in result
            assert "confidence_score" in result
            assert "recommendations" in result
            
            # Verify confidence score is valid
            confidence = result["confidence_score"]
            assert 0.0 <= confidence <= 1.0
            
            # Verify trend summary
            trend_summary = result["trend_summary"]
            assert trend_summary["metric"] == "revenue"
            assert trend_summary["direction"] == "down"
            assert trend_summary["magnitude_pct"] == 15.2
            
            logger.info(f"✅ Trend investigation completed with confidence: {confidence:.2f}")
    
    async def test_analyze_causal_factors_skill(self):
        """Test causal factors analysis skill."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "analyze_causal_factors",
                "params": {
                    "data_handle_id": "test_handle_123",
                    "target_variable": "revenue",
                    "treatment_variables": ["region", "customer_segment"]
                },
                "id": "test_causal_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            result = data["result"]
            assert result["status"] == "completed"
            assert "causal_analysis" in result
            
            logger.info("✅ Causal factors analysis completed")
    
    async def test_create_causal_brief_skill(self):
        """Test causal brief generation skill."""
        async with httpx.AsyncClient() as client:
            # First run an investigation to get results
            investigation_request = {
                "jsonrpc": "2.0",
                "method": "investigate_trend",
                "params": {
                    "trend_id": "brief_test",
                    "dataset_id": "test_data",
                    "data_handle_id": "test_handle_123",
                    "affected_metric": "revenue",
                    "direction": "down",
                    "magnitude_pct": 10.5,
                    "time_window": ["2024-01-01", "2024-03-31"]
                },
                "id": "test_brief_investigation"
            }
            
            investigation_response = await client.post(f"{self.BASE_URL}/", json=investigation_request)
            investigation_result = investigation_response.json()["result"]
            
            # Now create a brief from the results
            brief_request = {
                "jsonrpc": "2.0",
                "method": "create_causal_brief",
                "params": {
                    "analysis_results": investigation_result,
                    "format": "markdown"
                },
                "id": "test_brief_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=brief_request)
            
            assert response.status_code == 200
            data = response.json()
            
            result = data["result"]
            assert result["status"] == "completed"
            assert "brief_content" in result
            assert result["format"] == "markdown"
            
            # Verify brief content structure
            brief_content = result["brief_content"]
            assert "Root Cause Analysis Brief" in brief_content
            assert "Executive Summary" in brief_content
            assert "Key Findings" in brief_content
            assert "Recommendations" in brief_content
            
            logger.info("✅ Causal brief generated successfully")
    
    async def test_error_handling(self):
        """Test error handling for invalid requests."""
        async with httpx.AsyncClient() as client:
            # Test invalid method
            request_data = {
                "jsonrpc": "2.0",
                "method": "invalid_method",
                "params": {},
                "id": "test_error_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "error" in data
            assert data["error"]["code"] == -32601  # Method not found
            
            logger.info("✅ Error handling working correctly")
    
    async def test_missing_parameters(self):
        """Test handling of missing required parameters."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "investigate_trend",
                "params": {
                    # Missing required parameters
                    "trend_id": "incomplete_test"
                },
                "id": "test_missing_params"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should either have error or handle gracefully
            if "error" in data:
                logger.info("✅ Missing parameters handled with error response")
            else:
                # Agent handled missing params gracefully
                logger.info("✅ Missing parameters handled gracefully")
    
    async def test_confidence_and_escalation(self):
        """Test confidence scoring and escalation logic."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "jsonrpc": "2.0",
                "method": "investigate_trend",
                "params": {
                    "trend_id": "confidence_test",
                    "dataset_id": "small_data",
                    "data_handle_id": "test_handle_123",
                    "affected_metric": "revenue",
                    "direction": "up",
                    "magnitude_pct": 5.0,  # Small change
                    "time_window": ["2024-01-01", "2024-01-07"]  # Short window
                },
                "id": "test_confidence_1"
            }
            
            response = await client.post(f"{self.BASE_URL}/", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            result = data["result"]
            assert "confidence_score" in result
            
            confidence = result["confidence_score"]
            assert 0.0 <= confidence <= 1.0
            
            # Check if escalation logic triggered for low confidence
            if confidence < 0.7:
                logger.info(f"✅ Low confidence ({confidence:.2f}) detected - escalation may be recommended")
            else:
                logger.info(f"✅ Good confidence ({confidence:.2f}) - no escalation needed")

# Async test runner
async def run_all_tests():
    """Run all RootCause Analyst tests."""
    logger.info("🔍 Starting RootCause Analyst Agent Tests")
    
    test_instance = TestRootCauseAnalyst()
    
    # Setup
    await test_instance.setup()
    
    tests = [
        ("Health Endpoint", test_instance.test_health_endpoint),
        ("Capabilities Endpoint", test_instance.test_capabilities_endpoint),
        ("Generate Hypotheses Skill", test_instance.test_generate_hypotheses_skill),
        ("Explain Variance Skill", test_instance.test_explain_variance_skill),
        ("Investigate Trend Skill", test_instance.test_investigate_trend_skill),
        ("Analyze Causal Factors Skill", test_instance.test_analyze_causal_factors_skill),
        ("Create Causal Brief Skill", test_instance.test_create_causal_brief_skill),
        ("Error Handling", test_instance.test_error_handling),
        ("Missing Parameters", test_instance.test_missing_parameters),
        ("Confidence and Escalation", test_instance.test_confidence_and_escalation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"🧪 Running: {test_name}")
            await test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            failed += 1
    
    logger.info(f"\n📊 RootCause Analyst Test Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    # Run tests directly
    asyncio.run(run_all_tests()) 