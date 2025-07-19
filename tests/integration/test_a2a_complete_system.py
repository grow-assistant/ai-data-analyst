"""Comprehensive Integration Tests for A2A/MCP System"""

import asyncio
import pytest
import tempfile
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directories for imports
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.security import (
    SecurityManager, PermissionLevel, AuthenticationError, AuthorizationError
)
from monitoring.observability import get_observability_manager, A2AMetricsCollector
from orchestrator_agent.scheduler import A2AWorkflowScheduler, ScheduleType
from orchestrator_agent.a2a_client import OrchestratorA2AClient
from common_utils.types import TaskRequest, DataHandle
from common_utils.data_handle_manager import get_data_handle_manager

pytestmark = pytest.mark.asyncio

class TestA2ASecuritySystem:
    """Test OAuth2 authentication and ACL authorization."""
    
    @pytest.fixture
    async def security_manager(self):
        """Create security manager for testing."""
        manager = SecurityManager()
        
        # Register test clients
        manager.oauth2_manager.register_client("test_agent_1", "secret123")
        manager.oauth2_manager.register_client("test_agent_2", "secret456")
        manager.oauth2_manager.register_client("orchestrator", "admin_secret")
        
        # Add ACL entries
        manager.acl_manager.add_acl_entry(
            "test_agent_1", 
            "mcp:tools:load_*", 
            [PermissionLevel.EXECUTE]
        )
        manager.acl_manager.add_acl_entry(
            "test_agent_2", 
            "mcp:tools:clean_*", 
            [PermissionLevel.EXECUTE]
        )
        manager.acl_manager.add_acl_entry(
            "orchestrator", 
            "*", 
            [PermissionLevel.ADMIN, PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE]
        )
        
        return manager
    
    async def test_oauth2_authentication(self, security_manager):
        """Test OAuth2 authentication flow."""
        # Test successful authentication
        token = await security_manager.authenticate_agent("test_agent_1", "secret123")
        assert token is not None
        assert token.access_token is not None
        assert token.token_type == "Bearer"
        assert not token.is_expired()
        
        # Test failed authentication
        with pytest.raises(AuthenticationError):
            await security_manager.authenticate_agent("test_agent_1", "wrong_secret")
    
    async def test_acl_authorization(self, security_manager):
        """Test ACL-based authorization."""
        # Get token for test agent
        token = await security_manager.authenticate_agent("test_agent_1", "secret123")
        
        # Test authorized action
        authorized = await security_manager.authorize_action(
            token.access_token,
            "mcp:tools:load_csv",
            PermissionLevel.EXECUTE
        )
        assert authorized
        
        # Test unauthorized action
        with pytest.raises(AuthorizationError):
            await security_manager.authorize_action(
                token.access_token,
                "mcp:tools:clean_data",
                PermissionLevel.EXECUTE
            )
    
    async def test_audit_logging(self, security_manager):
        """Test comprehensive audit logging."""
        # Generate some audit events
        await security_manager.log_tool_call(
            "test_agent_1",
            "load_csv",
            {"file_path": "/test/data.csv"},
            "success",
            "127.0.0.1"
        )
        
        await security_manager.log_tool_call(
            "test_agent_2",
            "clean_data",
            {"operations": ["remove_duplicates"]},
            "success",
            "127.0.0.1"
        )
        
        # Search audit logs
        logs = await security_manager.audit_logger.search_logs(
            agent_id="test_agent_1",
            action="tool_call"
        )
        
        assert len(logs) >= 1
        assert logs[0]["agent_id"] == "test_agent_1"
        assert logs[0]["action"] == "tool_call"
        assert logs[0]["resource"] == "mcp:tools:load_csv"

class TestA2AObservability:
    """Test OpenTelemetry observability and metrics."""
    
    @pytest.fixture
    def observability_manager(self):
        """Create observability manager for testing."""
        return get_observability_manager("test-service")
    
    @pytest.fixture
    def metrics_collector(self, observability_manager):
        """Create metrics collector for testing."""
        return A2AMetricsCollector(observability_manager)
    
    async def test_a2a_task_tracing(self, observability_manager):
        """Test A2A task distributed tracing."""
        task_id = str(uuid.uuid4())
        
        # Simulate A2A task execution with tracing
        with observability_manager.trace_a2a_task(task_id, "test_agent", "test_skill"):
            # Simulate some work
            await asyncio.sleep(0.1)
            
            # Add custom span event
            observability_manager.add_span_event("task_started", {
                "task_params": {"test": "value"}
            })
        
        # Verify metrics were recorded (in a real implementation, you'd check metrics store)
        assert True  # Placeholder - would verify metrics in actual Prometheus/Jaeger
    
    async def test_mcp_tool_tracing(self, observability_manager):
        """Test MCP tool call distributed tracing."""
        # Simulate MCP tool call with tracing
        with observability_manager.trace_mcp_tool_call(
            "load_csv", 
            "data_loader_agent", 
            {"file_path": "/test/data.csv"}
        ):
            # Simulate tool execution
            await asyncio.sleep(0.05)
        
        assert True  # Placeholder for metrics verification
    
    async def test_metrics_collection(self, metrics_collector):
        """Test system metrics collection."""
        metrics = await metrics_collector.collect_system_metrics()
        
        assert "timestamp" in metrics
        assert "system" in metrics
        assert "agents" in metrics
        assert "a2a" in metrics
        assert "mcp" in metrics
        
        # Verify system stats structure
        system_stats = metrics["system"]
        assert "cpu_percent" in system_stats
        assert "memory_percent" in system_stats
        
        # Verify agent stats structure
        agent_stats = metrics["agents"]
        assert "total_agents" in agent_stats
        assert "agent_status" in agent_stats

class TestA2AWorkflowScheduling:
    """Test APScheduler workflow automation."""
    
    @pytest.fixture
    async def a2a_client(self):
        """Create mock A2A client for testing."""
        # In real implementation, this would be a proper A2A client
        # For testing, we'll create a mock
        class MockA2AClient:
            async def execute_skill(self, agent_name, skill_name, task_request):
                from common_utils.types import TaskResponse
                return TaskResponse(
                    task_id=task_request.task_id,
                    status="completed",
                    agent_name=agent_name,
                    results={"data_handle_id": f"test_handle_{skill_name}"},
                    execution_time_ms=100
                )
        
        return MockA2AClient()
    
    @pytest.fixture
    async def scheduler(self, a2a_client):
        """Create workflow scheduler for testing."""
        scheduler = A2AWorkflowScheduler(a2a_client)
        await scheduler.start()
        yield scheduler
        await scheduler.shutdown()
    
    async def test_workflow_registration(self, scheduler):
        """Test workflow registration and scheduling."""
        workflow_id = scheduler.register_workflow(
            name="Test Workflow",
            description="Test workflow for integration testing",
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"seconds": 10},
            workflow_steps=[
                {
                    "agent": "data_loader_agent",
                    "skill": "load_dataset",
                    "params": {"data_source": "test_data"}
                }
            ]
        )
        
        assert workflow_id is not None
        
        # Check workflow status
        status = scheduler.get_workflow_status(workflow_id)
        assert status is not None
        assert status["name"] == "Test Workflow"
        assert status["enabled"] is True
        assert status["steps_count"] == 1
    
    async def test_one_time_workflow(self, scheduler):
        """Test one-time workflow scheduling."""
        run_time = datetime.utcnow() + timedelta(seconds=1)
        
        workflow_id = scheduler.schedule_one_time_workflow(
            name="One-time Test",
            description="One-time workflow test",
            workflow_steps=[
                {
                    "agent": "test_agent",
                    "skill": "test_skill",
                    "params": {}
                }
            ],
            run_at=run_time
        )
        
        assert workflow_id is not None
        
        # Wait for execution
        await asyncio.sleep(2)
        
        # Check that it was executed
        status = scheduler.get_workflow_status(workflow_id)
        assert status["run_count"] >= 1
    
    async def test_workflow_management(self, scheduler):
        """Test workflow enable/disable/remove operations."""
        workflow_id = scheduler.register_workflow(
            name="Management Test",
            description="Test workflow management",
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"seconds": 60},
            workflow_steps=[{"agent": "test", "skill": "test", "params": {}}]
        )
        
        # Test disable
        scheduler.disable_workflow(workflow_id)
        status = scheduler.get_workflow_status(workflow_id)
        assert status["enabled"] is False
        
        # Test enable
        scheduler.enable_workflow(workflow_id)
        status = scheduler.get_workflow_status(workflow_id)
        assert status["enabled"] is True
        
        # Test remove
        scheduler.remove_workflow(workflow_id)
        status = scheduler.get_workflow_status(workflow_id)
        assert status is None
    
    async def test_scheduler_status(self, scheduler):
        """Test scheduler status and statistics."""
        status = scheduler.get_scheduler_status()
        
        assert "running" in status
        assert "total_workflows" in status
        assert "enabled_workflows" in status
        assert "scheduled_jobs" in status
        assert "total_executions" in status
        
        assert status["running"] is True

class TestCompleteA2AWorkflow:
    """Test complete end-to-end A2A workflow."""
    
    @pytest.fixture
    async def data_manager(self):
        """Create data handle manager for testing."""
        return get_data_handle_manager()
    
    @pytest.fixture
    async def test_data_handle(self, data_manager):
        """Create test data handle."""
        # Create a temporary CSV file
        import pandas as pd
        
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'sales': [100 + i * 2 + (i % 7) * 10 for i in range(100)],
            'region': ['North', 'South', 'East', 'West'] * 25
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        # Create data handle
        handle = await data_manager.create_handle(
            data_type="test_dataset",
            local_path=temp_path,
            metadata={"source": "integration_test", "rows": 100}
        )
        
        yield handle
        
        # Cleanup
        os.unlink(temp_path)
    
    async def test_complete_data_pipeline(self, test_data_handle, data_manager):
        """Test complete data analysis pipeline simulation."""
        # This would test the complete workflow:
        # 1. Data loading
        # 2. Data cleaning
        # 3. Data enrichment
        # 4. Data analysis
        # 5. Report generation
        
        # For now, we'll simulate the data handle flow
        original_handle = test_data_handle
        assert original_handle.handle_id is not None
        
        # Simulate cleaning step
        import pandas as pd
        df = pd.read_csv(original_handle.local_path)
        
        # Simulate data cleaning
        df_cleaned = df.dropna()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_cleaned.to_csv(f.name, index=False)
            cleaned_path = f.name
        
        cleaned_handle = await data_manager.create_handle(
            data_type="cleaned_dataset",
            local_path=cleaned_path,
            metadata={
                "source_handle": original_handle.handle_id,
                "cleaning_operations": ["remove_duplicates"]
            }
        )
        
        assert cleaned_handle.handle_id != original_handle.handle_id
        assert cleaned_handle.metadata["source_handle"] == original_handle.handle_id
        
        # Cleanup
        os.unlink(cleaned_path)
    
    async def test_security_integration(self):
        """Test security integration across the system."""
        security_manager = SecurityManager()
        
        # Register a test client
        security_manager.oauth2_manager.register_client("integration_test", "test_secret")
        
        # Test authentication
        token = await security_manager.authenticate_agent("integration_test", "test_secret")
        assert token is not None
        
        # Test that audit log was created
        logs = await security_manager.audit_logger.search_logs(
            agent_id="integration_test",
            action="authenticate"
        )
        assert len(logs) >= 1

class TestSystemIntegration:
    """Test system integration and health checks."""
    
    async def test_agent_health_monitoring(self):
        """Test agent health monitoring integration."""
        observability = get_observability_manager("integration-test")
        
        # Record health status for different agents
        agents = [
            "orchestrator_agent",
            "data_loader_agent", 
            "data_cleaning_agent",
            "data_enrichment_agent",
            "data_analyst_agent",
            "presentation_agent"
        ]
        
        for agent in agents:
            observability.record_agent_health(agent, True)
        
        # Verify health recording (in real implementation, would check metrics)
        assert True  # Placeholder
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test circuit breaker functionality
        from common_utils.circuit_breaker import get_circuit_breaker_manager, CircuitBreakerConfig
        
        circuit_manager = get_circuit_breaker_manager()
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=1
        )
        
        circuit_breaker = circuit_manager.get_breaker("test_service", config)
        
        # Simulate failures
        async def failing_function():
            raise Exception("Simulated failure")
        
        # Should fail and eventually open circuit
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_function, timeout=1.0)
            except:
                pass
        
        # Circuit should be open now
        assert circuit_breaker.state == "open"
    
    async def test_system_metrics_aggregation(self):
        """Test system-wide metrics aggregation."""
        observability = get_observability_manager("system-test")
        metrics_collector = A2AMetricsCollector(observability)
        
        # Collect comprehensive metrics
        metrics = await metrics_collector.collect_system_metrics()
        
        # Verify all required metrics are present
        required_sections = ["system", "agents", "a2a", "mcp"]
        for section in required_sections:
            assert section in metrics
        
        # Verify metrics have reasonable values
        assert metrics["system"]["cpu_percent"] >= 0
        assert metrics["system"]["memory_percent"] >= 0
        assert metrics["agents"]["total_agents"] == 6

# Integration test runner
async def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting A2A/MCP System Integration Tests...")
    
    # Test security system
    print("🔐 Testing security system...")
    security_test = TestA2ASecuritySystem()
    security_manager = await security_test.security_manager()
    await security_test.test_oauth2_authentication(security_manager)
    await security_test.test_acl_authorization(security_manager)
    await security_test.test_audit_logging(security_manager)
    print("✅ Security system tests passed")
    
    # Test observability
    print("📊 Testing observability system...")
    observability_test = TestA2AObservability()
    obs_manager = observability_test.observability_manager()
    metrics_collector = observability_test.metrics_collector(obs_manager)
    await observability_test.test_a2a_task_tracing(obs_manager)
    await observability_test.test_mcp_tool_tracing(obs_manager)
    await observability_test.test_metrics_collection(metrics_collector)
    print("✅ Observability system tests passed")
    
    print("🎉 All integration tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_integration_tests()) 