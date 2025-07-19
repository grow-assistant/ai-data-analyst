"""OpenTelemetry Observability Module for A2A/MCP System"""

import logging
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

logger = logging.getLogger(__name__)

class A2AObservabilityManager:
    """Centralized observability manager for A2A/MCP system."""
    
    def __init__(self, service_name: str = "a2a-mcp-system"):
        self.service_name = service_name
        self.resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
            ResourceAttributes.SERVICE_NAMESPACE: "data-analysis-agents"
        })
        
        # Initialize tracing
        self._setup_tracing()
        
        # Initialize metrics
        self._setup_metrics()
        
        # Instrument HTTP clients
        self._instrument_http()
        
        logger.info(f"Observability initialized for service: {service_name}")
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing with Jaeger exporter."""
        # Configure tracer provider
        trace.set_tracer_provider(TracerProvider(resource=self.resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set global propagator for distributed tracing
        set_global_textmap(B3MultiFormat())
        
        self.tracer = trace.get_tracer(__name__)
        logger.info("Distributed tracing configured with Jaeger")
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics with Prometheus exporter."""
        # Setup Prometheus metrics reader
        prometheus_reader = PrometheusMetricReader()
        
        # Configure metrics provider
        metrics.set_meter_provider(MeterProvider(
            resource=self.resource,
            metric_readers=[prometheus_reader]
        ))
        
        self.meter = metrics.get_meter(__name__)
        
        # Create A2A-specific metrics
        self.a2a_task_duration = self.meter.create_histogram(
            name="a2a_task_duration_seconds",
            description="Duration of A2A task execution",
            unit="s"
        )
        
        self.a2a_task_counter = self.meter.create_counter(
            name="a2a_tasks_total",
            description="Total number of A2A tasks processed"
        )
        
        self.mcp_tool_calls = self.meter.create_counter(
            name="mcp_tool_calls_total",
            description="Total number of MCP tool calls"
        )
        
        self.mcp_tool_duration = self.meter.create_histogram(
            name="mcp_tool_duration_seconds",
            description="Duration of MCP tool calls",
            unit="s"
        )
        
        self.agent_health = self.meter.create_up_down_counter(
            name="agent_health_status",
            description="Health status of agents (1=healthy, 0=unhealthy)"
        )
        
        logger.info("Metrics configured with Prometheus")
    
    def _instrument_http(self):
        """Instrument HTTP clients for automatic tracing."""
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTP client instrumentation enabled")
    
    @contextmanager
    def trace_a2a_task(self, task_id: str, agent_name: str, skill_name: str):
        """Context manager for tracing A2A task execution."""
        start_time = time.time()
        
        with self.tracer.start_as_current_span(
            f"a2a_task_{skill_name}",
            attributes={
                "a2a.task.id": task_id,
                "a2a.agent.name": agent_name,
                "a2a.skill.name": skill_name,
                "a2a.task.start_time": datetime.utcnow().isoformat()
            }
        ) as span:
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                # Record successful task
                duration = time.time() - start_time
                self.a2a_task_duration.record(duration, {
                    "agent_name": agent_name,
                    "skill_name": skill_name,
                    "status": "success"
                })
                self.a2a_task_counter.add(1, {
                    "agent_name": agent_name,
                    "skill_name": skill_name,
                    "status": "success"
                })
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                # Record failed task
                duration = time.time() - start_time
                self.a2a_task_duration.record(duration, {
                    "agent_name": agent_name,
                    "skill_name": skill_name,
                    "status": "error"
                })
                self.a2a_task_counter.add(1, {
                    "agent_name": agent_name,
                    "skill_name": skill_name,
                    "status": "error"
                })
                raise
    
    @contextmanager
    def trace_mcp_tool_call(self, tool_name: str, agent_name: str, parameters: Dict[str, Any]):
        """Context manager for tracing MCP tool calls."""
        start_time = time.time()
        
        with self.tracer.start_as_current_span(
            f"mcp_tool_{tool_name}",
            attributes={
                "mcp.tool.name": tool_name,
                "mcp.agent.name": agent_name,
                "mcp.tool.parameters": str(parameters),
                "mcp.call.start_time": datetime.utcnow().isoformat()
            }
        ) as span:
            try:
                yield span
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                # Record successful tool call
                duration = time.time() - start_time
                self.mcp_tool_duration.record(duration, {
                    "tool_name": tool_name,
                    "agent_name": agent_name,
                    "status": "success"
                })
                self.mcp_tool_calls.add(1, {
                    "tool_name": tool_name,
                    "agent_name": agent_name,
                    "status": "success"
                })
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                # Record failed tool call
                duration = time.time() - start_time
                self.mcp_tool_duration.record(duration, {
                    "tool_name": tool_name,
                    "agent_name": agent_name,
                    "status": "error"
                })
                self.mcp_tool_calls.add(1, {
                    "tool_name": tool_name,
                    "agent_name": agent_name,
                    "status": "error"
                })
                raise
    
    def record_agent_health(self, agent_name: str, is_healthy: bool):
        """Record agent health status."""
        status = 1 if is_healthy else 0
        self.agent_health.add(status, {"agent_name": agent_name})
        
        # Also create a span for health check
        with self.tracer.start_as_current_span(
            f"agent_health_check",
            attributes={
                "agent.name": agent_name,
                "health.status": "healthy" if is_healthy else "unhealthy",
                "health.timestamp": datetime.utcnow().isoformat()
            }
        ) as span:
            span.set_status(trace.Status(trace.StatusCode.OK))
    
    def create_custom_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a custom span for specific operations."""
        return self.tracer.start_as_current_span(name, attributes=attributes or {})
    
    def add_span_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span."""
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(event_name, attributes or {})

class A2AMetricsCollector:
    """Enhanced metrics collector for A2A-specific metrics."""
    
    def __init__(self, observability_manager: A2AObservabilityManager):
        self.observability = observability_manager
        self.metrics_cache: Dict[str, Any] = {}
        self.collection_interval = 60  # seconds
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": await self._collect_system_stats(),
            "agents": await self._collect_agent_stats(),
            "a2a": await self._collect_a2a_stats(),
            "mcp": await self._collect_mcp_stats()
        }
        
        self.metrics_cache.update(metrics)
        return metrics
    
    async def _collect_system_stats(self) -> Dict[str, Any]:
        """Collect system resource statistics."""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
    
    async def _collect_agent_stats(self) -> Dict[str, Any]:
        """Collect agent-specific statistics."""
        # This would integrate with the actual agent health checks
        return {
            "total_agents": 6,
            "healthy_agents": 6,
            "unhealthy_agents": 0,
            "agent_status": {
                "orchestrator_agent": "healthy",
                "data_loader_agent": "healthy",
                "data_cleaning_agent": "healthy",
                "data_enrichment_agent": "healthy",
                "data_analyst_agent": "healthy",
                "presentation_agent": "healthy"
            }
        }
    
    async def _collect_a2a_stats(self) -> Dict[str, Any]:
        """Collect A2A protocol statistics."""
        return {
            "active_tasks": 0,  # Would integrate with actual task tracking
            "completed_tasks_24h": 0,
            "failed_tasks_24h": 0,
            "average_task_duration": 0.0,
            "task_success_rate": 100.0
        }
    
    async def _collect_mcp_stats(self) -> Dict[str, Any]:
        """Collect MCP protocol statistics."""
        return {
            "active_connections": 0,  # Would integrate with actual MCP tracking
            "tool_calls_24h": 0,
            "failed_tool_calls_24h": 0,
            "average_call_duration": 0.0,
            "tool_success_rate": 100.0
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get cached metrics summary."""
        return self.metrics_cache

# Global observability manager
_observability_manager: Optional[A2AObservabilityManager] = None

def get_observability_manager(service_name: str = "a2a-mcp-system") -> A2AObservabilityManager:
    """Get global observability manager instance."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = A2AObservabilityManager(service_name)
    return _observability_manager

def instrument_fastapi_app(app, service_name: str):
    """Instrument FastAPI application for automatic tracing."""
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    logger.info(f"FastAPI app instrumented for service: {service_name}")

# Decorator for automatic function tracing
def trace_function(operation_name: str = None):
    """Decorator to automatically trace function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            observability = get_observability_manager()
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with observability.tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator 