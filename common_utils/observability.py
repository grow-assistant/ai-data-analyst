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
Observability Module for Multi-Agent Framework.

Provides OpenTelemetry tracing, Prometheus metrics, and comprehensive monitoring
for the AI Data Analyst Multi-Agent Framework.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import asyncio

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Prometheus imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ObservabilityManager:
    """
    Centralized manager for observability, tracing, and metrics.
    """
    
    def __init__(self, service_name: str = "ai-data-analyst", service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.enabled = OTEL_AVAILABLE or PROMETHEUS_AVAILABLE
        
        # Initialize OpenTelemetry
        self._init_opentelemetry()
        
        # Initialize Prometheus
        self._init_prometheus()
        
        logger.info(f"ObservabilityManager initialized. OpenTelemetry: {OTEL_AVAILABLE}, Prometheus: {PROMETHEUS_AVAILABLE}")

    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
            self.tracer = None
            self.meter = None
            return

        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: self.service_version,
            })

            # Set up tracing
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Add OTLP exporter (optional - can be configured with environment variables)
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint="http://localhost:4317",  # Default OTLP endpoint
                    insecure=True
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                logger.info("OTLP trace exporter configured")
            except Exception as e:
                logger.debug(f"OTLP trace exporter not configured: {e}")

            self.tracer = trace.get_tracer(__name__)

            # Set up metrics
            try:
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint="http://localhost:4317", insecure=True),
                    export_interval_millis=30000,  # Export every 30 seconds
                )
                metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
                logger.info("OTLP metrics exporter configured")
            except Exception as e:
                logger.debug(f"OTLP metrics exporter not configured: {e}")
                # Fallback to default meter provider
                metrics.set_meter_provider(MeterProvider(resource=resource))

            self.meter = metrics.get_meter(__name__)
            
            # Create common metrics
            self._create_otel_metrics()
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.tracer = None
            self.meter = None

    def _create_otel_metrics(self):
        """Create common OpenTelemetry metrics."""
        if not self.meter:
            return
            
        try:
            self.otel_request_counter = self.meter.create_counter(
                name="agent_requests_total",
                description="Total number of agent requests",
                unit="1"
            )
            
            self.otel_request_duration = self.meter.create_histogram(
                name="agent_request_duration_seconds",
                description="Duration of agent requests in seconds",
                unit="s"
            )
            
            self.otel_pipeline_counter = self.meter.create_counter(
                name="pipeline_executions_total",
                description="Total number of pipeline executions",
                unit="1"
            )
            
        except Exception as e:
            logger.error(f"Failed to create OpenTelemetry metrics: {e}")

    def _init_prometheus(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
            self.prometheus_registry = None
            return

        try:
            # Create custom registry
            self.prometheus_registry = CollectorRegistry()
            
            # Create metrics
            self.agent_requests_total = Counter(
                'agent_requests_total',
                'Total number of agent requests',
                ['agent_name', 'method', 'status'],
                registry=self.prometheus_registry
            )
            
            self.agent_request_duration = Histogram(
                'agent_request_duration_seconds',
                'Duration of agent requests in seconds',
                ['agent_name', 'method'],
                registry=self.prometheus_registry
            )
            
            self.pipeline_executions_total = Counter(
                'pipeline_executions_total',
                'Total number of pipeline executions',
                ['pipeline_type', 'status'],
                registry=self.prometheus_registry
            )
            
            self.active_sessions = Gauge(
                'active_sessions_total',
                'Number of active sessions',
                registry=self.prometheus_registry
            )
            
            self.data_handles_total = Gauge(
                'data_handles_total',
                'Total number of data handles',
                registry=self.prometheus_registry
            )
            
            self.system_info = Info(
                'system_info',
                'System information',
                registry=self.prometheus_registry
            )
            
            # Set system info
            self.system_info.info({
                'service_name': self.service_name,
                'service_version': self.service_version
            })
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
            self.prometheus_registry = None

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations."""
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                
                start_time = time.time()
                try:
                    yield span
                    span.set_status(trace.Status(trace.StatusCode.OK))
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration_seconds", duration)
        else:
            # No-op context manager when tracing is not available
            yield None

    def record_agent_request(self, agent_name: str, method: str, status: str, duration: float):
        """Record metrics for an agent request."""
        # Prometheus metrics
        if self.prometheus_registry:
            self.agent_requests_total.labels(
                agent_name=agent_name,
                method=method,
                status=status
            ).inc()
            
            self.agent_request_duration.labels(
                agent_name=agent_name,
                method=method
            ).observe(duration)
        
        # OpenTelemetry metrics
        if self.meter:
            try:
                self.otel_request_counter.add(1, {
                    "agent_name": agent_name,
                    "method": method,
                    "status": status
                })
                
                self.otel_request_duration.record(duration, {
                    "agent_name": agent_name,
                    "method": method
                })
            except Exception as e:
                logger.debug(f"Failed to record OpenTelemetry metrics: {e}")

    def record_pipeline_execution(self, pipeline_type: str, status: str):
        """Record metrics for a pipeline execution."""
        # Prometheus metrics
        if self.prometheus_registry:
            self.pipeline_executions_total.labels(
                pipeline_type=pipeline_type,
                status=status
            ).inc()
        
        # OpenTelemetry metrics
        if self.meter:
            try:
                self.otel_pipeline_counter.add(1, {
                    "pipeline_type": pipeline_type,
                    "status": status
                })
            except Exception as e:
                logger.debug(f"Failed to record OpenTelemetry pipeline metric: {e}")

    def update_session_count(self, count: int):
        """Update the active session count."""
        if self.prometheus_registry:
            self.active_sessions.set(count)

    def update_data_handles_count(self, count: int):
        """Update the data handles count."""
        if self.prometheus_registry:
            self.data_handles_total.set(count)

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if self.prometheus_registry:
            return generate_latest(self.prometheus_registry).decode('utf-8')
        return "# Prometheus metrics not available\n"

    def instrument_function(self, operation_name: str = None):
        """Decorator to instrument functions with tracing and metrics."""
        def decorator(func: Callable):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.trace_operation(name, function=func.__name__):
                        start_time = time.time()
                        try:
                            result = await func(*args, **kwargs)
                            duration = time.time() - start_time
                            self.record_agent_request("system", func.__name__, "success", duration)
                            return result
                        except Exception as e:
                            duration = time.time() - start_time
                            self.record_agent_request("system", func.__name__, "error", duration)
                            raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.trace_operation(name, function=func.__name__):
                        start_time = time.time()
                        try:
                            result = func(*args, **kwargs)
                            duration = time.time() - start_time
                            self.record_agent_request("system", func.__name__, "success", duration)
                            return result
                        except Exception as e:
                            duration = time.time() - start_time
                            self.record_agent_request("system", func.__name__, "error", duration)
                            raise
                return sync_wrapper
        return decorator

# Global instance
_observability_manager = None

def get_observability_manager() -> ObservabilityManager:
    """Get the global observability manager instance."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager

# Convenience functions
def trace_operation(operation_name: str, **attributes):
    """Convenience function for tracing operations."""
    return get_observability_manager().trace_operation(operation_name, **attributes)

def instrument(operation_name: str = None):
    """Convenience decorator for instrumenting functions."""
    return get_observability_manager().instrument_function(operation_name) 