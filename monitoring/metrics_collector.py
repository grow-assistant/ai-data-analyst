"""
Performance Metrics Collection System for MCP.

Collects and aggregates performance metrics from all components:
- Request latency and throughput
- Resource utilization (CPU, memory)
- Agent availability and health
- Error rates and patterns
- Security events
"""

import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import httpx
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    timestamp: float

@dataclass
class AgentMetrics:
    """Agent-specific performance metrics."""
    agent_name: str
    response_time_ms: float
    requests_per_minute: float
    error_rate_percent: float
    cpu_percent: float
    memory_mb: float
    is_healthy: bool
    timestamp: float

@dataclass
class MCPMetrics:
    """MCP server metrics."""
    total_requests: int
    active_connections: int
    avg_response_time_ms: float
    requests_per_minute: float
    error_rate_percent: float
    rate_limited_requests: int
    blocked_ips: int
    security_events: int
    timestamp: float

class MetricsCollector:
    """Collects and stores performance metrics from all system components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_network_stats = None
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_requests: Dict[str, int] = defaultdict(int)
        
        # Agent endpoints for health checks
        self.agent_endpoints = {
            "mcp_server": f"http://localhost:{config.get('mcp_port', 10001)}",
            "orchestrator": f"http://localhost:{config.get('orchestrator_port', 10000)}",
            "data_loader": f"http://localhost:{config.get('data_loader_port', 10006)}",
            "data_analyst": f"http://localhost:{config.get('data_analyst_port', 10007)}"
        }
        
        self.mcp_api_key = config.get('mcp_api_key', 'mcp-dev-key')
        
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network stats
            net_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                process_count=process_count,
                timestamp=time.time()
            )
            
            self.metrics_history['system'].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    async def collect_agent_metrics(self, agent_name: str, endpoint: str) -> Optional[AgentMetrics]:
        """Collect metrics for a specific agent."""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/health")
                
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            is_healthy = response.status_code == 200
            
            # Track request times for this agent
            self.request_times[agent_name].append(response_time)
            self.total_requests[agent_name] += 1
            
            if not is_healthy:
                self.error_counts[agent_name] += 1
            
            # Calculate rates
            recent_requests = len(self.request_times[agent_name])
            time_window = 60  # 1 minute
            requests_per_minute = min(recent_requests, 60)  # Approximate
            
            error_rate = (self.error_counts[agent_name] / max(self.total_requests[agent_name], 1)) * 100
            
            # Try to get process-specific metrics (simplified)
            cpu_percent = 0.0
            memory_mb = 0.0
            
            try:
                # This is a simplified approach - in production you'd track process PIDs
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                    if agent_name.replace('_', '-') in proc.info['name']:
                        cpu_percent = proc.info['cpu_percent'] or 0.0
                        memory_mb = (proc.info['memory_info'].rss / 1024 / 1024) if proc.info['memory_info'] else 0.0
                        break
            except:
                pass
            
            metrics = AgentMetrics(
                agent_name=agent_name,
                response_time_ms=response_time,
                requests_per_minute=requests_per_minute,
                error_rate_percent=error_rate,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                is_healthy=is_healthy,
                timestamp=time.time()
            )
            
            self.metrics_history[f'agent_{agent_name}'].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {agent_name}: {e}")
            self.error_counts[agent_name] += 1
            return None
    
    async def collect_mcp_metrics(self) -> Optional[MCPMetrics]:
        """Collect MCP server specific metrics."""
        try:
            headers = {"X-API-Key": self.mcp_api_key}
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get security status
                security_response = await client.get(
                    f"{self.agent_endpoints['mcp_server']}/security/status",
                    headers=headers
                )
                
                health_response = await client.get(
                    f"{self.agent_endpoints['mcp_server']}/health"
                )
            
            security_data = security_response.json() if security_response.status_code == 200 else {}
            
            # Extract metrics from security status
            rate_limited = len(security_data.get('rate_limiting', {}).get('current_limits', {}))
            blocked_ips = len(security_data.get('blocked_ips', []))
            security_events = len(security_data.get('security_events', []))
            
            # Calculate MCP-specific metrics
            mcp_requests = self.total_requests.get('mcp_server', 0)
            mcp_errors = self.error_counts.get('mcp_server', 0)
            error_rate = (mcp_errors / max(mcp_requests, 1)) * 100
            
            avg_response_time = 0.0
            if self.request_times['mcp_server']:
                avg_response_time = sum(self.request_times['mcp_server']) / len(self.request_times['mcp_server'])
            
            metrics = MCPMetrics(
                total_requests=mcp_requests,
                active_connections=0,  # Would need more sophisticated tracking
                avg_response_time_ms=avg_response_time,
                requests_per_minute=min(len(self.request_times['mcp_server']), 60),
                error_rate_percent=error_rate,
                rate_limited_requests=rate_limited,
                blocked_ips=blocked_ips,
                security_events=security_events,
                timestamp=time.time()
            )
            
            self.metrics_history['mcp'].append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting MCP metrics: {e}")
            return None
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all components."""
        metrics = {}
        
        # System metrics
        system_metrics = await self.collect_system_metrics()
        if system_metrics:
            metrics['system'] = asdict(system_metrics)
        
        # Agent metrics
        agent_metrics = {}
        for agent_name, endpoint in self.agent_endpoints.items():
            if agent_name != 'mcp_server':  # Handle MCP separately
                agent_metric = await self.collect_agent_metrics(agent_name, endpoint)
                if agent_metric:
                    agent_metrics[agent_name] = asdict(agent_metric)
        
        metrics['agents'] = agent_metrics
        
        # MCP metrics
        mcp_metrics = await self.collect_mcp_metrics()
        if mcp_metrics:
            metrics['mcp'] = asdict(mcp_metrics)
        
        return metrics
    
    def get_metrics_summary(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Get summarized metrics for the specified time window."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        summary = {}
        
        for metric_type, metric_list in self.metrics_history.items():
            recent_metrics = [m for m in metric_list if getattr(m, 'timestamp', 0) >= cutoff_time]
            
            if not recent_metrics:
                continue
                
            if metric_type == 'system':
                summary[metric_type] = {
                    'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                    'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                    'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
                    'max_memory_percent': max(m.memory_percent for m in recent_metrics),
                    'sample_count': len(recent_metrics)
                }
            elif metric_type.startswith('agent_'):
                agent_name = metric_type[6:]  # Remove 'agent_' prefix
                summary[metric_type] = {
                    'avg_response_time_ms': sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
                    'max_response_time_ms': max(m.response_time_ms for m in recent_metrics),
                    'error_rate_percent': recent_metrics[-1].error_rate_percent if recent_metrics else 0,
                    'is_healthy': recent_metrics[-1].is_healthy if recent_metrics else False,
                    'sample_count': len(recent_metrics)
                }
            elif metric_type == 'mcp':
                summary[metric_type] = {
                    'avg_response_time_ms': sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics),
                    'total_requests': recent_metrics[-1].total_requests if recent_metrics else 0,
                    'error_rate_percent': recent_metrics[-1].error_rate_percent if recent_metrics else 0,
                    'blocked_ips': recent_metrics[-1].blocked_ips if recent_metrics else 0,
                    'security_events': recent_metrics[-1].security_events if recent_metrics else 0,
                    'sample_count': len(recent_metrics)
                }
        
        return summary
    
    def export_metrics(self, filepath: str, time_window_hours: int = 24):
        """Export metrics to JSON file."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'metrics': {}
        }
        
        for metric_type, metric_list in self.metrics_history.items():
            recent_metrics = [
                asdict(m) if hasattr(m, '__dict__') else m 
                for m in metric_list 
                if getattr(m, 'timestamp', 0) >= cutoff_time
            ]
            export_data['metrics'][metric_type] = recent_metrics
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")


class MetricsServer:
    """HTTP server for exposing metrics."""
    
    def __init__(self, collector: MetricsCollector, port: int = 10002):
        self.collector = collector
        self.port = port
        
    async def get_metrics_endpoint(self) -> Dict[str, Any]:
        """Get current metrics."""
        return await self.collector.collect_all_metrics()
    
    async def get_summary_endpoint(self, time_window: int = 5) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.collector.get_metrics_summary(time_window)
    
    async def get_health_endpoint(self) -> Dict[str, Any]:
        """Health check for metrics service."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics_collected": sum(len(m) for m in self.collector.metrics_history.values())
        }


async def main():
    """Main metrics collection loop."""
    config = {
        'mcp_port': 10001,
        'orchestrator_port': 10000,
        'data_loader_port': 10006,
        'data_analyst_port': 10007,
        'mcp_api_key': 'mcp-dev-key'
    }
    
    collector = MetricsCollector(config)
    
    print("Starting metrics collection...")
    
    try:
        while True:
            metrics = await collector.collect_all_metrics()
            summary = collector.get_metrics_summary()
            
            print(f"\n--- Metrics Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            
            if 'system' in summary:
                sys_metrics = summary['system']
                print(f"System: CPU {sys_metrics['avg_cpu_percent']:.1f}%, Memory {sys_metrics['avg_memory_percent']:.1f}%")
            
            if 'agents' in metrics:
                print("Agents:")
                for agent_name, agent_data in metrics['agents'].items():
                    status = "✓" if agent_data['is_healthy'] else "✗"
                    print(f"  {agent_name}: {status} {agent_data['response_time_ms']:.1f}ms")
            
            if 'mcp' in summary:
                mcp_metrics = summary['mcp']
                print(f"MCP: {mcp_metrics['avg_response_time_ms']:.1f}ms avg, {mcp_metrics['total_requests']} requests")
            
            # Wait before next collection
            await asyncio.sleep(30)  # Collect every 30 seconds
            
    except KeyboardInterrupt:
        print("\nStopping metrics collection...")
        
        # Export final metrics
        export_path = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        collector.export_metrics(export_path)
        print(f"Final metrics exported to {export_path}")


if __name__ == "__main__":
    asyncio.run(main()) 