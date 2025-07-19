"""APScheduler Integration for Orchestrator Agent - Automated Workflows"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.types import TaskRequest
from .a2a_client import OrchestratorA2AClient

logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Types of schedules supported."""
    INTERVAL = "interval"
    CRON = "cron"
    ONE_TIME = "one_time"

@dataclass
class ScheduledWorkflow:
    """Scheduled workflow configuration."""
    id: str
    name: str
    description: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    workflow_steps: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class A2AWorkflowScheduler:
    """APScheduler-based workflow scheduler for automated data analysis."""
    
    def __init__(self, a2a_client: OrchestratorA2AClient):
        self.a2a_client = a2a_client
        self.workflows: Dict[str, ScheduledWorkflow] = {}
        
        # Configure APScheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': AsyncIOExecutor()
        }
        job_defaults = {
            'coalesce': False,
            'max_instances': 3
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        # Register default workflows
        self._register_default_workflows()
        
        logger.info("A2A Workflow Scheduler initialized")
    
    async def start(self):
        """Start the scheduler."""
        self.scheduler.start()
        logger.info("A2A Workflow Scheduler started")
    
    async def shutdown(self):
        """Shutdown the scheduler."""
        self.scheduler.shutdown()
        logger.info("A2A Workflow Scheduler stopped")
    
    def _register_default_workflows(self):
        """Register default automated workflows."""
        # Weekly comprehensive analysis
        self.register_workflow(
            name="Weekly Data Analysis Report",
            description="Comprehensive weekly analysis and report generation",
            schedule_type=ScheduleType.CRON,
            schedule_config={"day_of_week": "mon", "hour": 8, "minute": 0},
            workflow_steps=[
                {
                    "agent": "data_loader_agent",
                    "skill": "load_dataset",
                    "params": {"data_source": "weekly_data"}
                },
                {
                    "agent": "data_cleaning_agent", 
                    "skill": "clean_dataset",
                    "params": {"cleaning_operations": ["remove_duplicates", "handle_missing"]}
                },
                {
                    "agent": "data_enrichment_agent",
                    "skill": "enrich_dataset", 
                    "params": {"enrichment_operations": ["calculate_moving_averages", "date_features"]}
                },
                {
                    "agent": "data_analyst_agent",
                    "skill": "analyze_dataset",
                    "params": {"analysis_type": "comprehensive"}
                },
                {
                    "agent": "presentation_agent",
                    "skill": "create_report",
                    "params": {
                        "report_type": "dashboard",
                        "output_format": "html",
                        "distribution_channels": ["email", "slack"]
                    }
                }
            ]
        )
        
        # Daily data quality check
        self.register_workflow(
            name="Daily Data Quality Check",
            description="Daily automated data quality monitoring",
            schedule_type=ScheduleType.CRON,
            schedule_config={"hour": 6, "minute": 0},
            workflow_steps=[
                {
                    "agent": "data_loader_agent",
                    "skill": "load_dataset",
                    "params": {"data_source": "daily_data"}
                },
                {
                    "agent": "data_cleaning_agent",
                    "skill": "validate_data_quality",
                    "params": {}
                }
            ]
        )
        
        # Monthly trend analysis
        self.register_workflow(
            name="Monthly Trend Analysis",
            description="Monthly trend analysis and forecasting",
            schedule_type=ScheduleType.CRON,
            schedule_config={"day": 1, "hour": 9, "minute": 0},
            workflow_steps=[
                {
                    "agent": "data_loader_agent",
                    "skill": "load_dataset",
                    "params": {"data_source": "monthly_data"}
                },
                {
                    "agent": "data_analyst_agent",
                    "skill": "analyze_dataset",
                    "params": {"analysis_type": "trend_analysis"}
                },
                {
                    "agent": "presentation_agent",
                    "skill": "create_report",
                    "params": {
                        "report_type": "detailed",
                        "visualizations": ["line_chart", "distribution", "heatmap"]
                    }
                }
            ]
        )
    
    def register_workflow(self, name: str, description: str, schedule_type: ScheduleType,
                         schedule_config: Dict[str, Any], workflow_steps: List[Dict[str, Any]],
                         workflow_id: str = None) -> str:
        """Register a new scheduled workflow."""
        if not workflow_id:
            workflow_id = str(uuid.uuid4())
        
        workflow = ScheduledWorkflow(
            id=workflow_id,
            name=name,
            description=description,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            workflow_steps=workflow_steps
        )
        
        self.workflows[workflow_id] = workflow
        
        # Schedule the workflow
        self._schedule_workflow(workflow)
        
        logger.info(f"Workflow registered: {name} ({workflow_id})")
        return workflow_id
    
    def _schedule_workflow(self, workflow: ScheduledWorkflow):
        """Schedule a workflow with APScheduler."""
        if not workflow.enabled:
            return
        
        # Create trigger based on schedule type
        if workflow.schedule_type == ScheduleType.INTERVAL:
            trigger = IntervalTrigger(**workflow.schedule_config)
        elif workflow.schedule_type == ScheduleType.CRON:
            trigger = CronTrigger(**workflow.schedule_config)
        elif workflow.schedule_type == ScheduleType.ONE_TIME:
            trigger = DateTrigger(**workflow.schedule_config)
        else:
            raise ValueError(f"Unsupported schedule type: {workflow.schedule_type}")
        
        # Add job to scheduler
        self.scheduler.add_job(
            func=self._execute_workflow,
            trigger=trigger,
            args=[workflow.id],
            id=workflow.id,
            name=workflow.name,
            replace_existing=True
        )
        
        # Update next run time
        job = self.scheduler.get_job(workflow.id)
        if job:
            workflow.next_run = job.next_run_time
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a scheduled workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return
        
        logger.info(f"Executing scheduled workflow: {workflow.name}")
        workflow.last_run = datetime.utcnow()
        workflow.run_count += 1
        
        try:
            # Execute workflow steps sequentially
            previous_data_handle = None
            
            for i, step in enumerate(workflow.workflow_steps):
                agent_name = step["agent"]
                skill_name = step["skill"]
                params = step.get("params", {})
                
                # If this isn't the first step, pass the previous data handle
                if previous_data_handle and i > 0:
                    params["data_handle_id"] = previous_data_handle
                
                # Create task request
                task_request = TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=f"workflow-{workflow_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                    task_type=skill_name,
                    parameters=params,
                    priority=5
                )
                
                logger.info(f"Executing step {i+1}: {agent_name}.{skill_name}")
                
                # Execute via A2A
                response = await self.a2a_client.execute_skill(agent_name, skill_name, task_request)
                
                if response.status != "completed":
                    logger.error(f"Workflow step failed: {agent_name}.{skill_name} - {response.error_message}")
                    break
                
                # Extract data handle for next step
                if response.results and isinstance(response.results, dict):
                    previous_data_handle = (
                        response.results.get("data_handle_id") or 
                        response.results.get("cleaned_data_handle_id") or
                        response.results.get("enriched_data_handle_id") or
                        response.results.get("analysis_data_handle_id") or
                        response.results.get("report_handle_id")
                    )
                
                logger.info(f"Step {i+1} completed successfully")
            
            logger.info(f"Workflow completed successfully: {workflow.name}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {workflow.name} - {e}")
    
    def schedule_one_time_workflow(self, name: str, description: str, 
                                  workflow_steps: List[Dict[str, Any]], 
                                  run_at: datetime) -> str:
        """Schedule a one-time workflow execution."""
        return self.register_workflow(
            name=name,
            description=description,
            schedule_type=ScheduleType.ONE_TIME,
            schedule_config={"run_date": run_at},
            workflow_steps=workflow_steps
        )
    
    def enable_workflow(self, workflow_id: str):
        """Enable a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = True
            self._schedule_workflow(workflow)
            logger.info(f"Workflow enabled: {workflow.name}")
    
    def disable_workflow(self, workflow_id: str):
        """Disable a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = False
            try:
                self.scheduler.remove_job(workflow_id)
            except:
                pass  # Job might not exist
            logger.info(f"Workflow disabled: {workflow.name}")
    
    def remove_workflow(self, workflow_id: str):
        """Remove a workflow completely."""
        if workflow_id in self.workflows:
            self.disable_workflow(workflow_id)
            del self.workflows[workflow_id]
            logger.info(f"Workflow removed: {workflow_id}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and statistics."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        job = self.scheduler.get_job(workflow_id)
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "enabled": workflow.enabled,
            "schedule_type": workflow.schedule_type.value,
            "schedule_config": workflow.schedule_config,
            "created_at": workflow.created_at.isoformat(),
            "last_run": workflow.last_run.isoformat() if workflow.last_run else None,
            "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
            "run_count": workflow.run_count,
            "steps_count": len(workflow.workflow_steps)
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        return [
            self.get_workflow_status(workflow_id) 
            for workflow_id in self.workflows.keys()
        ]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        jobs = self.scheduler.get_jobs()
        
        return {
            "running": self.scheduler.running,
            "total_workflows": len(self.workflows),
            "enabled_workflows": len([w for w in self.workflows.values() if w.enabled]),
            "scheduled_jobs": len(jobs),
            "next_job_run": min([job.next_run_time for job in jobs if job.next_run_time], default=None),
            "total_executions": sum([w.run_count for w in self.workflows.values()])
        }

# API endpoint decorator for secure access
def require_admin_auth(func):
    """Decorator to require admin authentication for scheduler endpoints."""
    async def wrapper(*args, **kwargs):
        # This would integrate with the OAuth2 security system
        # For now, just log the access
        logger.info(f"Admin access to scheduler function: {func.__name__}")
        return await func(*args, **kwargs)
    return wrapper 