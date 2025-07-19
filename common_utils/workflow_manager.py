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
Advanced Workflow Management System for Multi-Agent Framework.

Provides dynamic task routing, progress monitoring, and intelligent workflow 
orchestration with support for conditional execution, retry logic, and real-time
progress tracking.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass

from .session_manager import get_session_manager, Session
from .observability import get_observability_manager, trace_operation
from .enhanced_logging import get_logger, correlated_operation

logger = get_logger(__name__)

class TaskStatus(Enum):
    """Status of workflow tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class WorkflowStatus(Enum):
    """Status of entire workflows."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None

class TaskDefinition(BaseModel):
    """Definition of a workflow task."""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str
    agent_name: str
    skill_name: str
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []
    optional: bool = False
    retry_count: int = 3
    timeout_seconds: int = 300
    condition: Optional[str] = None  # Python expression for conditional execution
    
class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow."""
    id: str = Field(default_factory=lambda: f"workflow_{uuid.uuid4().hex[:8]}")
    name: str
    description: str = ""
    tasks: List[TaskDefinition] = []
    parallel_groups: List[List[str]] = []  # Groups of tasks that can run in parallel
    global_timeout_seconds: int = 3600
    metadata: Dict[str, Any] = {}

class WorkflowExecution(BaseModel):
    """Runtime state of a workflow execution."""
    id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    workflow_id: str
    session_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_results: Dict[str, TaskResult] = {}
    current_tasks: List[str] = []
    progress_percentage: float = 0.0
    error_log: List[str] = []
    context: Dict[str, Any] = {}  # Shared context between tasks

class WorkflowManager:
    """
    Advanced workflow manager with dynamic routing and progress monitoring.
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.session_manager = get_session_manager()
        self.observability = get_observability_manager()
        self.agent_endpoints = {}
        self._load_agent_endpoints()
        
    def _load_agent_endpoints(self):
        """Load agent endpoints from configuration."""
        try:
            from .agent_config import get_agent_endpoints
            self.agent_endpoints = get_agent_endpoints()
        except Exception as e:
            logger.warning(f"Failed to load agent endpoints: {e}")
            # Fallback endpoints
            self.agent_endpoints = {
                "data_loader": "http://localhost:10006",
                "data_cleaning": "http://localhost:10008",
                "data_enrichment": "http://localhost:10009",
                "data_analyst": "http://localhost:10007",
                "presentation": "http://localhost:10010",
                "schema_profiler": "http://localhost:10012",
                "rootcause_analyst": "http://localhost:10011"
            }
    
    def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """Register a new workflow definition."""
        self.workflows[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
        return workflow.id
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by ID."""
        return self.workflows.get(workflow_id)
    
    def create_execution(self, workflow_id: str, session_id: Optional[str] = None, 
                        context: Dict[str, Any] = None) -> WorkflowExecution:
        """Create a new workflow execution."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            session_id=session_id,
            context=context or {}
        )
        
        self.executions[execution.id] = execution
        
        # Log in session if provided
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session:
                self.session_manager.add_event(session_id, "workflow_created", {
                    "workflow_id": workflow_id,
                    "execution_id": execution.id,
                    "workflow_name": workflow.name
                })
        
        logger.info(f"Created execution {execution.id} for workflow {workflow.name}")
        return execution
    
    async def execute_workflow(self, execution_id: str) -> WorkflowExecution:
        """Execute a workflow with dynamic routing and progress monitoring."""
        execution = self.executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")
        
        workflow = self.get_workflow(execution.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {execution.workflow_id}")
        
        with correlated_operation("workflow_execution", 
                                 workflow_id=execution.workflow_id, 
                                 execution_id=execution_id):
            try:
                execution.status = WorkflowStatus.RUNNING
                execution.started_at = datetime.utcnow()
                
                logger.info(f"Starting workflow execution: {workflow.name}")
                
                # Create task dependency graph
                task_graph = self._build_task_graph(workflow.tasks)
                
                # Execute tasks according to dependencies and parallel groups
                await self._execute_task_graph(execution, workflow, task_graph)
                
                # Determine final status
                if all(result.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] 
                      for result in execution.task_results.values()):
                    execution.status = WorkflowStatus.COMPLETED
                    execution.progress_percentage = 100.0
                    logger.info(f"Workflow {workflow.name} completed successfully")
                else:
                    execution.status = WorkflowStatus.FAILED
                    logger.error(f"Workflow {workflow.name} failed")
                
                execution.completed_at = datetime.utcnow()
                
                # Record metrics
                self.observability.record_pipeline_execution(
                    "workflow", 
                    execution.status.value
                )
                
                return execution
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.error_log.append(str(e))
                execution.completed_at = datetime.utcnow()
                
                logger.exception(f"Workflow execution failed: {e}")
                
                # Log in session
                if execution.session_id:
                    self.session_manager.add_event(execution.session_id, "workflow_failed", {
                        "execution_id": execution_id,
                        "error": str(e)
                    })
                
                self.observability.record_pipeline_execution("workflow", "failed")
                raise
    
    def _build_task_graph(self, tasks: List[TaskDefinition]) -> Dict[str, List[str]]:
        """Build a dependency graph for tasks."""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        return graph
    
    async def _execute_task_graph(self, execution: WorkflowExecution, 
                                  workflow: WorkflowDefinition,
                                  task_graph: Dict[str, List[str]]):
        """Execute tasks according to dependency graph."""
        task_dict = {task.id: task for task in workflow.tasks}
        completed_tasks = set()
        running_tasks = {}
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task_id, dependencies in task_graph.items():
                if (task_id not in completed_tasks and 
                    task_id not in running_tasks and
                    all(dep in completed_tasks for dep in dependencies)):
                    ready_tasks.append(task_id)
            
            if not ready_tasks and not running_tasks:
                # Deadlock - some tasks have unmet dependencies
                unfinished = set(task_graph.keys()) - completed_tasks
                execution.error_log.append(f"Deadlock detected: {unfinished}")
                break
            
            # Start ready tasks
            for task_id in ready_tasks:
                task = task_dict[task_id]
                
                # Check condition if specified
                if task.condition and not self._evaluate_condition(task.condition, execution.context):
                    execution.task_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED
                    )
                    completed_tasks.add(task_id)
                    continue
                
                # Start task execution
                task_future = asyncio.create_task(
                    self._execute_task(execution, task)
                )
                running_tasks[task_id] = task_future
                execution.current_tasks.append(task_id)
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for future in done:
                    for task_id, task_future in list(running_tasks.items()):
                        if task_future == future:
                            try:
                                result = await future
                                execution.task_results[task_id] = result
                                completed_tasks.add(task_id)
                                
                                if task_id in execution.current_tasks:
                                    execution.current_tasks.remove(task_id)
                                
                                # Update progress
                                execution.progress_percentage = (
                                    len(completed_tasks) / len(workflow.tasks) * 100
                                )
                                
                                logger.info(f"Task {task_id} completed with status {result.status}")
                                
                            except Exception as e:
                                logger.error(f"Task {task_id} failed: {e}")
                                execution.task_results[task_id] = TaskResult(
                                    task_id=task_id,
                                    status=TaskStatus.FAILED,
                                    error=str(e)
                                )
                                completed_tasks.add(task_id)
                                
                                if task_id in execution.current_tasks:
                                    execution.current_tasks.remove(task_id)
                            
                            del running_tasks[task_id]
                            break
    
    async def _execute_task(self, execution: WorkflowExecution, 
                           task: TaskDefinition) -> TaskResult:
        """Execute a single task with retry logic."""
        logger.info(f"Executing task: {task.name} ({task.id})")
        
        for attempt in range(task.retry_count):
            try:
                with trace_operation(f"task_execution_{task.name}", 
                                   task_id=task.id, 
                                   attempt=attempt + 1):
                    start_time = datetime.utcnow()
                    
                    # Prepare parameters with context substitution
                    parameters = self._substitute_parameters(task.parameters, execution.context)
                    
                    # Make agent call
                    result = await self._call_agent(task.agent_name, task.skill_name, parameters)
                    
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Update execution context with result
                    execution.context[f"task_{task.id}_result"] = result
                    
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.COMPLETED,
                        output=result,
                        duration=duration
                    )
                    
            except Exception as e:
                if attempt == task.retry_count - 1:
                    # Final attempt failed
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    
                    if task.optional:
                        logger.warning(f"Optional task {task.name} failed, continuing: {e}")
                        return TaskResult(
                            task_id=task.id,
                            status=TaskStatus.SKIPPED,
                            error=str(e),
                            duration=duration
                        )
                    else:
                        logger.error(f"Task {task.name} failed after {task.retry_count} attempts: {e}")
                        return TaskResult(
                            task_id=task.id,
                            status=TaskStatus.FAILED,
                            error=str(e),
                            duration=duration
                        )
                else:
                    logger.warning(f"Task {task.name} attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _substitute_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute context variables in task parameters."""
        substituted = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Context variable substitution
                var_name = value[2:-1]
                substituted[key] = context.get(var_name, value)
            else:
                substituted[key] = value
        return substituted
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression safely."""
        try:
            # Simple evaluation - in production, use a more secure expression evaluator
            safe_context = {k: v for k, v in context.items() if isinstance(k, str) and k.isidentifier()}
            return bool(eval(condition, {"__builtins__": {}}, safe_context))
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return True  # Default to true if evaluation fails
    
    async def _call_agent(self, agent_name: str, skill_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make a call to an agent."""
        endpoint = self.agent_endpoints.get(agent_name)
        if not endpoint:
            raise ValueError(f"Agent endpoint not found: {agent_name}")
        
        # Import here to avoid circular dependencies
        from .agent_security import make_secure_agent_call
        
        result = await make_secure_agent_call(
            agent_name, 
            f"{endpoint}/{skill_name}",
            parameters
        )
        
        return result
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress_percentage": execution.progress_percentage,
            "current_tasks": execution.current_tasks,
            "completed_tasks": len(execution.task_results),
            "total_tasks": len(self.workflows[execution.workflow_id].tasks),
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "errors": execution.error_log
        }

# Global instance
_workflow_manager = None

def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager 