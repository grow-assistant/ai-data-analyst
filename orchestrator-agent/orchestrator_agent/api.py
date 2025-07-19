"""Secure API Endpoints for Orchestrator Agent - On-Demand Analysis"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add parent directory for common_utils access
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.security import security_manager, PermissionLevel, AuthenticationError, AuthorizationError
from common_utils.types import TaskRequest, TaskResponse
from monitoring.observability import get_observability_manager, instrument_fastapi_app
from .scheduler import A2AWorkflowScheduler, ScheduleType
from .a2a_client import OrchestratorA2AClient

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalysisRequest(BaseModel):
    """Request model for on-demand analysis."""
    data_source: str = Field(..., description="Source of data to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    output_format: str = Field(default="html", description="Output format for report")
    distribution_channels: List[str] = Field(default_factory=list, description="Channels to distribute results")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")

class AnalysisResponse(BaseModel):
    """Response model for analysis request."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Current status of analysis")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")

class AnalysisStatus(BaseModel):
    """Model for analysis status."""
    analysis_id: str
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    current_step: str
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None

class WorkflowRequest(BaseModel):
    """Request model for workflow scheduling."""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    schedule_type: str = Field(..., description="Schedule type (interval, cron, one_time)")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")
    workflow_steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

# FastAPI app initialization
app = FastAPI(
    title="A2A Orchestrator API",
    description="Secure API for on-demand data analysis and workflow management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security setup
security = HTTPBearer(auto_error=False)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.local"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global state
a2a_client: Optional[OrchestratorA2AClient] = None
scheduler: Optional[A2AWorkflowScheduler] = None
analysis_tracker: Dict[str, AnalysisStatus] = {}

# Authentication dependency
async def get_current_agent(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Validate OAuth2 token and return agent info."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Validate token with security manager
        payload = await security_manager.oauth2_manager.validate_token(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # Log successful authentication
        await security_manager.audit_logger.log_action(
            agent_id=payload.get("client_id", "unknown"),
            action="api_access",
            resource=f"orchestrator:{request.url.path}",
            result="success",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return payload
        
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Authentication failed")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")

# Authorization dependency
async def require_permission(permission: PermissionLevel):
    """Create dependency for permission checking."""
    async def check_permission(
        request: Request,
        agent_info: Dict[str, Any] = Depends(get_current_agent)
    ):
        try:
            agent_id = agent_info.get("client_id")
            resource = f"orchestrator:{request.url.path}"
            
            # Check authorization
            authorized = await security_manager.acl_manager.check_permission(
                agent_id, resource, permission
            )
            
            if not authorized:
                await security_manager.audit_logger.log_action(
                    agent_id=agent_id,
                    action="authorization_denied",
                    resource=resource,
                    result="failure",
                    details={"required_permission": permission.value}
                )
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return agent_info
            
        except AuthorizationError:
            raise HTTPException(status_code=403, detail="Authorization failed")
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            raise HTTPException(status_code=500, detail="Authorization service error")
    
    return check_permission

@app.on_event("startup")
async def startup_event():
    """Initialize application components."""
    global a2a_client, scheduler
    
    logger.info("Starting Orchestrator API...")
    
    # Initialize observability
    observability = get_observability_manager("orchestrator-api")
    instrument_fastapi_app(app, "orchestrator-api")
    
    # Initialize A2A client
    a2a_client = OrchestratorA2AClient()
    
    # Initialize scheduler
    scheduler = A2AWorkflowScheduler(a2a_client)
    await scheduler.start()
    
    logger.info("Orchestrator API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup application components."""
    global scheduler
    
    logger.info("Shutting down Orchestrator API...")
    
    if scheduler:
        await scheduler.shutdown()
    
    logger.info("Orchestrator API shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    observability = get_observability_manager()
    
    components = {
        "a2a_client": "healthy" if a2a_client else "unhealthy",
        "scheduler": "healthy" if scheduler and scheduler.scheduler.running else "unhealthy",
        "security_manager": "healthy",
        "observability": "healthy"
    }
    
    # Collect basic metrics
    metrics = {}
    if scheduler:
        metrics.update(scheduler.get_scheduler_status())
    
    return HealthResponse(
        status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        components=components,
        metrics=metrics
    )

@app.post("/analysis", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.EXECUTE))
):
    """Start on-demand data analysis."""
    analysis_id = str(uuid.uuid4())
    trace_id = f"analysis-{analysis_id}"
    agent_id = agent_info.get("client_id")
    
    logger.info(f"Starting analysis {analysis_id} for agent {agent_id}")
    
    # Create analysis status tracker
    status = AnalysisStatus(
        analysis_id=analysis_id,
        status="queued",
        progress=0.0,
        current_step="initializing",
        started_at=datetime.utcnow()
    )
    analysis_tracker[analysis_id] = status
    
    # Start analysis in background
    background_tasks.add_task(
        execute_analysis,
        analysis_id,
        request,
        trace_id,
        agent_id
    )
    
    # Log analysis request
    await security_manager.audit_logger.log_action(
        agent_id=agent_id,
        action="analysis_requested",
        resource=f"analysis:{analysis_id}",
        result="success",
        details={
            "data_source": request.data_source,
            "analysis_type": request.analysis_type,
            "priority": request.priority
        }
    )
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="queued",
        message="Analysis queued for execution",
        estimated_completion=datetime.utcnow() + timedelta(minutes=10),
        trace_id=trace_id
    )

@app.get("/analysis/{analysis_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    analysis_id: str,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.READ))
):
    """Get analysis status and results."""
    if analysis_id not in analysis_tracker:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_tracker[analysis_id]

@app.get("/analysis", response_model=List[AnalysisStatus])
async def list_analyses(
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.READ))
):
    """List all analyses for the authenticated agent."""
    # In a production system, you'd filter by agent
    return list(analysis_tracker.values())

@app.post("/workflows")
async def create_workflow(
    request: WorkflowRequest,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.ADMIN))
):
    """Create a new scheduled workflow."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    agent_id = agent_info.get("client_id")
    
    try:
        # Map string schedule type to enum
        schedule_type = ScheduleType(request.schedule_type)
        
        workflow_id = scheduler.register_workflow(
            name=request.name,
            description=request.description,
            schedule_type=schedule_type,
            schedule_config=request.schedule_config,
            workflow_steps=request.workflow_steps
        )
        
        # Log workflow creation
        await security_manager.audit_logger.log_action(
            agent_id=agent_id,
            action="workflow_created",
            resource=f"workflow:{workflow_id}",
            result="success",
            details={"name": request.name, "schedule_type": request.schedule_type}
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "message": f"Workflow '{request.name}' created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid schedule type: {request.schedule_type}")
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@app.get("/workflows")
async def list_workflows(
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.READ))
):
    """List all workflows."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    return scheduler.list_workflows()

@app.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.READ))
):
    """Get workflow details."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    workflow = scheduler.get_workflow_status(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow

@app.put("/workflows/{workflow_id}/enable")
async def enable_workflow(
    workflow_id: str,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.ADMIN))
):
    """Enable a workflow."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    scheduler.enable_workflow(workflow_id)
    
    # Log workflow enable
    await security_manager.audit_logger.log_action(
        agent_id=agent_info.get("client_id"),
        action="workflow_enabled",
        resource=f"workflow:{workflow_id}",
        result="success"
    )
    
    return {"message": "Workflow enabled"}

@app.put("/workflows/{workflow_id}/disable")
async def disable_workflow(
    workflow_id: str,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.ADMIN))
):
    """Disable a workflow."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    scheduler.disable_workflow(workflow_id)
    
    # Log workflow disable
    await security_manager.audit_logger.log_action(
        agent_id=agent_info.get("client_id"),
        action="workflow_disabled",
        resource=f"workflow:{workflow_id}",
        result="success"
    )
    
    return {"message": "Workflow disabled"}

@app.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.ADMIN))
):
    """Delete a workflow."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    scheduler.remove_workflow(workflow_id)
    
    # Log workflow deletion
    await security_manager.audit_logger.log_action(
        agent_id=agent_info.get("client_id"),
        action="workflow_deleted",
        resource=f"workflow:{workflow_id}",
        result="success"
    )
    
    return {"message": "Workflow deleted"}

@app.get("/metrics")
async def get_metrics(
    agent_info: Dict[str, Any] = Depends(require_permission(PermissionLevel.READ))
):
    """Get system metrics and observability data."""
    observability = get_observability_manager()
    
    # This would return metrics in a production system
    # For now, return basic information
    return {
        "message": "Metrics endpoint - would integrate with Prometheus",
        "observability_enabled": True,
        "tracing_enabled": True
    }

async def execute_analysis(analysis_id: str, request: AnalysisRequest, trace_id: str, agent_id: str):
    """Execute analysis workflow in background."""
    status = analysis_tracker[analysis_id]
    observability = get_observability_manager()
    
    try:
        with observability.trace_a2a_task(analysis_id, "orchestrator", "execute_analysis"):
            status.status = "running"
            status.current_step = "data_loading"
            status.progress = 10.0
            
            # Step 1: Data Loading
            load_response = await a2a_client.execute_skill(
                "data_loader_agent",
                "load_dataset",
                TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    task_type="load_dataset",
                    parameters={"data_source": request.data_source},
                    priority=request.priority
                )
            )
            
            if load_response.status != "completed":
                raise Exception(f"Data loading failed: {load_response.error_message}")
            
            data_handle_id = load_response.results.get("data_handle_id")
            status.progress = 30.0
            status.current_step = "data_cleaning"
            
            # Step 2: Data Cleaning
            clean_response = await a2a_client.execute_skill(
                "data_cleaning_agent",
                "clean_dataset",
                TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    task_type="clean_dataset",
                    parameters={"data_handle_id": data_handle_id},
                    priority=request.priority
                )
            )
            
            if clean_response.status != "completed":
                raise Exception(f"Data cleaning failed: {clean_response.error_message}")
            
            cleaned_handle_id = clean_response.results.get("cleaned_data_handle_id")
            status.progress = 50.0
            status.current_step = "data_enrichment"
            
            # Step 3: Data Enrichment
            enrich_response = await a2a_client.execute_skill(
                "data_enrichment_agent",
                "enrich_dataset",
                TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    task_type="enrich_dataset",
                    parameters={"data_handle_id": cleaned_handle_id},
                    priority=request.priority
                )
            )
            
            if enrich_response.status != "completed":
                raise Exception(f"Data enrichment failed: {enrich_response.error_message}")
            
            enriched_handle_id = enrich_response.results.get("enriched_data_handle_id")
            status.progress = 70.0
            status.current_step = "data_analysis"
            
            # Step 4: Data Analysis
            analysis_response = await a2a_client.execute_skill(
                "data_analyst_agent",
                "analyze_dataset",
                TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    task_type="analyze_dataset",
                    parameters={
                        "data_handle_id": enriched_handle_id,
                        "analysis_type": request.analysis_type
                    },
                    priority=request.priority
                )
            )
            
            if analysis_response.status != "completed":
                raise Exception(f"Data analysis failed: {analysis_response.error_message}")
            
            analysis_handle_id = analysis_response.results.get("analysis_data_handle_id")
            status.progress = 90.0
            status.current_step = "report_generation"
            
            # Step 5: Report Generation
            report_response = await a2a_client.execute_skill(
                "presentation_agent",
                "create_report",
                TaskRequest(
                    task_id=str(uuid.uuid4()),
                    trace_id=trace_id,
                    task_type="create_report",
                    parameters={
                        "data_handle_id": analysis_handle_id,
                        "output_format": request.output_format,
                        "distribution_channels": request.distribution_channels
                    },
                    priority=request.priority
                )
            )
            
            if report_response.status != "completed":
                raise Exception(f"Report generation failed: {report_response.error_message}")
            
            # Complete analysis
            status.status = "completed"
            status.progress = 100.0
            status.current_step = "completed"
            status.completed_at = datetime.utcnow()
            status.execution_time_ms = int((status.completed_at - status.started_at).total_seconds() * 1000)
            status.results = {
                "report_handle_id": report_response.results.get("report_handle_id"),
                "report_path": report_response.results.get("report_path"),
                "visualizations": report_response.results.get("visualizations_created", [])
            }
            
            # Log successful completion
            await security_manager.audit_logger.log_action(
                agent_id=agent_id,
                action="analysis_completed",
                resource=f"analysis:{analysis_id}",
                result="success",
                details={"execution_time_ms": status.execution_time_ms}
            )
            
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        status.status = "failed"
        status.error_message = str(e)
        status.completed_at = datetime.utcnow()
        
        # Log failure
        await security_manager.audit_logger.log_action(
            agent_id=agent_id,
            action="analysis_failed",
            resource=f"analysis:{analysis_id}",
            result="failure",
            details={"error": str(e)}
        ) 