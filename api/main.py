"""
FastAPI gateway for job submission and monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

# Job storage (in production, use Redis or database)
jobs_db: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="AutoSLM API",
    description="Agentic SLM Training System - Production API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CreateJobRequest(BaseModel):
    """Request to create a new training job."""
    user_request: str = Field(..., description="Natural language description of desired model")
    domain_config_path: Optional[str] = Field(None, description="Path to domain config YAML")
    base_model: Optional[str] = Field("unsloth/Llama-3.2-3B-Instruct", description="Base model to fine-tune")
    quality_preset: Optional[str] = Field("balanced", description="Training profile: fast, balanced, quality")
    max_iterations: Optional[int] = Field(3, description="Maximum training iterations")


class JobResponse(BaseModel):
    """Response with job information."""
    job_id: str
    status: str
    user_request: str
    created_at: str
    updated_at: str
    current_iteration: Optional[int] = None
    evaluation_score: Optional[float] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    status: str
    user_request: str
    created_at: str
    updated_at: str
    iteration: int
    max_iterations: int
    evaluation_score: Optional[float] = None
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    report_path: Optional[str] = None
    error_log: List[str] = []


# Background task runner
def run_training_workflow(job_id: str, request: CreateJobRequest):
    """Run training workflow in background."""
    from orchestration.workflow import run_workflow
    
    try:
        # Update job status
        jobs_db[job_id]["status"] = "running"
        jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
        
        # Run workflow
        result = run_workflow(
            job_id=job_id,
            user_request=request.user_request,
            domain_config_path=request.domain_config_path,
            max_iterations=request.max_iterations
        )
        
        # Update job with results
        jobs_db[job_id].update({
            "status": result.get("status", "unknown"),
            "iteration": result.get("iteration", 0),
            "evaluation_score": result.get("evaluation_score"),
            "model_path": result.get("model_path"),
            "adapter_path": result.get("adapter_path"),
            "report_path": result.get("report_path"),
            "deployment_info": result.get("deployment_info"),
            "error_log": result.get("error_log", []),
            "updated_at": datetime.now().isoformat()
        })
        
        console.print(f"[green]✓ Job {job_id} completed with status: {result.get('status')}[/green]")
    
    except Exception as e:
        console.print(f"[red]✗ Job {job_id} failed: {e}[/red]")
        jobs_db[job_id].update({
            "status": "error",
            "error": str(e),
            "updated_at": datetime.now().isoformat()
        })


# API Endpoints
@app.post("/jobs/create", response_model=JobResponse)
async def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks):
    """
    Create a new training job.
    
    The job will run in the background. Use /jobs/{job_id}/status to monitor progress.
    """
    # Generate job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    # Create job record
    job_data = {
        "job_id": job_id,
        "user_request": request.user_request,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "iteration": 0,
        "max_iterations": request.max_iterations,
        "config": request.dict()
    }
    
    jobs_db[job_id] = job_data
    
    # Start background task
    background_tasks.add_task(run_training_workflow, job_id, request)
    
    console.print(f"[blue]📋 Created job {job_id}[/blue]")
    
    return JobResponse(**job_data)


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the current status of a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_db[job_id]
    
    return JobStatusResponse(
        job_id=job_data["job_id"],
        status=job_data["status"],
        user_request=job_data["user_request"],
        created_at=job_data["created_at"],
        updated_at=job_data["updated_at"],
        iteration=job_data.get("iteration", 0),
        max_iterations=job_data.get("max_iterations", 3),
        evaluation_score=job_data.get("evaluation_score"),
        model_path=job_data.get("adapter_path"),  # Return adapter path as main model
        adapter_path=job_data.get("adapter_path"),
        report_path=job_data.get("report_path"),
        error_log=job_data.get("error_log", [])
    )


@app.get("/jobs/{job_id}/report")
async def get_evaluation_report(job_id: str):
    """Get the evaluation report for a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_db[job_id]
    report_path = job_data.get("report_path")
    
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    return report


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all jobs."""
    return [
        JobResponse(
            job_id=job["job_id"],
            status=job["status"],
            user_request=job["user_request"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            current_iteration=job.get("iteration"),
            evaluation_score=job.get("evaluation_score")
        )
        for job in jobs_db.values()
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "total_jobs": len(jobs_db)
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AutoSLM API",
        "version": "1.0.0",
        "description": "Agentic SLM Training System",
        "endpoints": {
            "create_job": "POST /jobs/create",
            "get_status": "GET /jobs/{job_id}/status",
            "get_report": "GET /jobs/{job_id}/report",
            "list_jobs": "GET /jobs",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    console.print("\n[bold blue]🚀 Starting AutoSLM API Server[/bold blue]")
    console.print("[dim]API will be available at http://localhost:8000[/dim]")
    console.print("[dim]Docs available at http://localhost:8000/docs[/dim]\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
