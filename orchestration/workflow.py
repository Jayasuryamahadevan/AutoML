"""
Main LangGraph workflow - Orchestrates the entire training pipeline.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from rich.console import Console

from orchestration.nodes import (
    dataset_generation_node,
    trainer_node,
    critic_node,
    decision_node,
    adjustment_node,
    deploy_node,
    failure_node
)

console = Console()


class WorkflowState(TypedDict):
    """State maintained throughout the workflow."""
    job_id: str
    user_request: str
    iteration: int
    max_iterations: int
    hyperparameters: dict
    domain_config: dict
    dataset_path: str
    test_set_path: str
    model_path: str
    adapter_path: str
    training_metrics: dict
    evaluation_score: float
    evaluation_report: dict
    report_path: str
    status: str
    error_log: list
    deployment_info: dict


def create_workflow() -> StateGraph:
    """
    Create the LangGraph orchestration workflow.
    
    Flow:
    1. Generate Dataset
    2. Train Model (Iteration 1..N)
    3. Evaluate Model
    4. Decision: Pass → Deploy | Fail + Iter < Max → Adjust → Retry | Fail + Iter >= Max → Max Retries
    """
    
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("generate_dataset", dataset_generation_node)
    workflow.add_node("train", trainer_node)
    workflow.add_node("evaluate", critic_node)
    workflow.add_node("adjust", adjustment_node)
    workflow.add_node("deploy", deploy_node)
    workflow.add_node("max_retries", failure_node)
    
    # Set entry point
    workflow.set_entry_point("generate_dataset")
    
    # Define edges
    workflow.add_edge("generate_dataset", "train")
    workflow.add_edge("train", "evaluate")
    
    # Conditional routing after evaluation
    workflow.add_conditional_edges(
        "evaluate",
        decision_node,
        {
            "deploy": "deploy",
            "retry": "adjust",
            "max_retries": "max_retries"
        }
    )
    
    # Retry loop
    workflow.add_edge("adjust", "train")
    
    # Terminal states
    workflow.add_edge("deploy", END)
    workflow.add_edge("max_retries", END)
    
    return workflow


def run_workflow(
    job_id: str,
    user_request: str,
    domain_config_path: str = None,
    max_iterations: int = 3
) -> dict:
    """
    Run the complete training workflow.
    
    Args:
        job_id: Unique job identifier
        user_request: Natural language description of desired model
        domain_config_path: Optional path to domain config
        max_iterations: Maximum training iterations
        
    Returns:
        Final state dictionary
    """
    import yaml
    
    console.print("\n" + "="*80)
    console.print(f"[bold blue]🚀 Starting Agentic Training Workflow[/bold blue]")
    console.print(f"[bold]Job ID:[/bold] {job_id}")
    console.print(f"[bold]Request:[/bold] {user_request}")
    console.print("="*80)
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize state
    initial_state = {
        "job_id": job_id,
        "user_request": user_request,
        "iteration": 0,
        "max_iterations": max_iterations,
        "hyperparameters": {},
        "domain_config": {"path": domain_config_path} if domain_config_path else {},
        "status": "initialized",
        "error_log": []
    }
    
    # Create and compile workflow
    workflow = create_workflow()
    app = workflow.compile()
    
    # Run workflow
    try:
        final_state = app.invoke(initial_state)
        
        console.print("\n" + "="*80)
        console.print("[bold green]✓ Workflow Complete![/bold green]")
        console.print(f"[bold]Final Status:[/bold] {final_state.get('status')}")
        
        if final_state.get('status') == 'deployed':
            info = final_state.get('deployment_info', {})
            console.print(f"[bold]Final Score:[/bold] {info.get('final_score'):.2f}/10")
            console.print(f"[bold]Iterations:[/bold] {info.get('iterations')}")
            console.print(f"[bold]Model Path:[/bold] {info.get('adapter_path')}")
        
        console.print("="*80 + "\n")
        
        return final_state
    
    except Exception as e:
        console.print(f"\n[bold red]✗ Workflow failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def main():
    """Test the workflow."""
    import uuid
    
    # Generate unique job ID
    job_id = f"yoga_bot_{uuid.uuid4().hex[:8]}"
    
    # Run workflow
    result = run_workflow(
        job_id=job_id,
        user_request="Create a yoga training chatbot that helps with poses, breathing, and sequences",
        domain_config_path="config/model_configs/yoga_chatbot.yaml",
        max_iterations=3
    )
    
    console.print(f"\n[bold]Final result:[/bold]")
    console.print(result)


if __name__ == "__main__":
    main()
