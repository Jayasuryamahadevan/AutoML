"""
LangGraph orchestration nodes - Individual workflow components.
"""

import json
from pathlib import Path
from typing import Dict, Any
from rich.console import Console

console = Console()


def dataset_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic training dataset."""
    import yaml
    from agents.dataset_generator import DatasetGenerator
    
    console.print(f"\n[bold cyan]📊 Node: Dataset Generation[/bold cyan]")
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = DatasetGenerator(config)
    
    # Generate dataset
    domain_config = state.get('domain_config', {}).get('path')
    dataset = generator.generate_from_request(
        user_request=state['user_request'],
        domain_config_path=domain_config
    )
    
    # Split and deduplicate BEFORE writing any training artifacts. The locked
    # test partition is never passed to the generator or trainer.
    from autoslm_v3.data import split_dataset

    split = split_dataset(
        [example.dict() for example in dataset.examples],
        test_fraction=config.get("dataset", {}).get("test_split", 0.1),
        validation_fraction=0.1,
        seed=config.get("training", {}).get("seed", 42),
    )

    dataset_path = Path('data/datasets') / f"{state['job_id']}_train.jsonl"
    validation_path = Path('data/validation_sets') / f"{state['job_id']}_validation.jsonl"
    test_path = Path('data/test_sets') / f"{state['job_id']}_test.json"
    for path in (dataset_path, validation_path, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def write_training_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as stream:
            for example in rows:
                record = {
                    "instruction": example["instruction"],
                    "input": "",
                    "output": example["response"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\\n")

    write_training_jsonl(dataset_path, split.train)
    write_training_jsonl(validation_path, split.validation)

    # Strip internal split hashes from the held-out test artifact.
    test_data = [
        {
            "instruction": example["instruction"],
            "response": example["response"],
            "category": example.get("category"),
            "metadata": example.get("metadata") or {},
        }
        for example in split.test
    ]
    with test_path.open("w", encoding="utf-8") as stream:
        json.dump(test_data, stream, ensure_ascii=False, indent=2)

    manifest_path = Path("data/manifests") / f"{state['job_id']}_dataset.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_fingerprint": split.fingerprint,
        "duplicates_removed": split.duplicates_removed,
        "train_examples": len(split.train),
        "validation_examples": len(split.validation),
        "test_examples": len(split.test),
        "seed": config.get("training", {}).get("seed", 42),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.print(f"[green]✓ Training dataset saved: {dataset_path} ({len(split.train)} examples)[/green]")
    console.print(f"[green]✓ Validation dataset saved: {validation_path} ({len(split.validation)} examples)[/green]")
    console.print(f"[green]✓ Locked test set saved: {test_path} ({len(split.test)} examples)[/green]")

    return {
        "dataset_path": str(dataset_path),
        "validation_set_path": str(validation_path),
        "test_set_path": str(test_path),
        "dataset_fingerprint": split.fingerprint,
        "dataset_manifest_path": str(manifest_path),
        "status": "dataset_ready"
    }


def trainer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute model training."""
    import yaml
    from infrastructure.trainer_service import PEFTTrainer
    
    console.print(f"\n[bold cyan]🚂 Node: Training (Iteration {state['iteration'] + 1})[/bold cyan]")
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    job_id_iter = f"{state['job_id']}_iter{state['iteration'] + 1}"
    trainer = PEFTTrainer(config, job_id=job_id_iter)
    
    # Run training
    result = trainer.run_training(
        dataset_path=state['dataset_path'],
        hyperparameters=state.get('hyperparameters', {})
    )
    
    if result.success:
        console.print(f"[green]✓ Training completed successfully[/green]")
        return {
            "model_path": result.model_path,
            "adapter_path": result.adapter_path,
            "training_metrics": result.metrics,
            "status": "training_complete",
            "iteration": state['iteration'] + 1
        }
    else:
        console.print(f"[red]✗ Training failed: {result.error}[/red]")
        error_log = state.get('error_log', [])
        error_log.append(f"Iteration {state['iteration'] + 1}: {result.error}")
        return {
            "status": "training_failed",
            "error_log": error_log
        }


def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate trained model."""
    import yaml
    import json
    from agents.critic import CriticAgent
    from agents.schemas import TrainingExample
    from infrastructure.inference_service import InferenceEngine, load_gguf_to_ollama
    
    console.print(f"\n[bold cyan]🔍 Node: Evaluation[/bold cyan]")
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test set
    with open(state['test_set_path'], 'r') as f:
        test_data = json.load(f)
    
    test_cases = [TrainingExample(**ex) for ex in test_data]
    
    # Run inference
    inference_engine = InferenceEngine(config)
    
    # Try to use GGUF model via Ollama if available
    if state.get('model_path'):
        model_name = f"autoslm_{state['job_id']}_iter{state['iteration']}"
        
        # Load into Ollama
        if load_gguf_to_ollama(state['model_path'], model_name):
            # Use Ollama for inference
            test_instructions = [tc.instruction for tc in test_cases]
            model_responses = inference_engine.run_inference_ollama(
                model_name=model_name,
                test_cases=test_instructions
            )
        else:
            # Fall back to adapter inference
            model_responses = inference_engine.run_inference_adapter(
                adapter_path=state['adapter_path'],
                test_cases=[tc.instruction for tc in test_cases]
            )
    else:
        # Use adapter directly
        model_responses = inference_engine.run_inference_adapter(
            adapter_path=state['adapter_path'],
            test_cases=[tc.instruction for tc in test_cases]
        )
    
    # Evaluate with critic
    critic = CriticAgent(config)
    report = critic.evaluate_model(
        test_cases=test_cases,
        model_responses=model_responses,
        domain=state.get('domain', 'general')
    )
    
    # Save report
    report_path = Path('outputs/reports') / f"{state['job_id']}_iter{state['iteration']}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report.dict(), f, indent=2)
    
    console.print(f"[green]✓ Evaluation report saved: {report_path}[/green]")
    
    return {
        "evaluation_score": report.average_score,
        "evaluation_report": report.dict(),
        "report_path": str(report_path),
        "status": "evaluation_complete"
    }


def decision_node(state: Dict[str, Any]) -> str:
    """Decide next action based on evaluation results."""
    import yaml
    
    console.print(f"\n[bold cyan]🤔 Node: Decision Engine[/bold cyan]")
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    score = state['evaluation_score']
    iteration = state['iteration']
    max_iterations = state.get('max_iterations', config['orchestration']['max_iterations'])
    threshold = config['evaluation']['pass_threshold']
    
    console.print(f"Score: {score:.2f}/10 | Threshold: {threshold} | Iteration: {iteration}/{max_iterations}")
    
    # Decision logic
    if score >= threshold:
        console.print("[bold green]✓ Model passed evaluation! Deploying...[/bold green]")
        return "deploy"
    
    if iteration >= max_iterations:
        console.print("[bold red]✗ Max iterations reached. Using best model so far.[/bold red]")
        return "max_retries"
    
    console.print("[bold yellow]⟳ Retrying with adjusted hyperparameters...[/bold yellow]")
    return "retry"


def adjustment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust hyperparameters for retry."""
    import yaml
    
    console.print(f"\n[bold cyan]⚙️ Node: Hyperparameter Adjustment[/bold cyan]")
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get current hyperparameters
    hyperparameters = state.get('hyperparameters', {})
    
    # Apply retry strategy
    retry_cfg = config['orchestration']['retry']
    lr_decay = retry_cfg['learning_rate_decay']
    epochs_increment = retry_cfg['epochs_increment']
    
    # Adjust learning rate
    current_lr = hyperparameters.get('learning_rate', config['training']['learning_rate']['default'])
    new_lr = max(
        current_lr * lr_decay,
        config['training']['learning_rate']['min']
    )
    
    # Adjust max steps (proxy for epochs)
    current_steps = hyperparameters.get('max_steps', config['training']['max_steps'])
    new_steps = int(current_steps * (1 + epochs_increment))
    
    hyperparameters['learning_rate'] = new_lr
    hyperparameters['max_steps'] = new_steps
    
    console.print(f"[yellow]Adjusted: LR {current_lr:.2e} → {new_lr:.2e}, Steps {current_steps} → {new_steps}[/yellow]")
    
    return {
        "hyperparameters": hyperparameters,
        "status": "hyperparameters_adjusted"
    }


def deploy_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy the final model."""
    console.print(f"\n[bold green]🚀 Node: Deployment[/bold green]")
    
    # In a real system, this would:
    # - Push to model registry (HuggingFace Hub, S3, etc.)
    # - Create deployment config
    # - Notify monitoring systems
    
    console.print(f"[green]✓ Model deployed successfully![/green]")
    console.print(f"   Adapter: {state['adapter_path']}")
    console.print(f"   Model: {state.get('model_path', 'N/A')}")
    console.print(f"   Score: {state['evaluation_score']:.2f}/10")
    
    return {
        "status": "deployed",
        "deployment_info": {
            "adapter_path": state['adapter_path'],
            "model_path": state.get('model_path'),
            "final_score": state['evaluation_score'],
            "iterations": state['iteration']
        }
    }


def failure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle max retries reached."""
    console.print(f"\n[bold red]❌ Node: Maximum Retries Reached[/bold red]")
    
    console.print(f"[red]Training did not reach target quality after {state['iteration']} iterations.[/red]")
    console.print(f"[yellow]Best score achieved: {state['evaluation_score']:.2f}/10[/yellow]")
    
    # Save best model anyway
    console.print(f"[yellow]Saving best model from iteration {state['iteration']}[/yellow]")
    
    return {
        "status": "failed_max_retries",
        "best_score": state['evaluation_score'],
        "best_model": state.get('adapter_path')
    }
