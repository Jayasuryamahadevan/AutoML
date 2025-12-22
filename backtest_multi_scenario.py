"""
Multi-Scenario Training Backtest
=================================
Real training tests across 5+ different prompts/scenarios.
No simulations - actual GPU training execution.
"""

import sys
import yaml
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class ScenarioResult:
    name: str
    prompt: str
    success: bool
    loss: float = 0.0
    duration_seconds: float = 0.0
    adapter_path: str = ""
    error: str = ""


# ============================================================================
# TEST SCENARIOS - Different domains and use cases
# ============================================================================
SCENARIOS = [
    {
        "name": "Yoga Instructor",
        "prompt": "Create a yoga instructor chatbot that teaches poses and breathing",
        "dataset": "data/datasets/test_yoga_small.jsonl",
        "steps": 3
    },
    {
        "name": "Customer Support",
        "prompt": "Create a customer support bot for an e-commerce store",
        "dataset": None,  # Will generate
        "steps": 3
    },
    {
        "name": "Coding Assistant",
        "prompt": "Create a Python coding assistant for beginners",
        "dataset": None,
        "steps": 3
    },
    {
        "name": "Recipe Helper",
        "prompt": "Create a cooking recipe chatbot that suggests healthy meals",
        "dataset": None,
        "steps": 3
    },
    {
        "name": "Fitness Coach",
        "prompt": "Create a personal fitness trainer chatbot",
        "dataset": None,
        "steps": 3
    }
]


def create_synthetic_dataset(prompt: str, num_examples: int = 10) -> Path:
    """Create synthetic dataset using Ollama."""
    import ollama
    
    dataset_path = Path(f"data/datasets/backtest_{int(time.time())}.jsonl")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract domain from prompt
    domain = prompt.split("Create a ")[-1].split(" chatbot")[0] if "chatbot" in prompt else "assistant"
    
    # Generate examples
    examples = []
    
    system_prompt = f"""You are generating training data for a {domain} AI assistant.
Generate exactly 5 training examples in this JSON format:
[
  {{"instruction": "user question", "input": "", "output": "helpful response"}},
  ...
]
Only output valid JSON, no explanation."""

    try:
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"Generate 5 training examples for: {prompt}",
            system=system_prompt,
            options={'temperature': 0.8}
        )
        
        text = response['response']
        # Try to extract JSON
        import re
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            examples = data[:num_examples]
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate dataset: {e}[/yellow]")
        # Create minimal fallback dataset
        examples = [
            {"instruction": f"What is {domain}?", "input": "", "output": f"{domain.capitalize()} is a helpful service."},
            {"instruction": f"How does {domain} work?", "input": "", "output": f"It works by understanding your needs."},
            {"instruction": f"Tell me more about {domain}", "input": "", "output": f"Here's more information about {domain}."},
        ]
    
    # Write dataset
    with open(dataset_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    return dataset_path


def run_training_scenario(scenario: dict, config: dict) -> ScenarioResult:
    """Run a single training scenario."""
    from infrastructure.trainer_service import PEFTTrainer
    
    name = scenario["name"]
    prompt = scenario["prompt"]
    steps = scenario["steps"]
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]SCENARIO: {name}[/bold]")
    console.print(f"[dim]Prompt: {prompt[:60]}...[/dim]")
    console.print(f"[cyan]{'='*60}[/cyan]")
    
    start_time = time.time()
    
    try:
        # Get or create dataset
        if scenario["dataset"]:
            dataset_path = Path(scenario["dataset"])
            if not dataset_path.exists():
                console.print(f"[yellow]Dataset not found, generating...[/yellow]")
                dataset_path = create_synthetic_dataset(prompt)
        else:
            console.print(f"[dim]Generating synthetic dataset...[/dim]")
            dataset_path = create_synthetic_dataset(prompt)
        
        # Count examples
        with open(dataset_path, 'r') as f:
            num_examples = sum(1 for _ in f)
        console.print(f"[dim]Dataset: {dataset_path} ({num_examples} examples)[/dim]")
        
        if num_examples == 0:
            raise ValueError("Empty dataset")
        
        # Create unique job id
        job_id = f"backtest_{name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Run training
        trainer = PEFTTrainer(config, job_id=job_id)
        result = trainer.run_training(
            dataset_path=str(dataset_path),
            hyperparameters={'max_steps': steps}
        )
        
        duration = time.time() - start_time
        
        if result.success:
            console.print(f"[bold green]✓ PASSED - Loss: {result.metrics.get('train_loss', 0):.4f}[/bold green]")
            return ScenarioResult(
                name=name,
                prompt=prompt,
                success=True,
                loss=result.metrics.get('train_loss', 0),
                duration_seconds=duration,
                adapter_path=result.adapter_path
            )
        else:
            console.print(f"[bold red]✗ FAILED - {result.error}[/bold red]")
            return ScenarioResult(
                name=name,
                prompt=prompt,
                success=False,
                duration_seconds=duration,
                error=result.error
            )
            
    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[bold red]✗ ERROR - {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        return ScenarioResult(
            name=name,
            prompt=prompt,
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def main():
    console.print(Panel.fit(
        "[bold green]🧪 Multi-Scenario Training Backtest[/bold green]\n"
        "[dim]Testing training pipeline with 5 different prompts[/dim]\n"
        "[bold yellow]No simulations - Real GPU training[/bold yellow]",
        border_style="green"
    ))
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"\n[dim]Model: {config['models']['base_model']}[/dim]")
    console.print(f"[dim]4-bit: {config['models']['load_in_4bit']}[/dim]")
    
    # Run all scenarios
    results: List[ScenarioResult] = []
    
    for i, scenario in enumerate(SCENARIOS, 1):
        console.print(f"\n[bold]Test {i}/{len(SCENARIOS)}[/bold]")
        result = run_training_scenario(scenario, config)
        results.append(result)
    
    # Summary
    console.print("\n" + "="*70)
    console.print("[bold]BACKTEST SUMMARY[/bold]")
    console.print("="*70)
    
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Loss", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Error", style="red", max_width=30)
    
    for r in results:
        status = "[green]✓ PASS[/green]" if r.success else "[red]✗ FAIL[/red]"
        loss = f"{r.loss:.4f}" if r.success else "-"
        time_str = f"{r.duration_seconds:.1f}s"
        error = r.error[:30] if r.error else ""
        table.add_row(r.name, status, loss, time_str, error)
    
    console.print(table)
    
    console.print(f"\n[bold]Results: {passed}/{len(results)} passed ({passed/len(results)*100:.0f}%)[/bold]")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "model": config['models']['base_model'],
        "results": [
            {
                "name": r.name,
                "prompt": r.prompt,
                "success": r.success,
                "loss": r.loss,
                "duration_seconds": r.duration_seconds,
                "adapter_path": r.adapter_path,
                "error": r.error
            }
            for r in results
        ]
    }
    
    report_path = Path("outputs/reports/multi_scenario_backtest.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    console.print(f"\n[dim]Report saved: {report_path}[/dim]")
    
    # Exit code
    if failed > 0:
        console.print(f"\n[bold red]⚠ {failed} scenario(s) failed[/bold red]")
        return 1
    else:
        console.print(f"\n[bold green]✓ All scenarios passed![/bold green]")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
