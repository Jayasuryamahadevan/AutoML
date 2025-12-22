"""
Direct Training Test - Bypasses dataset generation to test pure training pipeline.
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

console = Console()

console.print(Panel.fit(
    "[bold green]🚂 Direct Training Pipeline Test[/bold green]\n"
    "[dim]Using existing dataset, testing PEFT + QLoRA training[/dim]",
    border_style="green"
))

# Load config
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use minimal steps for quick test
config['training']['max_steps'] = 5

# Check for existing dataset
dataset_path = Path("data/datasets/test_yoga_small.jsonl")
if not dataset_path.exists():
    console.print(f"[red]Dataset not found: {dataset_path}[/red]")
    sys.exit(1)

console.print(f"[green]✓ Using dataset: {dataset_path}[/green]")

# Count examples
with open(dataset_path, 'r') as f:
    num_examples = sum(1 for _ in f)
console.print(f"[dim]Dataset has {num_examples} examples[/dim]")

# Run training
console.print("\n[yellow]Starting training (5 steps)...[/yellow]\n")

from infrastructure.trainer_service import PEFTTrainer

trainer = PEFTTrainer(config, job_id="direct_test")

result = trainer.run_training(
    dataset_path=str(dataset_path),
    hyperparameters={'max_steps': 5}
)

if result.success:
    console.print(Panel.fit(
        f"[bold green]✅ Training Successful![/bold green]\n\n"
        f"Loss: {result.metrics.get('train_loss', 'N/A'):.4f}\n"
        f"Adapter: {result.adapter_path}\n"
        f"Model: {result.model_path}",
        title="🎉 Success",
        border_style="green"
    ))
else:
    console.print(Panel.fit(
        f"[bold red]❌ Training Failed[/bold red]\n\n"
        f"Error: {result.error}",
        title="Error",
        border_style="red"
    ))
    sys.exit(1)
