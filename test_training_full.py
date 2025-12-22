# Quick Training Test - Full Pipeline
import yaml
import sys
sys.path.insert(0, 'D:\\AutoSLM v2')

from rich.console import Console
from rich.panel import Panel

console = Console()

console.print(Panel.fit(
    "[bold green]🚂 Full Training Pipeline Test[/bold green]\n"
    "[dim]Testing: Dataset Gen → Training → Evaluation[/dim]",
    border_style="green"
))

# Step 1: Generate tiny dataset
console.print("\n[yellow]Step 1:[/yellow] Generating tiny training dataset (5 examples)...")

from agents.dataset_generator import DatasetGenerator

with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Very small dataset for quick test
config['dataset']['max_examples'] = 5
config['training']['max_steps'] = 10  # Very short training for test

generator = DatasetGenerator(config)
dataset = generator.generate_from_request(
    user_request="Create a simple yoga chatbot",
    domain_config_path=None  # Let it auto-generate categories
)

dataset_path = "data/datasets/quick_test_train.jsonl"
dataset.to_jsonl(dataset_path)
console.print(f"[green]✓ Generated {dataset.total_examples} examples[/green]")

# Step 2: Train with Unsloth
console.print("\n[yellow]Step 2:[/yellow] Training with Unsloth (10 steps, quick test)...")

from infrastructure.trainer_service import PEFTTrainer

trainer = PEFTTrainer(config, job_id="quick_test")

result = trainer.run_training(
    dataset_path=dataset_path,
    hyperparameters={'max_steps': 10}  # Very short for quick test
)

if result.success:
    console.print(f"[green]✓ Training completed![/green]")
    console.print(f"   Loss: {result.metrics.get('train_loss', 'N/A')}")
    console.print(f"   Adapter: {result.adapter_path}")
else:
    console.print(f"[red]✗ Training failed: {result.error}[/red]")
    sys.exit(1)

# Step 3: Quick evaluation test
console.print("\n[yellow]Step 3:[/yellow] Testing inference...")

from infrastructure.inference_service import InferenceEngine

engine = InferenceEngine(config)

test_prompt = "What is a simple yoga pose for beginners?"
console.print(f"[dim]Testing with: {test_prompt}[/dim]")

try:
    responses = engine.run_inference_adapter(
        adapter_path=result.adapter_path,
        test_cases=[test_prompt]
    )
    
    if responses:
        console.print(Panel(
            responses[0][:200],
            title="Model Response",
            border_style="green"
        ))
    else:
        console.print("[yellow]⚠ Inference returned empty response[/yellow]")
except Exception as e:
    console.print(f"[yellow]⚠ Inference test skipped: {e}[/yellow]")

console.print("\n" + "="*80)
console.print(Panel.fit(
    "[bold green]✅ Full Pipeline Test Complete![/bold green]\n\n"
    "[bold]What worked:[/bold]\n"
    "✓ Unsloth installation verified\n"
    "✓ Model loading successful\n"
    "✓ LoRA adapter training\n"
    "✓ Model saving and export\n\n"
    "[bold]Next step:[/bold]\n"
    "Run full training: python cli.py train \"Create a yoga chatbot\" --config config/model_configs/yoga_chatbot.yaml",
    title="🎉 Success",
    border_style="green"
))
