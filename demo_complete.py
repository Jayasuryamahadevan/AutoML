# Complete Demo - Without Training (shows data generation and evaluation)
import yaml
import json
from pathlib import Path
import sys
sys.path.insert(0, 'D:\\AutoSLM v2')

from agents.dataset_generator import DatasetGenerator
from agents.critic import CriticAgent
from agents.schemas import TrainingExample
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

console.print(Panel.fit(
    "[bold cyan]AutoSLM v2 - Complete Demo[/bold cyan]\n"
    "[dim]Dataset Generation + Evaluation (No Training Required)[/dim]",
    border_style="cyan"
))

# Load config
console.print("\n[yellow]Step 1:[/yellow] Loading configuration...")
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Reduce dataset size for demo
config['dataset']['max_examples'] = 20

# Step 1: Generate Dataset
console.print("\n[yellow]Step 2:[/yellow] Generating yoga training dataset (20 examples)...")
generator = DatasetGenerator(config)
dataset = generator.generate_from_request(
    user_request="Create a yoga training chatbot",
    domain_config_path="config/model_configs/yoga_chatbot.yaml"
)

console.print(f"[green]✓ Generated {dataset.total_examples} training examples[/green]")

# Show samples
console.print("\n[yellow]Sample Training Examples:[/yellow]")
table = Table(show_header=True)
table.add_column("ID", style="cyan", width=5)
table.add_column("Instruction", style="white", width=50)
table.add_column("Response Preview", style="dim", width=40)

for i, ex in enumerate(dataset.examples[:5], 1):
    table.add_row(
        str(i),
        ex.instruction[:47] + "..." if len(ex.instruction) > 50 else ex.instruction,
        ex.response[:37] + "..." if len(ex.response) > 40 else ex.response
    )

console.print(table)

# Save dataset
output_path = "data/datasets/demo_yoga.jsonl"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
dataset.to_jsonl(output_path)
console.print(f"\n[green]✓ Dataset saved to {output_path}[/green]")

# Step 2: Create Test Set
console.print("\n[yellow]Step 3:[/yellow] Creating test set...")
test_dataset = generator.create_test_set(dataset, test_split=0.2)
console.print(f"[green]✓ Created {test_dataset.total_examples} test cases[/green]")

# Step 3: Simulate Model Responses (in real scenario, this comes from trained model)
console.print("\n[yellow]Step 4:[/yellow] Simulating model responses...")
console.print("[dim](In production, these would come from your fine-tuned model)[/dim]")

# For demo, use Ollama to generate example responses
import ollama

model_responses = []
for i, test_case in enumerate(test_dataset.examples[:3], 1):  # Test first 3
    console.print(f"[dim]Generating response {i}/3...[/dim]")
    
    prompt = f"""You are a helpful yoga instructor. Answer this question:
{test_case.instruction}

Provide a clear, calm, instructional response (2-3 sentences):"""
    
    result = ollama.generate(
        model='llama3.2:3b',
        prompt=prompt,
        options={'temperature': 0.7, 'num_predict': 100}
    )
    
    model_responses.append(result['response'].strip())

console.print(f"[green]✓ Generated {len(model_responses)} model responses[/green]")

# Step 4: Evaluate with Critic
console.print("\n[yellow]Step 5:[/yellow] Evaluating model quality with LLM-as-Judge...")

critic = CriticAgent(config)
report = critic.evaluate_model(
    test_cases=test_dataset.examples[:3],
    model_responses=model_responses,
    domain="yoga instruction"
)

# Save report
report_path = "outputs/reports/demo_evaluation.json"
Path(report_path).parent.mkdir(parents=True, exist_ok=True)
with open(report_path, 'w') as f:
    json.dump(report.dict(), f, indent=2)

console.print(f"\n[green]✓ Evaluation report saved to {report_path}[/green]")

# Step 5: Show Results
console.print("\n" + "="*80)
console.print(Panel.fit(
    f"[bold]Average Score:[/bold] {report.average_score:.2f}/10\n"
    f"[bold]Pass Rate:[/bold] {report.pass_rate:.1%}\n"
    f"[bold]Overall Verdict:[/bold] {'✓ PASS' if report.passed else '✗ FAIL'}\n\n"
    f"[dim]Note: In production, if score < 8.0, the system would automatically:\n"
    f"  1. Adjust hyperparameters (reduce LR, increase epochs)\n"
    f"  2. Retry training (up to 3 iterations)\n"
    f"  3. Deploy when quality threshold is met[/dim]",
    title="🎯 Evaluation Results",
    border_style="green" if report.passed else "red"
))

# Show detailed example
if report.test_cases:
    console.print("\n[yellow]Detailed Example:[/yellow]")
    case = report.test_cases[0]
    
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan]\n{case.instruction}\n\n"
        f"[bold green]Model Response:[/bold green]\n{case.model_response}\n\n"
        f"[bold yellow]Scores:[/bold yellow]",
        border_style="blue"
    ))
    
    score_table = Table(show_header=True)
    score_table.add_column("Dimension", style="cyan")
    score_table.add_column("Score", style="yellow", justify="center")
    score_table.add_column("Reasoning", style="white")
    
    for score in case.scores:
        score_table.add_row(
            score.dimension.title(),
            f"{score.score}/10",
            score.reasoning[:60] + "..." if len(score.reasoning) > 60 else score.reasoning
        )
    
    console.print(score_table)
    
    verdict_style = "green" if case.verdict == "PASS" else "red"
    console.print(f"\n[{verdict_style}]Overall: {case.overall_score:.1f}/10 - {case.verdict}[/{verdict_style}]")

console.print("\n" + "="*80)
console.print(Panel.fit(
    "[bold green]✅ Demo Complete![/bold green]\n\n"
    "[bold]What happened:[/bold]\n"
    "✓ Generated 20 synthetic training examples\n"
    "✓ Created 4 test cases\n"
    "✓ Generated model responses (simulated)\n"
    "✓ Evaluated quality with LLM-as-judge\n"
    "✓ Provided actionable feedback\n\n"
    "[bold]For full training:[/bold]\n"
    "Install Unsloth and run: python cli.py train \"Create a yoga chatbot\"",
    title="🎉 Success",
    border_style="green"
))
