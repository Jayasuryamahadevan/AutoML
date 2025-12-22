"""
Command-line interface for AutoSLM.
"""

import click
import yaml
import uuid
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
def cli():
    """AutoSLM - Agentic SLM Training System"""
    pass


@cli.command()
@click.argument('user_request')
@click.option('--config', '-c', help='Path to domain config YAML')
@click.option('--profile', '-p', default='balanced', help='Training profile: fast, balanced, quality')
@click.option('--max-iterations', '-m', default=3, help='Maximum training iterations')
@click.option('--job-id', help='Custom job ID (auto-generated if not specified)')
def train(user_request, config, profile, max_iterations, job_id):
    """
    Start a new training job.
    
    Example:
        autoslm train "Create a yoga training chatbot" --config config/model_configs/yoga_chatbot.yaml
    """
    from orchestration.workflow import run_workflow
    
    # Generate job ID if not provided
    if not job_id:
        job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    console.print(Panel.fit(
        f"[bold]Request:[/bold] {user_request}\n"
        f"[bold]Job ID:[/bold] {job_id}\n"
        f"[bold]Profile:[/bold] {profile}\n"
        f"[bold]Max Iterations:[/bold] {max_iterations}",
        title="🚀 Starting Training Job",
        border_style="blue"
    ))
    
    # Run workflow
    try:
        result = run_workflow(
            job_id=job_id,
            user_request=user_request,
            domain_config_path=config,
            max_iterations=max_iterations
        )
        
        if result.get('status') == 'deployed':
            console.print(Panel.fit(
                f"[bold green]✓ Training Successful![/bold green]\n\n"
                f"[bold]Final Score:[/bold] {result['deployment_info']['final_score']:.2f}/10\n"
                f"[bold]Iterations:[/bold] {result['deployment_info']['iterations']}\n"
                f"[bold]Model Path:[/bold] {result['deployment_info']['adapter_path']}",
                title="🎉 Success",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[bold yellow]⚠ Training completed with status: {result.get('status')}[/bold yellow]\n"
                f"See outputs/reports/ for details",
                title="⚠ Warning",
                border_style="yellow"
            ))
    
    except Exception as e:
        console.print(f"[bold red]✗ Training failed:[/bold red] {e}")


@cli.command()
@click.argument('user_request')
@click.option('--size', '-s', default=100, type=int, help='Number of examples to generate')
@click.option('--output', '-o', help='Output path (auto-generated if not specified)')
def generate_data(user_request, size, output):
    """
    Generate a synthetic dataset.
    
    Example:
        autoslm generate-data "yoga training" --size 100
    """
    import yaml
    from agents.dataset_generator import DatasetGenerator
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override dataset size
    config['dataset']['max_examples'] = size
    
    # Create generator
    generator = DatasetGenerator(config)
    
    console.print(f"[bold blue]Generating dataset for:[/bold blue] {user_request}")
    
    # Generate
    dataset = generator.generate_from_request(user_request)
    
    # Save
    if not output:
        output = f"data/datasets/{uuid.uuid4().hex[:8]}_data.jsonl"
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_jsonl(output)
    
    console.print(f"[bold green]✓ Dataset saved to {output}[/bold green]")
    console.print(f"Total examples: {dataset.total_examples}")


@cli.command()
@click.argument('report_path')
def show_report(report_path):
    """
    Display an evaluation report.
    
    Example:
        autoslm show-report outputs/reports/yoga_bot_12345_iter1_report.json
    """
    import json
    
    if not Path(report_path).exists():
        console.print(f"[red]Report not found: {report_path}[/red]")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Summary
    console.print(Panel.fit(
        f"[bold]Average Score:[/bold] {report['average_score']:.2f}/10\n"
        f"[bold]Pass Rate:[/bold] {report['pass_rate']:.1%}\n"
        f"[bold]Overall:[/bold] {'✓ PASS' if report['passed'] else '✗ FAIL'}",
        title="📊 Evaluation Summary",
        border_style="blue"
    ))
    
    # Detailed table
    table = Table(title="Test Case Results", show_header=True)
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Instruction", style="white", width=50)
    table.add_column("Score", justify="center", style="yellow", width=8)
    table.add_column("Verdict", justify="center", width=10)
    
    for i, case in enumerate(report['test_cases'][:10], 1):
        instruction_short = case['instruction'][:47] + "..." if len(case['instruction']) > 50 else case['instruction']
        
        verdict_style = "green" if case['verdict'] == "PASS" else "red"
        verdict_icon = "✓" if case['verdict'] == "PASS" else "✗"
        
        table.add_row(
            str(i),
            instruction_short,
            f"{case['overall_score']:.1f}/10",
            f"[{verdict_style}]{verdict_icon} {case['verdict']}[/{verdict_style}]"
        )
    
    if len(report['test_cases']) > 10:
        table.add_row("...", "...", "...", "...")
    
    console.print(table)


@cli.command()
def list_models():
    """List all trained models."""
    models_dir = Path('outputs/adapters')
    
    if not models_dir.exists():
        console.print("[yellow]No models found[/yellow]")
        return
    
    table = Table(title="Trained Models", show_header=True)
    table.add_column("Job ID", style="cyan")
    table.add_column("Path", style="white")
    table.add_column("Created", style="dim")
    
    for model_path in sorted(models_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if model_path.is_dir():
            from datetime import datetime
            created = datetime.fromtimestamp(model_path.stat().st_mtime)
            
            table.add_row(
                model_path.name,
                str(model_path),
                created.strftime("%Y-%m-%d %H:%M")
            )
    
    console.print(table)


@cli.command()
@click.argument('adapter_path')
@click.argument('prompt')
def test_model(adapter_path, prompt):
    """
    Test a trained model with a prompt.
    
    Example:
        autoslm test-model outputs/adapters/yoga_bot_12345_iter1 "What is Downward Dog?"
    """
    import yaml
    from infrastructure.inference_service import InferenceEngine
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inference engine
    engine = InferenceEngine(config)
    
    console.print(f"\n[bold blue]Testing model:[/bold blue] {adapter_path}")
    console.print(f"[bold blue]Prompt:[/bold blue] {prompt}\n")
    
    # Run inference
    responses = engine.run_inference_adapter(
        adapter_path=adapter_path,
        test_cases=[prompt]
    )
    
    if responses:
        console.print(Panel(
            responses[0],
            title="Model Response",
            border_style="green"
        ))
    else:
        console.print("[red]Inference failed[/red]")


if __name__ == '__main__':
    cli()
