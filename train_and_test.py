"""
Interactive Training & Testing
===============================
1. Enter your custom prompt for what kind of chatbot you want
2. Watch the training happen
3. After training, test the model interactively

Usage: python train_and_test.py
"""

import sys
import yaml
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()


def create_dataset_from_prompt(prompt: str, num_examples: int = 100) -> Path:
    """Generate synthetic training data based on user's prompt - PRODUCTION QUALITY."""
    import ollama
    
    console.print("\n[yellow]📝 Generating PRODUCTION-QUALITY training dataset...[/yellow]")
    console.print(f"[dim]Target: {num_examples} examples (generated in batches)[/dim]")
    
    dataset_path = Path(f"data/datasets/prod_{int(time.time())}.jsonl")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dynamic system prompt that adapts to ANY chatbot type
    system_prompt = f"""You are generating HIGH-QUALITY training data for an AI assistant/chatbot.

USER'S REQUEST: {prompt}

Generate exactly 25 diverse and realistic training examples in JSON format.

INSTRUCTIONS:
1. Analyze what type of chatbot the user wants (persona, assistant, tutor, Q&A, customer service, etc.)
2. Generate training examples that teach the model to behave exactly as requested
3. If it's a PERSONA: use their speech style, vocabulary, mannerisms consistently
4. If it's an ASSISTANT: focus on helpful, accurate, task-oriented responses
5. If it's a TUTOR: make responses educational, patient, step-by-step
6. Cover diverse scenarios: greetings, common questions, edge cases, follow-ups

OUTPUT FORMAT (only valid JSON, no explanation):
[
  {{"instruction": "user message", "input": "", "output": "ideal chatbot response"}},
  ...25 examples total...
]

QUALITY: Responses should be 1-3 sentences, natural, helpful, and on-topic."""

    all_examples = []
    generation_model = "qwen3-next:80b-cloud"
    batches_needed = (num_examples + 24) // 25  # 25 examples per batch
    
    console.print(f"[dim]Using model: {generation_model}[/dim]")
    console.print(f"[dim]Generating {batches_needed} batches of 25 examples each...[/dim]")
    
    for batch_num in range(batches_needed):
        try:
            with console.status(f"[bold green]Generating batch {batch_num + 1}/{batches_needed}..."):
                # Vary the prompt slightly for diversity
                batch_prompts = [
                    f"Generate 25 training examples for: {prompt}",
                    f"Create 25 diverse Q&A pairs for a chatbot that is: {prompt}",
                    f"Generate 25 realistic conversation examples for: {prompt}",
                    f"Create 25 varied training samples for: {prompt}",
                ]
                
                response = ollama.generate(
                    model=generation_model,
                    prompt=batch_prompts[batch_num % len(batch_prompts)],
                    system=system_prompt,
                    options={'temperature': 0.8 + (batch_num * 0.05), 'num_predict': 8000}  # Vary temperature for diversity
                )
            
            text = response['response']
            
            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                valid_examples = [ex for ex in data if ex.get('instruction') and ex.get('output')]
                all_examples.extend(valid_examples)
                console.print(f"[green]  ✓ Batch {batch_num + 1}: Got {len(valid_examples)} examples[/green]")
            else:
                console.print(f"[yellow]  ⚠ Batch {batch_num + 1}: Could not parse JSON[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]  ⚠ Batch {batch_num + 1} error: {str(e)[:50]}[/yellow]")
        
        # Stop if we have enough
        if len(all_examples) >= num_examples:
            break
    
    # If we got very few examples, add fallback
    if len(all_examples) < 10:
        console.print("[yellow]Adding fallback examples...[/yellow]")
        fallback = [
            {"instruction": "Hello", "input": "", "output": f"Hello! I'm here to help you with {prompt}. How can I assist you today?"},
            {"instruction": "What can you do?", "input": "", "output": f"I'm specialized in {prompt}. I can answer questions, provide guidance, and help you with related topics."},
            {"instruction": "Tell me about yourself", "input": "", "output": f"I'm an AI assistant focused on {prompt}. My goal is to be helpful, accurate, and friendly."},
            {"instruction": "How does this work?", "input": "", "output": "Just ask me any question related to my specialty, and I'll do my best to help you!"},
            {"instruction": "Can you help me?", "input": "", "output": "Absolutely! That's what I'm here for. What would you like help with?"},
            {"instruction": "Thank you", "input": "", "output": "You're welcome! Feel free to ask if you have any more questions."},
        ]
        all_examples.extend(fallback)
    
    # Deduplicate and limit
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = ex['instruction'].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)
    
    examples = unique_examples[:num_examples]
    
    # Write dataset
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    console.print(f"\n[bold green]✓ Created dataset with {len(examples)} examples[/bold green]")
    console.print(f"[dim]Saved to: {dataset_path}[/dim]")
    
    return dataset_path


def run_training(prompt: str, config: dict) -> tuple:
    """Run training and return adapter/model paths."""
    from infrastructure.trainer_service import PEFTTrainer
    
    # Create dataset
    dataset_path = create_dataset_from_prompt(prompt)
    
    # Create job ID
    job_id = f"interactive_{int(time.time())}"
    
    console.print(f"\n[bold green]🚂 Starting Training[/bold green]")
    console.print(f"[dim]Job ID: {job_id}[/dim]")
    console.print(f"[dim]Model: {config['models']['base_model']}[/dim]")
    console.print(f"[dim]Training steps: 300 (production quality)[/dim]")
    
    # Train
    trainer = PEFTTrainer(config, job_id=job_id)
    result = trainer.run_training(
        dataset_path=str(dataset_path),
        hyperparameters={'max_steps': 300}  # Production quality training
    )
    
    if result.success:
        console.print(Panel.fit(
            f"[bold green]✅ Training Complete![/bold green]\n\n"
            f"Loss: {result.metrics.get('train_loss', 0):.4f}\n"
            f"Adapter: {result.adapter_path}",
            title="🎉 Success",
            border_style="green"
        ))
        return result.adapter_path, result.model_path
    else:
        console.print(f"[bold red]❌ Training failed: {result.error}[/bold red]")
        return None, None


def test_model_interactive(adapter_path: str, config: dict):
    """Run interactive testing session with the trained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    console.print("\n[bold cyan]🤖 Loading Fine-tuned Model for Testing...[/bold cyan]")
    
    # Load base model
    base_model = config['models']['base_model']
    
    with console.status("[bold green]Loading model..."):
        # Determine dtype
        if torch.cuda.is_available():
            compute_dtype = torch.float16
            device = "cuda"
        else:
            compute_dtype = torch.float32
            device = "cpu"
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=compute_dtype,
            device_map={"": 0} if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    
    console.print("[green]✓ Model loaded![/green]")
    console.print("\n" + "="*60)
    console.print("[bold]Interactive Testing Mode[/bold]")
    console.print("[dim]Type your questions to test the model[/dim]")
    console.print("[dim]Type 'quit' or 'exit' to stop[/dim]")
    console.print("="*60 + "\n")
    
    while True:
        # Get user input
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("[yellow]Exiting test mode...[/yellow]")
            break
        
        if not user_input.strip():
            continue
        
        # Format prompt (Alpaca style)
        prompt_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_input}

### Response:
"""
        
        # Generate response
        with console.status("[bold green]Generating response..."):
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,  # Shorter responses to prevent rambling
                    do_sample=True,
                    temperature=0.6,  # Lower for more coherent output
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,  # Prevent repeating phrases
                    no_repeat_ngram_size=3,  # Don't repeat 3-grams
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the response part
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            # Clean up any trailing incomplete sentences
            if response and response[-1] not in '.!?':
                last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_punct > 0:
                    response = response[:last_punct+1]
        
        console.print(f"[bold green]Bot[/bold green]: {response}\n")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[green]✓ Model unloaded, GPU memory freed[/green]")


def main():
    console.print(Panel.fit(
        "[bold magenta]🎓 Interactive Training & Testing[/bold magenta]\n\n"
        "[dim]1. Enter what kind of chatbot you want to create[/dim]\n"
        "[dim]2. Watch the training run on GPU[/dim]\n"
        "[dim]3. Test your fine-tuned model interactively[/dim]",
        border_style="magenta"
    ))
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"\n[dim]Base Model: {config['models']['base_model']}[/dim]")
    
    # Get user prompt
    console.print("\n[bold]What kind of chatbot do you want to create?[/bold]")
    console.print("[dim]Example: 'A helpful cooking assistant that suggests healthy recipes'[/dim]")
    console.print("[dim]Example: 'A Python coding tutor for beginners'[/dim]")
    console.print("[dim]Example: 'A meditation and mindfulness guide'[/dim]\n")
    
    prompt = Prompt.ask("[bold cyan]Your prompt[/bold cyan]")
    
    if not prompt.strip():
        console.print("[red]Empty prompt. Exiting.[/red]")
        return 1
    
    console.print(f"\n[bold]Creating chatbot for:[/bold] {prompt}")
    
    # Run training
    adapter_path, model_path = run_training(prompt, config)
    
    if adapter_path is None:
        console.print("[red]Training failed. Cannot proceed to testing.[/red]")
        return 1
    
    # Ask if user wants to test
    console.print("\n" + "="*60)
    choice = Prompt.ask(
        "[bold yellow]Training complete! Test the model?[/bold yellow]",
        choices=["y", "c"],
        default="y"
    )
    
    if choice == "y":
        test_model_interactive(adapter_path, config)
    else:
        console.print("[dim]Skipping testing. Your trained model is saved at:[/dim]")
        console.print(f"  Adapter: {adapter_path}")
        console.print(f"  Model: {model_path}")
    
    console.print("\n[bold green]✓ Done![/bold green]")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        exit_code = 1
    sys.exit(exit_code)
