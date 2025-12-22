"""
Inference Service - Run inference on trained models for evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

console = Console()


class InferenceEngine:
    """Run inference using Ollama or directly with PEFT adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def run_inference_ollama(
        self,
        model_name: str,
        test_cases: List[str],
        temperature: float = 0.7
    ) -> List[str]:
        """
        Run inference using Ollama (for GGUF models).
        
        Args:
            model_name: Name of the model in Ollama
            test_cases: List of instructions to test
            temperature: Sampling temperature
            
        Returns:
            List of model responses
        """
        import ollama
        
        console.print(f"\n[bold blue]🔮 Running inference with {model_name}[/bold blue]")
        
        responses = []
        
        for i, instruction in enumerate(test_cases, 1):
            console.print(f"[dim]Test {i}/{len(test_cases)}[/dim]")
            
            # Format prompt
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
            
            # Generate response
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 256
                }
            )
            
            # Extract response (remove prompt if included)
            generated_text = response['response'].strip()
            responses.append(generated_text)
        
        console.print(f"[bold green]✓ Generated {len(responses)} responses[/bold green]")
        
        return responses
    
    def run_inference_adapter(
        self,
        adapter_path: str,
        test_cases: List[str],
        base_model: str = None
    ) -> List[str]:
        """
        Run inference using a PEFT LoRA adapter directly.
        
        Args:
            adapter_path: Path to adapter directory
            test_cases: List of instructions
            base_model: Base model to load (uses config default if not specified)
            
        Returns:
            List of model responses
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            
            if base_model is None:
                base_model = self.config['models']['base_model']
            
            console.print(f"\n[bold blue]🔮 Running inference with adapter[/bold blue]")
            console.print(f"[dim]Base model: {base_model}[/dim]")
            console.print(f"[dim]Adapter: {adapter_path}[/dim]")
            
            # Load base model
            console.print("[dim]Loading base model...[/dim]")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load PEFT adapter
            console.print("[dim]Loading PEFT adapter...[/dim]")
            model = PeftModel.from_pretrained(model, adapter_path)
            
            # Set to eval mode
            model.eval()
            
            responses = []
            
            for i, instruction in enumerate(test_cases, 1):
                console.print(f"[dim]Test {i}/{len(test_cases)}[/dim]")
                
                # Format prompt
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        use_cache=True
                    )
                
                # Decode
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract response part
                response_text = generated.split("### Response:")[-1].strip()
                responses.append(response_text)
            
            console.print(f"[bold green]✓ Generated {len(responses)} responses[/bold green]")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return responses
        
        except ImportError as e:
            console.print(f"[red]Required packages not available: {e}[/red]")
            console.print("[yellow]Falling back to Ollama (if available)[/yellow]")
            return []
        except Exception as e:
            console.print(f"[red]Inference failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return []


def load_gguf_to_ollama(gguf_path: str, model_name: str):
    """
    Load a GGUF file into Ollama for inference.
    
    Args:
        gguf_path: Path to GGUF file
        model_name: Name to give the model in Ollama
    """
    import subprocess
    
    console.print(f"\n[bold blue]📦 Loading {gguf_path} into Ollama as '{model_name}'[/bold blue]")
    
    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
"""
    
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    
    # Run ollama create
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print(f"[bold green]✓ Model loaded as '{model_name}'[/bold green]")
            return True
        else:
            console.print(f"[red]Failed to load model: {result.stderr}[/red]")
            return False
    
    except FileNotFoundError:
        console.print("[red]Ollama not found. Make sure it's installed and in PATH.[/red]")
        return False


def main():
    """Test inference."""
    import yaml
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inference engine
    engine = InferenceEngine(config)
    
    # Test cases
    test_cases = [
        "What are the steps to perform Downward Facing Dog?",
        "Suggest a morning yoga routine for beginners"
    ]
    
    # Test with Ollama (assuming llama3.2:3b is available)
    responses = engine.run_inference_ollama("llama3.2:3b", test_cases)
    
    for q, a in zip(test_cases, responses):
        console.print(f"\n[bold cyan]Q:[/bold cyan] {q}")
        console.print(f"[bold green]A:[/bold green] {a}")


if __name__ == "__main__":
    main()
