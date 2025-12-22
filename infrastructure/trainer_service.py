"""
PEFT Training Service - Production-grade wrapper for fine-tuning with QLoRA.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from agents.schemas import TrainingResult

console = Console()


class PEFTTrainer:
    """
    Disposable trainer pattern: Load → Train → Save → Kill process to free VRAM.
    Uses PEFT (Parameter-Efficient Fine-Tuning) with QLoRA for memory-efficient training.
    """
    
    def __init__(self, config: Dict[str, Any], job_id: str):
        """
        Args:
            config: Base configuration dictionary
            job_id: Unique job identifier
        """
        self.config = config
        self.job_id = job_id
        
        # Extract settings
        self.model_name = config['models']['base_model']
        self.max_seq_length = config['models']['max_seq_length']
        self.load_in_4bit = config['models']['load_in_4bit']
        
        # LoRA settings
        lora_cfg = config['models']['lora']
        self.lora_rank = lora_cfg['rank']
        self.lora_alpha = lora_cfg['alpha']
        self.lora_dropout = lora_cfg['dropout']
        self.target_modules = lora_cfg['target_modules']
        self.use_gradient_checkpointing = lora_cfg['use_gradient_checkpointing']
        
        # Quantization settings (QLoRA)
        quant_cfg = config['models'].get('quantization', {})
        self.bnb_4bit_compute_dtype = quant_cfg.get('bnb_4bit_compute_dtype', 'float16')
        self.bnb_4bit_quant_type = quant_cfg.get('bnb_4bit_quant_type', 'nf4')
        self.bnb_4bit_use_double_quant = quant_cfg.get('bnb_4bit_use_double_quant', True)
        
        # Training settings
        train_cfg = config['training']
        self.learning_rate = train_cfg['learning_rate']['default']
        self.max_steps = train_cfg['max_steps']
        self.warmup_steps = train_cfg['warmup_steps']
        self.batch_size = train_cfg['batch_size']
        self.gradient_accumulation_steps = train_cfg['gradient_accumulation_steps']
        
        # Device
        self.device = self._get_device()
        
        # Output
        self.output_dir = Path(config['storage']['adapters_dir']) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_device(self) -> str:
        """Determine device (CUDA/CPU)."""
        device_cfg = self.config['system']['device']
        
        if device_cfg == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_cfg
    
    def run_training(
        self,
        dataset_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Execute the training job.
        
        Args:
            dataset_path: Path to training dataset (JSONL)
            hyperparameters: Optional hyperparameter overrides
            
        Returns:
            TrainingResult with paths and metrics
        """
        try:
            console.print(f"\n[bold blue]🚂 Starting training job {self.job_id}[/bold blue]")
            
            # Override hyperparameters if provided
            if hyperparameters:
                self._apply_hyperparameters(hyperparameters)
            
            # Check if CUDA is available
            if self.device == 'cuda' and not torch.cuda.is_available():
                console.print("[yellow]Warning: CUDA not available, falling back to CPU (will be slow!)[/yellow]")
                self.device = 'cpu'
                self.load_in_4bit = False  # Can't use 4bit on CPU
            
            # Import dependencies
            console.print("[dim]Loading dependencies...[/dim]")
            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                    TrainingArguments
                )
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_kbit_training
                )
                from datasets import load_dataset
                from trl import SFTTrainer
            except ImportError as e:
                return TrainingResult(
                    success=False,
                    error=f"Required packages not installed: {e}\nRun: pip install transformers peft accelerate bitsandbytes trl"
                )
            
            # 1. Configure Quantization (QLoRA)
            bnb_config = None
            if self.load_in_4bit and self.device == 'cuda':
                console.print("[dim]Configuring 4-bit quantization (QLoRA)...[/dim]")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                )
            
            # 2. Load Base Model
            console.print(f"[dim]Loading base model: {self.model_name}[/dim]")
            
            # Determine dtype
            if self.device == 'cuda':
                compute_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
            else:
                compute_dtype = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map={"": 0} if self.device == 'cuda' and bnb_config is None else ("auto" if self.device == 'cuda' else None),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            # 3. Prepare model for k-bit training
            if self.load_in_4bit and self.device == 'cuda':
                console.print("[dim]Preparing model for k-bit training...[/dim]")
                model = prepare_model_for_kbit_training(
                    model, 
                    use_gradient_checkpointing=self.use_gradient_checkpointing
                )
            elif self.use_gradient_checkpointing:
                # For non-quantized models, enable gradient checkpointing normally
                model.gradient_checkpointing_enable()
            
            # 4. Configure LoRA
            console.print(f"[dim]Adding LoRA adapters (rank={self.lora_rank})[/dim]")
            peft_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.target_modules,
            )
            
            # Apply PEFT
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            # 5. Load Dataset
            console.print(f"[dim]Loading dataset: {dataset_path}[/dim]")
            dataset = load_dataset('json', data_files=dataset_path, split='train')
            
            # Format dataset for Alpaca-style
            def formatting_func(examples):
                texts = []
                for instruction, input_text, output in zip(
                    examples['instruction'],
                    examples['input'],
                    examples['output']
                ):
                    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
                    texts.append(text)
                return texts
            
            # 6. Configure Training
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                fp16=not torch.cuda.is_bf16_supported() if self.device == 'cuda' else False,
                bf16=torch.cuda.is_bf16_supported() if self.device == 'cuda' else False,
                logging_steps=10,
                optim=self.config['training']['optim'],
                weight_decay=self.config['training']['weight_decay'],
                lr_scheduler_type=self.config['training']['lr_scheduler_type'],
                seed=self.config['training']['seed'],
                
                # Disable features we don't need
                save_strategy="no",
                report_to="none",
            )
            
            # 7. Create Trainer - Note: formatting_func returns list of formatted text
            # Pre-format dataset to add 'text' field for older TRL versions
            def add_text_field(example):
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
                return {"text": text}
            
            dataset = dataset.map(add_text_field)
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                args=training_args,
            )
            
            # 8. Train
            console.print("\n[bold green]🏋️ Training started...[/bold green]")
            trainer_stats = trainer.train()
            
            # Extract metrics
            metrics = {
                "train_loss": float(trainer_stats.training_loss) if hasattr(trainer_stats, 'training_loss') else 0.0,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "lora_rank": self.lora_rank,
            }
            
            if self.device == 'cuda':
                metrics["gpu_mem_used_gb"] = torch.cuda.max_memory_allocated() / 1e9
            
            console.print(f"[bold green]✓ Training complete! Loss: {metrics['train_loss']:.4f}[/bold green]")
            
            # 9. Save Adapter
            adapter_path = self.output_dir / "adapter"
            console.print(f"[dim]Saving LoRA adapter to {adapter_path}[/dim]")
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))
            
            # 10. Save merged model (for GGUF conversion)
            model_path = Path(self.config['storage']['models_dir']) / f"{self.job_id}"
            model_path.mkdir(parents=True, exist_ok=True)
            
            console.print(f"[dim]Merging LoRA weights and saving model to {model_path}[/dim]")
            try:
                # Merge LoRA weights back into base model
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                
                console.print("[bold cyan]📦 Model saved in HuggingFace format[/bold cyan]")
                console.print(f"[dim]To convert to GGUF, run:[/dim]")
                console.print(f"[dim]  python -m llama_cpp.convert --outfile {model_path}.gguf {model_path}[/dim]")
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not merge model: {e}[/yellow]")
                model_path = None
            
            # 11. Cleanup GPU memory
            if self.device == 'cuda':
                del model
                del trainer
                torch.cuda.empty_cache()
                console.print("[dim]GPU memory cleaned up[/dim]")
            
            return TrainingResult(
                success=True,
                model_path=str(model_path) if model_path else None,
                adapter_path=str(adapter_path),
                metrics=metrics
            )
        
        except Exception as e:
            console.print(f"[bold red]✗ Training failed: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            
            return TrainingResult(
                success=False,
                error=str(e)
            )
    
    def _apply_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Apply hyperparameter overrides."""
        if 'learning_rate' in hyperparameters:
            self.learning_rate = hyperparameters['learning_rate']
        if 'max_steps' in hyperparameters:
            self.max_steps = hyperparameters['max_steps']
        if 'lora_rank' in hyperparameters:
            self.lora_rank = hyperparameters['lora_rank']
        
        console.print(f"[dim]Applied hyperparameters: LR={self.learning_rate}, Steps={self.max_steps}[/dim]")


def main():
    """Test the trainer."""
    import yaml
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = PEFTTrainer(config, job_id="test_run_001")
    
    # Run training (assuming dataset exists)
    dataset_path = "data/datasets/yoga_training.jsonl"
    
    if not Path(dataset_path).exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        console.print("Run dataset_generator.py first!")
        return
    
    result = trainer.run_training(dataset_path)
    
    if result.success:
        console.print(f"\n[bold green]✓ Training successful![/bold green]")
        console.print(f"Adapter: {result.adapter_path}")
        console.print(f"Model: {result.model_path}")
        console.print(f"Metrics: {result.metrics}")
    else:
        console.print(f"\n[bold red]✗ Training failed:[/bold red] {result.error}")


if __name__ == "__main__":
    main()
