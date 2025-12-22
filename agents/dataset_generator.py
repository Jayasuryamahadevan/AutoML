"""
Dataset Generator Agent - Creates synthetic training data using Ollama.
"""

import json
import yaml
from typing import List, Dict, Any
from pathlib import Path
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.schemas import TrainingExample, Dataset

console = Console()


class DatasetGenerator:
    """Generates domain-specific training datasets using LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Base configuration dictionary
        """
        self.config = config
        self.ollama_model = config['dataset']['ollama']['model']
        self.temperature = config['dataset']['ollama']['temperature']
        self.num_ctx = config['dataset']['ollama']['num_ctx']
        
    def generate_from_request(self, user_request: str, domain_config_path: str = None) -> Dataset:
        """
        Generate a dataset based on user request.
        
        Args:
            user_request: Natural language description (e.g., "Create a yoga training chatbot")
            domain_config_path: Optional path to domain-specific config
            
        Returns:
            Dataset object with generated examples
        """
        console.print(f"\n[bold blue]🤖 Generating dataset for:[/bold blue] {user_request}")
        
        # Load domain config if provided
        domain_config = self._load_domain_config(domain_config_path) if domain_config_path else None
        
        # Extract domain and generate categories
        if domain_config:
            domain = domain_config['dataset_generation']['domain']
            categories = domain_config['dataset_generation']['categories']
        else:
            # Parse from user request
            domain = self._extract_domain(user_request)
            categories = self._generate_categories(domain)
        
        # Generate examples for each category
        all_examples = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for category in categories:
                task = progress.add_task(
                    f"Generating {category['name']}...",
                    total=category['examples']
                )
                
                examples = self._generate_category_examples(
                    domain=domain,
                    category=category,
                    count=category['examples']
                )
                
                all_examples.extend(examples)
                progress.update(task, completed=category['examples'])
        
        # Create dataset
        dataset = Dataset(
            examples=all_examples,
            domain=domain,
            total_examples=len(all_examples)
        )
        
        console.print(f"[bold green]✓ Generated {len(all_examples)} training examples[/bold green]")
        
        return dataset
    
    def _load_domain_config(self, config_path: str) -> Dict[str, Any]:
        """Load domain-specific configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _extract_domain(self, user_request: str) -> str:
        """Extract domain from user request using LLM."""
        prompt = f"""Extract the core domain/topic from this request in 3-5 words:
Request: "{user_request}"

Return ONLY the domain, nothing else.
Examples:
- "Create a yoga training chatbot" -> "yoga and wellness instruction"
- "Build a legal advice bot" -> "legal advice and consultation"
"""
        
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={'temperature': 0.3}
        )
        
        return response['response'].strip().strip('"')
    
    def _generate_categories(self, domain: str) -> List[Dict[str, Any]]:
        """Generate categories for a domain using LLM."""
        prompt = f"""For the domain "{domain}", create 5 specific categories of questions/topics to cover.

Return as JSON array with this exact format:
[
  {{"name": "category_name", "description": "what this covers", "examples": 20}},
  ...
]

Make sure examples add up to around 100 total.
Return ONLY valid JSON, no markdown or explanation."""
        
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={'temperature': 0.5, 'num_ctx': self.num_ctx}
        )
        
        # Parse JSON
        try:
            categories = json.loads(response['response'])
            return categories
        except json.JSONDecodeError:
            # Fallback to generic categories
            console.print("[yellow]Warning: Could not parse categories, using defaults[/yellow]")
            return [
                {"name": "general_questions", "description": f"General questions about {domain}", "examples": 100}
            ]
    
    def _generate_category_examples(
        self,
        domain: str,
        category: Dict[str, Any],
        count: int
    ) -> List[TrainingExample]:
        """Generate examples for a specific category."""
        
        prompt = f"""You are a dataset creator for training a chatbot in the domain: {domain}

Category: {category['name']}
Description: {category['description']}

Generate {count} diverse instruction-response pairs for this category.

IMPORTANT:
- Instructions should be natural questions/requests users would ask
- Responses should be helpful, accurate, and appropriate in tone
- Make examples diverse (different phrasing, difficulty levels, contexts)
- Responses should be 2-4 sentences typically

Return as JSON array:
[
  {{"instruction": "question here", "response": "helpful answer here"}},
  ...
]

Return ONLY valid JSON, no markdown or explanation."""
        
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_ctx': self.num_ctx
            }
        )
        
        # Parse response
        try:
            examples_data = json.loads(response['response'])
            examples = [
                TrainingExample(
                    instruction=ex['instruction'],
                    response=ex['response'],
                    category=category['name']
                )
                for ex in examples_data
            ]
            return examples
        
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[red]Error parsing examples for {category['name']}: {e}[/red]")
            # Return empty list for this category
            return []
    
    def create_test_set(self, dataset: Dataset, test_split: float = 0.1) -> Dataset:
        """Create a test set from the dataset."""
        import random
        
        examples = dataset.examples.copy()
        random.shuffle(examples)
        
        test_size = int(len(examples) * test_split)
        test_examples = examples[:test_size]
        
        return Dataset(
            examples=test_examples,
            domain=dataset.domain,
            total_examples=len(test_examples)
        )


def main():
    """Test the dataset generator."""
    import sys
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = DatasetGenerator(config)
    
    # Generate dataset
    user_request = "Create a yoga training chatbot"
    domain_config = "config/model_configs/yoga_chatbot.yaml"
    
    dataset = generator.generate_from_request(user_request, domain_config)
    
    # Save dataset
    output_path = "data/datasets/yoga_training.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_jsonl(output_path)
    
    console.print(f"\n[bold green]✓ Dataset saved to {output_path}[/bold green]")
    
    # Create test set
    test_dataset = generator.create_test_set(dataset, test_split=0.1)
    test_path = "data/test_sets/yoga_test.jsonl"
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_dataset.to_jsonl(test_path)
    
    console.print(f"[bold green]✓ Test set saved to {test_path}[/bold green]")


if __name__ == "__main__":
    main()
