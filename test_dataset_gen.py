# Quick Test Script for Dataset Generation
import yaml
import sys
sys.path.insert(0, 'D:\\AutoSLM v2')

from agents.dataset_generator import DatasetGenerator
from pathlib import Path

print("Loading configuration...")
with open('config/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override to generate fewer examples for testing
config['dataset']['min_examples'] = 10
config['dataset']['max_examples'] = 10

print("\nCreating dataset generator...")
generator = DatasetGenerator(config)

print("\nGenerating small test dataset (10 examples)...")
dataset = generator.generate_from_request(
    user_request="Create a simple yoga guidance chatbot",
    domain_config_path="config/model_configs/yoga_chatbot.yaml"
)

print(f"\n✓ Generated {dataset.total_examples} examples")
print(f"\nSample example:")
print(f"  Instruction: {dataset.examples[0].instruction}")
print(f"  Response: {dataset.examples[0].response[:100]}...")

# Save test dataset
output_path = "data/datasets/test_yoga_small.jsonl"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
dataset.to_jsonl(output_path)

print(f"\n✓ Dataset saved to {output_path}")
print("\n✅ Dataset generation test PASSED!")
