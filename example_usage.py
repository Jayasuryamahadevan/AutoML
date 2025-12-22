# Example: Using AutoSLM Programmatically

import yaml
from orchestration.workflow import run_workflow

def main():
    """
    Example: Create a custom chatbot programmatically
    """
    
    # Option 1: Use existing yoga chatbot config
    print("Example 1: Using pre-configured domain")
    print("-" * 50)
    
    result = run_workflow(
        job_id="example_yoga_001",
        user_request="Create a yoga training chatbot that helps with poses, breathing, and sequences",
        domain_config_path="config/model_configs/yoga_chatbot.yaml",
        max_iterations=2  # Reduce for faster testing
    )
    
    print(f"\nResult status: {result.get('status')}")
    if result.get('status') == 'deployed':
        print(f"Model path: {result['deployment_info']['adapter_path']}")
        print(f"Final score: {result['deployment_info']['final_score']:.2f}/10")
    
    
    # Option 2: Generic request without domain config
    print("\n\nExample 2: Generic request (auto-generate categories)")
    print("-" * 50)
    
    result2 = run_workflow(
        job_id="example_cooking_001",
        user_request="Create a cooking assistant chatbot that helps with recipes and techniques",
        max_iterations=2
    )
    
    print(f"\nResult status: {result2.get('status')}")
    
    
    # Option 3: Test individual components
    print("\n\nExample 3: Testing individual components")
    print("-" * 50)
    
    from agents.dataset_generator import DatasetGenerator
    from agents.critic import CriticAgent
    
    # Load config
    with open('config/base_config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Generate dataset
    print("Generating dataset...")
    generator = DatasetGenerator(config)
    dataset = generator.generate_from_request("fitness training chatbot")
    
    print(f"Generated {dataset.total_examples} examples")
    print(f"First example: {dataset.examples[0].instruction[:50]}...")
    
    # You can also directly evaluate a model
    # critic = CriticAgent(config)
    # report = critic.evaluate_model(test_cases, model_responses)


if __name__ == "__main__":
    main()
