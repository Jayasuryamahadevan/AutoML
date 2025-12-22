# AutoSLM v2 - Agentic SLM Training System

A production-grade agentic system for automated fine-tuning of small language models. Simply describe what you want in natural language (e.g., "Create a yoga training chatbot"), and the system will:

1. **Generate synthetic training data** using Ollama
2. **Fine-tune a model** with Unsloth (4-bit QLoRA)
3. **Evaluate quality** using LLM-as-judge
4. **Automatically retry** with adjusted hyperparameters if needed
5. **Deploy** the final model when it passes quality checks

## 🏗️ Architecture

- **Orchestration**: LangGraph for workflow management
- **Training**: Unsloth for memory-efficient fine-tuning
- **Evaluation**: LLM-based critic agent
- **Inference**: Ollama for GGUF models
- **API**: FastAPI for job management

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended) or CPU (will be slower)
- [Ollama](https://ollama.ai) installed locally

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "d:\AutoSLM v2"

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install Unsloth (special installation)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install CLI dependencies
pip install click
```

### 2. Install Ollama Models

```bash
# Install the model used for data generation and evaluation
ollama pull llama3.2:3b
```

### 3. Run Your First Training Job

**Option A: Command Line Interface**

```bash
python cli.py train "Create a yoga training chatbot" --config config/model_configs/yoga_chatbot.yaml
```

**Option B: Python Workflow**

```python
from orchestration.workflow import run_workflow

result = run_workflow(
    job_id="yoga_bot_001",
    user_request="Create a yoga training chatbot",
    domain_config_path="config/model_configs/yoga_chatbot.yaml",
    max_iterations=3
)

print(f"Final status: {result['status']}")
print(f"Model path: {result.get('adapter_path')}")
```

**Option C: FastAPI Server**

```bash
# Start API server
python api/main.py

# In another terminal, submit a job
curl -X POST http://localhost:8000/jobs/create \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Create a yoga training chatbot",
    "domain_config_path": "config/model_configs/yoga_chatbot.yaml",
    "max_iterations": 3
  }'

# Check job status
curl http://localhost:8000/jobs/{job_id}/status
```

## 📖 CLI Commands

```bash
# Train a model
python cli.py train "Create a [domain] chatbot" --config [config_path]

# Generate synthetic dataset only
python cli.py generate-data "yoga training" --size 100 --output data/my_dataset.jsonl

# View evaluation report
python cli.py show-report outputs/reports/[report_file].json

# List all trained models
python cli.py list-models

# Test a trained model
python cli.py test-model outputs/adapters/[model_dir] "Your test prompt here"
```

## 🎯 How It Works

### Workflow Overview

```
User Request → Dataset Generation → Training (Iter 1)
                                        ↓
                                   Evaluation
                                        ↓
                            ┌───────────┴──────────┐
                            ↓                      ↓
                    Score >= 8.0?              Score < 8.0?
                            ↓                      ↓
                        Deploy              Adjust Hyperparams
                                                   ↓
                                        Training (Iter N) → ...
```

### Key Features

- **Synthetic Data Generation**: Automatically creates domain-specific training data using category decomposition
- **4-bit QLoRA Training**: Memory-efficient fine-tuning with LoRA adapters
- **LLM-as-Judge Evaluation**: Multi-dimensional scoring (accuracy, tone, completeness, safety)
- **Agentic Retry Logic**: Automatically adjusts learning rate and training steps on failure
- **GGUF Export**: Ready-to-deploy quantized models for Ollama

## 📂 Project Structure

```
d:/AutoSLM v2/
├── config/              # Configuration files
│   ├── base_config.yaml
│   ├── training_profiles.yaml
│   └── model_configs/   # Domain-specific configs
├── agents/              # AI agents (dataset gen, critic)
├── infrastructure/      # Training and inference services
├── orchestration/       # LangGraph workflow
├── api/                 # FastAPI gateway
├── data/                # Datasets and test sets
├── outputs/             # Trained models and reports
│   ├── models/          # GGUF models
│   ├── adapters/        # LoRA adapters
│   └── reports/         # Evaluation reports
└── cli.py               # Command-line interface
```

## ⚙️ Configuration

### Base Configuration

Edit `config/base_config.yaml` to adjust:

- **Model settings**: Base model, sequence length, LoRA rank
- **Training**: Learning rate, batch size, max steps
- **Evaluation**: Pass threshold, judge model
- **Orchestration**: Max iterations, retry strategy

### Training Profiles

Choose a profile in `config/training_profiles.yaml`:

- **fast**: Quick iteration (50 steps, rank 8)
- **balanced**: Good quality (100 steps, rank 16) - default
- **quality**: Maximum quality (200 steps, rank 32)

### Domain Configs

Create custom domain configs in `config/model_configs/`:

```yaml
task_name: "my_chatbot"
description: "Helpful description"

dataset_generation:
  domain: "my domain"
  categories:
    - name: "category1"
      description: "What this covers"
      examples: 50
```

## 🔧 Advanced Usage

### Custom Dataset from File

```python
from infrastructure.trainer_service import UnslothTrainer
import yaml

with open('config/base_config.yaml') as f:
    config = yaml.safe_load(f)

trainer = UnslothTrainer(config, job_id="my_job")
result = trainer.run_training("path/to/your/dataset.jsonl")
```

### Evaluate Existing Model

```python
from agents.critic import CriticAgent
from agents.schemas import TrainingExample
import yaml

with open('config/base_config.yaml') as f:
    config = yaml.safe_load(f)

critic = CriticAgent(config)

test_cases = [
    TrainingExample(instruction="Q1", response="Expected answer"),
    # ...
]

model_responses = ["Model's answer to Q1", ...]

report = critic.evaluate_model(test_cases, model_responses)
print(report.summary())
```

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in `config/base_config.yaml`
- Lower `lora_rank` (16 → 8)
- Reduce `max_seq_length`

### Ollama Connection Issues

```bash
# Make sure Ollama is running
ollama serve

# Check available models
ollama list
```

### Training Not Starting

- Check if Unsloth is installed: `pip list | grep unsloth`
- Verify dataset exists and is in JSONL format
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

## 🚧 Future Enhancements

- [ ] Docker containerization
- [ ] MLflow integration for experiment tracking
- [ ] Distributed training with Celery + Redis
- [ ] S3/HuggingFace Hub model registry
- [ ] Web dashboard for job monitoring
- [ ] Multi-GPU support

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Ollama](https://ollama.ai) for local LLM inference
