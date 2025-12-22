"""
Pydantic schemas for type-safe data handling across the system.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class TrainingExample(BaseModel):
    """Single instruction-response training pair"""
    instruction: str = Field(description="The user's question or prompt")
    response: str = Field(description="The expected model response")
    category: Optional[str] = Field(default=None, description="Category/topic of this example")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Dataset(BaseModel):
    """Complete dataset for training"""
    examples: List[TrainingExample]
    domain: str
    total_examples: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_jsonl(self, filepath: str):
        """Export to JSONL format"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in self.examples:
                # Format for training (Alpaca-style)
                entry = {
                    "instruction": example.instruction,
                    "input": "",
                    "output": example.response
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')


class EvaluationScore(BaseModel):
    """Score for a single evaluation dimension"""
    dimension: str
    score: int = Field(ge=1, le=10, description="Score from 1-10")
    reasoning: str


class EvaluationResult(BaseModel):
    """Result for a single test case"""
    instruction: str
    model_response: str
    expected_content: Optional[str] = None
    scores: List[EvaluationScore]
    overall_score: float = Field(ge=1.0, le=10.0)
    verdict: Literal["PASS", "FAIL"]
    reasoning: str


class EvaluationReport(BaseModel):
    """Complete evaluation report"""
    test_cases: List[EvaluationResult]
    average_score: float
    pass_rate: float
    passed: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def summary(self) -> str:
        """Generate a summary string"""
        return (
            f"Evaluation Report:\n"
            f"  Average Score: {self.average_score:.2f}/10\n"
            f"  Pass Rate: {self.pass_rate:.1%}\n"
            f"  Overall: {'✓ PASS' if self.passed else '✗ FAIL'}\n"
            f"  Test Cases: {len(self.test_cases)}"
        )


class JobConfig(BaseModel):
    """Configuration for a training job"""
    job_id: str
    user_request: str
    domain_config_path: Optional[str] = None
    
    # Model settings
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    
    # Training settings
    profile: str = "balanced"
    learning_rate: Optional[float] = None
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    lora_rank: Optional[int] = None
    
    # Dataset settings
    dataset_size: Optional[int] = None
    
    # Output
    output_dir: str = "outputs"


class GraphState(BaseModel):
    """State tracked across the LangGraph workflow"""
    job_id: str
    user_request: str
    
    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 3
    
    # Configuration
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    domain_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Paths
    dataset_path: Optional[str] = None
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    
    # Results
    evaluation_score: Optional[float] = None
    evaluation_report: Optional[Dict[str, Any]] = None
    
    # Status
    status: str = "initialized"
    error_log: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        arbitrary_types_allowed = True


class TrainingResult(BaseModel):
    """Result from a training job"""
    success: bool
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None
