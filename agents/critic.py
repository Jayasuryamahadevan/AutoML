"""
Critic Agent - LLM-based evaluator for model quality assessment.
"""

import json
from typing import List, Dict, Any
import ollama
from rich.console import Console
from rich.table import Table

from agents.schemas import EvaluationScore, EvaluationResult, EvaluationReport, TrainingExample

console = Console()


class CriticAgent:
    """Evaluates trained models using LLM-as-judge methodology."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Base configuration dictionary
        """
        self.config = config
        self.judge_model = config['evaluation']['judge']['model']
        self.temperature = config['evaluation']['judge']['temperature']
        self.pass_threshold = config['evaluation']['pass_threshold']
        self.scoring_dimensions = config['evaluation']['judge']['scoring_dimensions']
        
    def evaluate_model(
        self,
        test_cases: List[TrainingExample],
        model_responses: List[str],
        domain: str = "general"
    ) -> EvaluationReport:
        """
        Evaluate model responses against test cases.
        
        Args:
            test_cases: List of test examples with expected content
            model_responses: List of actual model outputs
            domain: Domain name for context
            
        Returns:
            EvaluationReport with scores and verdict
        """
        console.print(f"\n[bold blue]🔍 Evaluating {len(test_cases)} test cases...[/bold blue]")
        
        results = []
        
        for test_case, response in zip(test_cases, model_responses):
            result = self._evaluate_single_case(
                instruction=test_case.instruction,
                expected_content=test_case.response,
                model_response=response,
                domain=domain
            )
            results.append(result)
        
        # Calculate aggregate metrics
        avg_score = sum(r.overall_score for r in results) / len(results)
        pass_count = sum(1 for r in results if r.verdict == "PASS")
        pass_rate = pass_count / len(results)
        
        passed = avg_score >= self.pass_threshold
        
        report = EvaluationReport(
            test_cases=results,
            average_score=avg_score,
            pass_rate=pass_rate,
            passed=passed
        )
        
        # Display summary
        self._display_report(report)
        
        return report
    
    def _evaluate_single_case(
        self,
        instruction: str,
        expected_content: str,
        model_response: str,
        domain: str
    ) -> EvaluationResult:
        """Evaluate a single test case using LLM judge."""
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            instruction=instruction,
            expected_content=expected_content,
            model_response=model_response,
            domain=domain
        )
        
        # Call judge model
        response = ollama.generate(
            model=self.judge_model,
            prompt=prompt,
            options={'temperature': self.temperature}
        )
        
        # Parse response
        try:
            eval_data = json.loads(response['response'])
            
            # Extract scores
            scores = [
                EvaluationScore(
                    dimension=dim,
                    score=eval_data['scores'][dim]['score'],
                    reasoning=eval_data['scores'][dim]['reasoning']
                )
                for dim in self.scoring_dimensions
            ]
            
            # Calculate overall score
            overall_score = sum(s.score for s in scores) / len(scores)
            
            verdict = "PASS" if overall_score >= self.pass_threshold else "FAIL"
            
            return EvaluationResult(
                instruction=instruction,
                model_response=model_response,
                expected_content=expected_content,
                scores=scores,
                overall_score=overall_score,
                verdict=verdict,
                reasoning=eval_data.get('overall_reasoning', '')
            )
        
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Warning: Could not parse evaluation, using default scores[/yellow]")
            # Return conservative default
            default_score = EvaluationScore(
                dimension="overall",
                score=5,
                reasoning="Could not evaluate properly"
            )
            
            return EvaluationResult(
                instruction=instruction,
                model_response=model_response,
                expected_content=expected_content,
                scores=[default_score],
                overall_score=5.0,
                verdict="FAIL",
                reasoning="Evaluation error"
            )
    
    def _build_evaluation_prompt(
        self,
        instruction: str,
        expected_content: str,
        model_response: str,
        domain: str
    ) -> str:
        """Build the prompt for LLM judge."""
        
        dimensions_desc = "\n".join([f"- {dim}" for dim in self.scoring_dimensions])
        
        prompt = f"""You are an expert evaluator for a {domain} chatbot.

**Task:** Evaluate the model's response to a user question.

**User Question:**
{instruction}

**Expected Content (Reference):**
{expected_content}

**Actual Model Response:**
{model_response}

**Evaluation Criteria:**
Score each dimension from 1-10:
{dimensions_desc}

**Return format (JSON only, no markdown):**
{{
  "scores": {{
    "accuracy": {{"score": X, "reasoning": "why"}},
    "tone": {{"score": X, "reasoning": "why"}},
    "completeness": {{"score": X, "reasoning": "why"}},
    "safety": {{"score": X, "reasoning": "why"}}
  }},
  "overall_reasoning": "summary of evaluation"
}}

Be strict but fair. A score of 8+ means excellent quality."""
        
        return prompt
    
    def _display_report(self, report: EvaluationReport):
        """Display evaluation report in a nice table."""
        
        # Summary
        console.print(f"\n[bold]Evaluation Summary:[/bold]")
        console.print(report.summary())
        
        # Detailed table
        table = Table(title="Detailed Results", show_header=True)
        table.add_column("Test Case", style="cyan", width=40)
        table.add_column("Score", justify="center", style="yellow")
        table.add_column("Verdict", justify="center")
        
        for i, result in enumerate(report.test_cases[:10], 1):  # Show first 10
            instruction_short = result.instruction[:40] + "..." if len(result.instruction) > 40 else result.instruction
            
            verdict_style = "green" if result.verdict == "PASS" else "red"
            verdict_icon = "✓" if result.verdict == "PASS" else "✗"
            
            table.add_row(
                f"{i}. {instruction_short}",
                f"{result.overall_score:.1f}/10",
                f"[{verdict_style}]{verdict_icon} {result.verdict}[/{verdict_style}]"
            )
        
        if len(report.test_cases) > 10:
            table.add_row("...", "...", "...")
        
        console.print(table)


def main():
    """Test the critic agent."""
    import yaml
    
    # Load config
    with open('config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create critic
    critic = CriticAgent(config)
    
    # Mock test cases
    test_cases = [
        TrainingExample(
            instruction="What are the steps to perform Downward Facing Dog?",
            response="Start on hands and knees. Lift hips up and back, straightening legs. Press heels toward floor."
        ),
        TrainingExample(
            instruction="How can I modify Chair Pose if I have knee pain?",
            response="Don't bend knees as deeply, or use a wall for support. Listen to your body."
        )
    ]
    
    # Mock model responses (simulating inference)
    model_responses = [
        "To do Downward Dog: Begin on all fours, then lift your hips upward and back, forming an inverted V. Straighten your legs and press your heels down. Keep your head between your arms.",
        "If you have knee pain during Chair Pose, try these modifications: 1) Don't squat as deep, 2) Place a block under your hips for support, 3) Always stop if you feel pain."
    ]
    
    # Evaluate
    report = critic.evaluate_model(test_cases, model_responses, domain="yoga instruction")
    
    # Save report
    import json
    with open('outputs/reports/test_evaluation.json', 'w') as f:
        json.dump(report.dict(), f, indent=2)
    
    console.print("\n[bold green]✓ Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
