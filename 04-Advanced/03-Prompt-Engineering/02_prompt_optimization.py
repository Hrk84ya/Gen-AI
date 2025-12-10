"""
Prompt Optimization and Evaluation Framework

This module provides tools for systematically optimizing prompts through
automated testing, performance evaluation, and iterative improvement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
import json
import time
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import re
from collections import defaultdict, Counter


@dataclass
class EvaluationMetric:
    """Represents an evaluation metric for prompt performance"""
    name: str
    function: Callable
    higher_is_better: bool = True
    description: str = ""
    weight: float = 1.0


@dataclass
class TestCase:
    """Represents a test case for prompt evaluation"""
    input_variables: Dict[str, Any]
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class PromptResult:
    """Results from evaluating a prompt"""
    prompt_id: str
    prompt_text: str
    test_case: TestCase
    actual_output: str
    metrics: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for prompt evaluators"""
    
    @abstractmethod
    def evaluate(self, prompt: str, test_case: TestCase) -> Dict[str, float]:
        """Evaluate a prompt on a test case and return metrics"""
        pass


class AccuracyEvaluator(BaseEvaluator):
    """Evaluates prompt accuracy for classification tasks"""
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
    
    def evaluate(self, prompt: str, test_case: TestCase) -> Dict[str, float]:
        """Evaluate accuracy by exact match"""
        # In a real implementation, this would call an LLM API
        # For testing, we'll simulate responses
        actual_output = self._simulate_llm_response(prompt, test_case)
        
        expected = str(test_case.expected_output)
        actual = str(actual_output)
        
        if not self.case_sensitive:
            expected = expected.lower().strip()
            actual = actual.lower().strip()
        
        accuracy = 1.0 if expected == actual else 0.0
        
        return {
            "accuracy": accuracy,
            "exact_match": accuracy
        }
    
    def _simulate_llm_response(self, prompt: str, test_case: TestCase) -> str:
        """Simulate LLM response for testing purposes"""
        # Simple simulation based on prompt content
        if "classify" in prompt.lower():
            categories = ["positive", "negative", "neutral", "spam", "not spam"]
            return random.choice(categories)
        elif "generate" in prompt.lower():
            return "Generated response based on the prompt"
        else:
            return "Default response"


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Evaluates semantic similarity between expected and actual outputs"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def evaluate(self, prompt: str, test_case: TestCase) -> Dict[str, float]:
        """Evaluate semantic similarity"""
        actual_output = self._simulate_llm_response(prompt, test_case)
        
        # Simulate semantic similarity calculation
        # In practice, you'd use embeddings or other NLP techniques
        similarity = self._calculate_similarity(
            str(test_case.expected_output), 
            str(actual_output)
        )
        
        return {
            "semantic_similarity": similarity,
            "meets_threshold": 1.0 if similarity >= self.similarity_threshold else 0.0
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)"""
        # Simple word overlap similarity for demonstration
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _simulate_llm_response(self, prompt: str, test_case: TestCase) -> str:
        """Simulate LLM response"""
        # Return something similar to expected output for testing
        expected = str(test_case.expected_output)
        words = expected.split()
        
        # Randomly modify some words to simulate variation
        modified_words = []
        for word in words:
            if random.random() < 0.3:  # 30% chance to modify
                modified_words.append(f"similar_{word}")
            else:
                modified_words.append(word)
        
        return " ".join(modified_words)


class LengthEvaluator(BaseEvaluator):
    """Evaluates output length characteristics"""
    
    def __init__(self, target_length: Optional[int] = None, 
                 length_tolerance: float = 0.2):
        self.target_length = target_length
        self.length_tolerance = length_tolerance
    
    def evaluate(self, prompt: str, test_case: TestCase) -> Dict[str, float]:
        """Evaluate length-related metrics"""
        actual_output = self._simulate_llm_response(prompt, test_case)
        
        actual_length = len(actual_output.split())
        
        metrics = {
            "output_length": actual_length,
            "length_score": 1.0  # Default score
        }
        
        if self.target_length:
            length_diff = abs(actual_length - self.target_length)
            tolerance_range = self.target_length * self.length_tolerance
            
            if length_diff <= tolerance_range:
                metrics["length_score"] = 1.0
            else:
                # Penalize based on how far outside tolerance
                penalty = (length_diff - tolerance_range) / self.target_length
                metrics["length_score"] = max(0.0, 1.0 - penalty)
        
        return metrics
    
    def _simulate_llm_response(self, prompt: str, test_case: TestCase) -> str:
        """Simulate LLM response with varying lengths"""
        base_length = random.randint(10, 100)
        words = [f"word{i}" for i in range(base_length)]
        return " ".join(words)


class PromptOptimizer:
    """Main class for optimizing prompts through systematic evaluation"""
    
    def __init__(self, evaluators: List[BaseEvaluator], 
                 metrics: Optional[List[EvaluationMetric]] = None):
        self.evaluators = evaluators
        self.metrics = metrics or self._default_metrics()
        self.optimization_history = []
        self.results_cache = {}
    
    def _default_metrics(self) -> List[EvaluationMetric]:
        """Default evaluation metrics"""
        return [
            EvaluationMetric("accuracy", lambda x: x.get("accuracy", 0), True, "Exact match accuracy"),
            EvaluationMetric("semantic_similarity", lambda x: x.get("semantic_similarity", 0), True, "Semantic similarity"),
            EvaluationMetric("length_score", lambda x: x.get("length_score", 1), True, "Length appropriateness")
        ]
    
    def evaluate_prompt(self, prompt: str, test_cases: List[TestCase], 
                       prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single prompt on multiple test cases
        
        Args:
            prompt: The prompt to evaluate
            test_cases: List of test cases
            prompt_id: Optional identifier for the prompt
        
        Returns:
            Evaluation results
        """
        if prompt_id is None:
            prompt_id = f"prompt_{hash(prompt) % 10000}"
        
        results = []
        
        for test_case in test_cases:
            start_time = time.time()
            
            # Collect metrics from all evaluators
            all_metrics = {}
            for evaluator in self.evaluators:
                metrics = evaluator.evaluate(prompt, test_case)
                all_metrics.update(metrics)
            
            execution_time = time.time() - start_time
            
            result = PromptResult(
                prompt_id=prompt_id,
                prompt_text=prompt,
                test_case=test_case,
                actual_output="",  # Would be filled by actual LLM call
                metrics=all_metrics,
                execution_time=execution_time
            )
            
            results.append(result)
        
        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(results)
        
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "individual_results": results,
            "aggregated_metrics": aggregated_metrics,
            "total_test_cases": len(test_cases),
            "average_execution_time": np.mean([r.execution_time for r in results])
        }
    
    def _aggregate_metrics(self, results: List[PromptResult]) -> Dict[str, float]:
        """Aggregate metrics across multiple test cases"""
        if not results:
            return {}
        
        # Collect all metric names
        all_metric_names = set()
        for result in results:
            all_metric_names.update(result.metrics.keys())
        
        aggregated = {}
        
        for metric_name in all_metric_names:
            values = [r.metrics.get(metric_name, 0) for r in results]
            weights = [r.test_case.weight for r in results]
            
            # Weighted average
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)
            
            aggregated[f"{metric_name}_mean"] = weighted_sum / total_weight if total_weight > 0 else 0
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_min"] = min(values)
            aggregated[f"{metric_name}_max"] = max(values)
        
        return aggregated
    
    def optimize_prompt(self, base_prompt: str, test_cases: List[TestCase],
                       optimization_strategies: List[str],
                       max_iterations: int = 10) -> Dict[str, Any]:
        """
        Optimize a prompt using various strategies
        
        Args:
            base_prompt: Starting prompt
            test_cases: Test cases for evaluation
            optimization_strategies: List of optimization strategies to try
            max_iterations: Maximum number of optimization iterations
        
        Returns:
            Optimization results including best prompt
        """
        optimization_run = {
            "base_prompt": base_prompt,
            "strategies": optimization_strategies,
            "iterations": [],
            "best_prompt": base_prompt,
            "best_score": 0,
            "improvement_history": []
        }
        
        current_prompt = base_prompt
        current_score = 0
        
        # Evaluate base prompt
        base_results = self.evaluate_prompt(current_prompt, test_cases, "base_prompt")
        current_score = self._calculate_composite_score(base_results["aggregated_metrics"])
        optimization_run["best_score"] = current_score
        
        for iteration in range(max_iterations):
            print(f"Optimization iteration {iteration + 1}/{max_iterations}")
            
            # Generate prompt variations
            variations = self._generate_prompt_variations(
                current_prompt, optimization_strategies
            )
            
            iteration_results = {
                "iteration": iteration + 1,
                "variations": [],
                "best_variation": None,
                "improvement": 0
            }
            
            best_iteration_score = current_score
            best_iteration_prompt = current_prompt
            
            # Evaluate each variation
            for i, variation in enumerate(variations):
                variation_id = f"iter_{iteration+1}_var_{i+1}"
                results = self.evaluate_prompt(variation, test_cases, variation_id)
                score = self._calculate_composite_score(results["aggregated_metrics"])
                
                variation_result = {
                    "prompt": variation,
                    "score": score,
                    "results": results
                }
                iteration_results["variations"].append(variation_result)
                
                # Check if this is the best so far
                if score > best_iteration_score:
                    best_iteration_score = score
                    best_iteration_prompt = variation
                    iteration_results["best_variation"] = variation_result
            
            # Update current prompt if improvement found
            improvement = best_iteration_score - current_score
            if improvement > 0:
                current_prompt = best_iteration_prompt
                current_score = best_iteration_score
                optimization_run["best_prompt"] = current_prompt
                optimization_run["best_score"] = current_score
                
                print(f"  Improvement found: +{improvement:.4f}")
            else:
                print(f"  No improvement in iteration {iteration + 1}")
            
            iteration_results["improvement"] = improvement
            optimization_run["iterations"].append(iteration_results)
            optimization_run["improvement_history"].append(improvement)
            
            # Early stopping if no improvement for several iterations
            if len(optimization_run["improvement_history"]) >= 3:
                recent_improvements = optimization_run["improvement_history"][-3:]
                if all(imp <= 0 for imp in recent_improvements):
                    print("  Early stopping: no improvement in last 3 iterations")
                    break
        
        self.optimization_history.append(optimization_run)
        return optimization_run
    
    def _generate_prompt_variations(self, base_prompt: str, 
                                  strategies: List[str]) -> List[str]:
        """Generate prompt variations using different strategies"""
        variations = []
        
        for strategy in strategies:
            if strategy == "add_examples":
                variations.append(f"{base_prompt}\n\nHere are some examples to guide you:")
            
            elif strategy == "be_specific":
                variations.append(f"{base_prompt}\n\nBe as specific and detailed as possible in your response.")
            
            elif strategy == "step_by_step":
                variations.append(f"{base_prompt}\n\nThink through this step by step:")
            
            elif strategy == "format_request":
                variations.append(f"{base_prompt}\n\nFormat your response with clear headings and bullet points.")
            
            elif strategy == "role_play":
                variations.append(f"You are an expert in this field. {base_prompt}")
            
            elif strategy == "constraint_addition":
                variations.append(f"{base_prompt}\n\nKeep your response concise and under 100 words.")
            
            elif strategy == "clarification":
                variations.append(f"{base_prompt}\n\nIf anything is unclear, state your assumptions clearly.")
            
            elif strategy == "positive_framing":
                # Replace negative words with positive alternatives
                positive_prompt = base_prompt.replace("don't", "please avoid")
                positive_prompt = positive_prompt.replace("can't", "unable to")
                variations.append(positive_prompt)
        
        return variations
    
    def _calculate_composite_score(self, aggregated_metrics: Dict[str, float]) -> float:
        """Calculate composite score from aggregated metrics"""
        total_score = 0
        total_weight = 0
        
        for metric in self.metrics:
            metric_key = f"{metric.name}_mean"
            if metric_key in aggregated_metrics:
                value = aggregated_metrics[metric_key]
                
                # Normalize score based on whether higher is better
                if not metric.higher_is_better:
                    value = 1.0 - value
                
                total_score += value * metric.weight
                total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def compare_prompts(self, prompts: Dict[str, str], test_cases: List[TestCase]) -> pd.DataFrame:
        """
        Compare multiple prompts and return results as DataFrame
        
        Args:
            prompts: Dictionary of prompt_id -> prompt_text
            test_cases: Test cases for evaluation
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for prompt_id, prompt_text in prompts.items():
            evaluation = self.evaluate_prompt(prompt_text, test_cases, prompt_id)
            
            row = {
                "prompt_id": prompt_id,
                "prompt_length": len(prompt_text.split()),
                "total_test_cases": evaluation["total_test_cases"],
                "avg_execution_time": evaluation["average_execution_time"]
            }
            
            # Add aggregated metrics
            for metric_name, value in evaluation["aggregated_metrics"].items():
                row[metric_name] = value
            
            # Calculate composite score
            row["composite_score"] = self._calculate_composite_score(
                evaluation["aggregated_metrics"]
            )
            
            results.append(row)
        
        df = pd.DataFrame(results)
        return df.sort_values("composite_score", ascending=False)
    
    def plot_optimization_history(self, optimization_run: Dict[str, Any]):
        """Plot optimization history"""
        improvements = optimization_run["improvement_history"]
        iterations = range(1, len(improvements) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, improvements, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Score Improvement')
        plt.title('Prompt Optimization Progress')
        plt.grid(True, alpha=0.3)
        
        # Annotate best improvement
        if improvements:
            best_iter = np.argmax(improvements) + 1
            best_improvement = max(improvements)
            plt.annotate(f'Best: {best_improvement:.4f}', 
                        xy=(best_iter, best_improvement),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, optimization_run: Dict[str, Any]) -> str:
        """Generate a text report of optimization results"""
        report = []
        report.append("=== PROMPT OPTIMIZATION REPORT ===\n")
        
        report.append(f"Base Prompt: {optimization_run['base_prompt'][:100]}...")
        report.append(f"Optimization Strategies: {', '.join(optimization_run['strategies'])}")
        report.append(f"Total Iterations: {len(optimization_run['iterations'])}")
        report.append(f"Final Score: {optimization_run['best_score']:.4f}")
        
        total_improvement = optimization_run['best_score'] - optimization_run.get('base_score', 0)
        report.append(f"Total Improvement: {total_improvement:.4f}")
        
        report.append("\n=== BEST PROMPT ===")
        report.append(optimization_run['best_prompt'])
        
        report.append("\n=== ITERATION SUMMARY ===")
        for iteration in optimization_run['iterations']:
            iter_num = iteration['iteration']
            improvement = iteration['improvement']
            report.append(f"Iteration {iter_num}: {improvement:+.4f}")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Prompt Optimization Framework ===")
    
    # Create evaluators
    evaluators = [
        AccuracyEvaluator(case_sensitive=False),
        SemanticSimilarityEvaluator(similarity_threshold=0.7),
        LengthEvaluator(target_length=50, length_tolerance=0.3)
    ]
    
    # Create optimizer
    optimizer = PromptOptimizer(evaluators)
    
    # Create test cases
    test_cases = [
        TestCase(
            input_variables={"text": "I love this product!", "categories": "positive, negative, neutral"},
            expected_output="positive",
            metadata={"domain": "sentiment"}
        ),
        TestCase(
            input_variables={"text": "This is terrible quality", "categories": "positive, negative, neutral"},
            expected_output="negative",
            metadata={"domain": "sentiment"}
        ),
        TestCase(
            input_variables={"text": "It's okay, nothing special", "categories": "positive, negative, neutral"},
            expected_output="neutral",
            metadata={"domain": "sentiment"}
        )
    ]
    
    # Test single prompt evaluation
    print("\n1. Testing Single Prompt Evaluation")
    base_prompt = "Classify the sentiment of this text: {text}. Categories: {categories}. Answer:"
    
    results = optimizer.evaluate_prompt(base_prompt, test_cases)
    print(f"Prompt evaluated on {results['total_test_cases']} test cases")
    print(f"Average execution time: {results['average_execution_time']:.4f}s")
    
    # Print some metrics
    for metric_name, value in results['aggregated_metrics'].items():
        if 'mean' in metric_name:
            print(f"{metric_name}: {value:.4f}")
    
    # Test prompt comparison
    print("\n2. Testing Prompt Comparison")
    prompts_to_compare = {
        "basic": "Classify: {text}. Categories: {categories}",
        "detailed": "Carefully analyze the sentiment of this text: {text}. Choose from: {categories}. Consider the emotional tone and context.",
        "step_by_step": "Classify the sentiment step by step: {text}. Categories: {categories}. Think about key words and phrases first."
    }
    
    comparison_df = optimizer.compare_prompts(prompts_to_compare, test_cases)
    print("Comparison Results:")
    print(comparison_df[['prompt_id', 'composite_score', 'accuracy_mean', 'semantic_similarity_mean']].to_string(index=False))
    
    # Test prompt optimization
    print("\n3. Testing Prompt Optimization")
    optimization_strategies = [
        "add_examples",
        "be_specific", 
        "step_by_step",
        "role_play"
    ]
    
    optimization_result = optimizer.optimize_prompt(
        base_prompt=base_prompt,
        test_cases=test_cases,
        optimization_strategies=optimization_strategies,
        max_iterations=3  # Reduced for testing
    )
    
    print(f"Optimization completed!")
    print(f"Best score: {optimization_result['best_score']:.4f}")
    print(f"Total iterations: {len(optimization_result['iterations'])}")
    
    # Generate and print report
    print("\n4. Optimization Report")
    report = optimizer.generate_report(optimization_result)
    print(report)
    
    # Test plotting (would show plot in interactive environment)
    print("\n5. Plotting optimization history...")
    try:
        optimizer.plot_optimization_history(optimization_result)
        print("Plot generated successfully!")
    except Exception as e:
        print(f"Plotting failed (expected in non-interactive environment): {e}")
    
    print("\n✅ All prompt optimization tests completed successfully!")