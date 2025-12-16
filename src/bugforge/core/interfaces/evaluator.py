"""
Evaluator interface - defines the contract for performance evaluation.

Evaluators are components that can assess the performance of models,
policies, or other components using various metrics and criteria.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Action, Feedback, Observation, State, Trajectory


class Evaluator(ABC):
    """
    Abstract base class for all evaluators in BugForge.
    
    Evaluators are responsible for assessing performance using
    metrics, benchmarks, and evaluation protocols.
    """
    
    @abstractmethod
    def evaluate(self, component: Union["Model", "Policy"], 
                test_data: Union[List[Observation], Trajectory, List[Trajectory]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate a component on test data.
        
        Args:
            component: The component to evaluate (model, policy, etc.)
            test_data: Test data for evaluation
            context: Additional evaluation context (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Get the list of metrics this evaluator provides.
        
        Returns:
            List of metric names
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the evaluator's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    def batch_evaluate(self, components: List[Union["Model", "Policy"]], 
                      test_data: Union[List[Observation], Trajectory, List[Trajectory]], 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple components.
        
        Args:
            components: List of components to evaluate
            test_data: Test data for evaluation
            context: Additional evaluation context (optional)
            
        Returns:
            Dictionary mapping component names to metric dictionaries
        """
        results = {}
        for component in components:
            component_name = component.__class__.__name__
            results[component_name] = self.evaluate(component, test_data, context)
        return results
    
    def compare(self, components: List[Union["Model", "Policy"]], 
               test_data: Union[List[Observation], Trajectory, List[Trajectory]], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare multiple components and return ranking.
        
        Args:
            components: List of components to compare
            test_data: Test data for comparison
            context: Additional evaluation context (optional)
            
        Returns:
            Dictionary with comparison results and rankings
        """
        results = self.batch_evaluate(components, test_data, context)
        
        # Rank components by primary metric (first metric in list)
        primary_metric = self.get_metrics()[0] if self.get_metrics() else "score"
        
        rankings = sorted(
            [(name, metrics[primary_metric]) for name, metrics in results.items()],
            key=lambda x: x[1],
            reverse=True  # Higher is better
        )
        
        return {
            "results": results,
            "rankings": rankings,
            "primary_metric": primary_metric
        }
    
    def reset(self) -> None:
        """
        Reset evaluator state for new evaluation sessions.
        
        Called to clear any accumulated evaluation state.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save evaluator state to disk.
        
        Args:
            path: Path where to save the evaluator
        """
        raise NotImplementedError("Evaluator saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load evaluator state from disk.
        
        Args:
            path: Path from which to load the evaluator
        """
        raise NotImplementedError("Evaluator loading not implemented")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.get_metrics()})"


class PerformanceEvaluator(Evaluator):
    """
    Evaluator focused on performance metrics.
    
    Evaluates components based on performance metrics like accuracy,
    precision, recall, F1-score, etc.
    """
    
    @abstractmethod
    def compute_accuracy(self, predictions: List[Action], targets: List[Action]) -> float:
        """
        Compute accuracy metric.
        
        Args:
            predictions: List of predicted actions
            targets: List of target actions
            
        Returns:
            Accuracy score
        """
        pass
    
    @abstractmethod
    def compute_precision(self, predictions: List[Action], targets: List[Action]) -> float:
        """
        Compute precision metric.
        
        Args:
            predictions: List of predicted actions
            targets: List of target actions
            
        Returns:
            Precision score
        """
        pass
    
    @abstractmethod
    def compute_recall(self, predictions: List[Action], targets: List[Action]) -> float:
        """
        Compute recall metric.
        
        Args:
            predictions: List of predicted actions
            targets: List of target actions
            
        Returns:
            Recall score
        """
        pass
    
    def compute_f1_score(self, precision: float, recall: float) -> float:
        """
        Compute F1-score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1-score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def get_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return ["accuracy", "precision", "recall", "f1_score"]


class SafetyEvaluator(Evaluator):
    """
    Evaluator focused on safety and reliability metrics.
    
    Evaluates components based on safety constraints, robustness,
    and reliability criteria.
    """
    
    @abstractmethod
    def check_safety_constraints(self, actions: List[Action]) -> bool:
        """
        Check if actions violate safety constraints.
        
        Args:
            actions: List of actions to check
            
        Returns:
            True if all actions are safe
        """
        pass
    
    @abstractmethod
    def compute_robustness_score(self, component: Union["Model", "Policy"], 
                               perturbed_data: List[Observation]) -> float:
        """
        Compute robustness score under perturbed conditions.
        
        Args:
            component: Component to evaluate
            perturbed_data: Perturbed test observations
            
        Returns:
            Robustness score
        """
        pass
    
    @abstractmethod
    def compute_reliability_score(self, component: Union["Model", "Policy"], 
                                test_data: List[Observation]) -> float:
        """
        Compute reliability score over multiple runs.
        
        Args:
            component: Component to evaluate
            test_data: Test observations
            
        Returns:
            Reliability score
        """
        pass
    
    def get_metrics(self) -> List[str]:
        """Get list of available safety metrics."""
        return ["safety_violations", "robustness", "reliability"]


class BenchmarkEvaluator(Evaluator):
    """
    Evaluator that uses standardized benchmarks.
    
    Evaluates components against predefined benchmark suites
    and standard evaluation protocols.
    """
    
    @abstractmethod
    def load_benchmark(self, benchmark_name: str) -> Any:
        """
        Load a benchmark suite.
        
        Args:
            benchmark_name: Name of the benchmark to load
            
        Returns:
            Loaded benchmark data/tasks
        """
        pass
    
    @abstractmethod
    def run_benchmark(self, component: Union["Model", "Policy"], 
                     benchmark: Any) -> Dict[str, float]:
        """
        Run a benchmark evaluation.
        
        Args:
            component: Component to evaluate
            benchmark: Benchmark suite to run
            
        Returns:
            Dictionary of benchmark scores
        """
        pass
    
    @abstractmethod
    def get_available_benchmarks(self) -> List[str]:
        """
        Get list of available benchmarks.
        
        Returns:
            List of benchmark names
        """
        pass
    
    def evaluate(self, component: Union["Model", "Policy"], 
                test_data: Union[List[Observation], Trajectory, List[Trajectory]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate using benchmark if specified in context.
        
        Args:
            component: Component to evaluate
            test_data: Test data (may be ignored for benchmarks)
            context: Evaluation context
            
        Returns:
            Dictionary of evaluation metrics
        """
        if context and "benchmark" in context:
            benchmark_name = context["benchmark"]
            benchmark = self.load_benchmark(benchmark_name)
            return self.run_benchmark(component, benchmark)
        else:
            raise ValueError("Benchmark name must be specified in context")
    
    def get_metrics(self) -> List[str]:
        """Get list of available benchmark metrics."""
        return ["benchmark_score", "benchmark_rank", "benchmark_percentile"]


class ContinuousEvaluator(Evaluator):
    """
    Evaluator that continuously monitors performance.
    
    Provides ongoing evaluation and monitoring of component
    performance during operation.
    """
    
    @abstractmethod
    def start_monitoring(self, component: Union["Model", "Policy"]) -> None:
        """
        Start continuous monitoring of a component.
        
        Args:
            component: Component to monitor
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        pass
    
    @abstractmethod
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of current metrics
        """
        pass
    
    @abstractmethod
    def check_performance_degradation(self) -> bool:
        """
        Check if performance has degraded significantly.
        
        Returns:
            True if performance degradation detected
        """
        pass
    
    def get_metrics(self) -> List[str]:
        """Get list of continuous monitoring metrics."""
        return ["current_performance", "performance_trend", "degradation_detected"]
    
    def evaluate(self, component: Union["Model", "Policy"], 
                test_data: Union[List[Observation], Trajectory, List[Trajectory]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get current monitoring metrics as evaluation.
        
        Args:
            component: Component to evaluate
            test_data: Test data (may be ignored for continuous monitoring)
            context: Evaluation context
            
        Returns:
            Dictionary of current metrics
        """
        return self.get_current_metrics()
