"""
EvaluationMetrics implementation for performance assessment.

EvaluationMetrics provides standardized metrics for evaluating
models, policies, and learning systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import math

from ..primitives import Action, Feedback, Observation, State, Trajectory


class EvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Provides standardized interfaces for computing various
    performance metrics across different domains.
    """
    
    @abstractmethod
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """
        Compute the metric value.
        
        Args:
            predictions: List of predicted actions
            targets: List of target actions or feedback
            **kwargs: Additional parameters
            
        Returns:
            Computed metric value
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the metric name.
        
        Returns:
            Metric name string
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get metric description.
        
        Returns:
            Description of what the metric measures
        """
        pass
    
    def is_higher_better(self) -> bool:
        """
        Check if higher values are better.
        
        Returns:
            True if higher values indicate better performance
        """
        return True
    
    def get_range(self) -> Tuple[float, float]:
        """
        Get the valid range for this metric.
        
        Returns:
            Tuple of (min_value, max_value)
        """
        return (float('-inf'), float('inf'))


class AccuracyMetric(EvaluationMetric):
    """Accuracy metric for classification tasks."""
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute accuracy as correct predictions / total predictions."""
        if not predictions or not targets or len(predictions) != len(targets):
            return 0.0
        
        correct = 0
        for pred, target in zip(predictions, targets):
            if self._is_correct(pred, target):
                correct += 1
        
        return correct / len(predictions)
    
    def get_name(self) -> str:
        return "accuracy"
    
    def get_description(self) -> str:
        return "Ratio of correct predictions to total predictions"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def _is_correct(self, prediction: Action, target: Union[Action, Feedback]) -> bool:
        """Check if prediction matches target."""
        if isinstance(target, Feedback):
            # For feedback, check if action type matches expected
            return prediction.action_type == str(target.value)
        else:
            return prediction.action_type == target.action_type


class PrecisionMetric(EvaluationMetric):
    """Precision metric for classification tasks."""
    
    def __init__(self, positive_class: str = None):
        """
        Initialize precision metric.
        
        Args:
            positive_class: The class to consider as positive
        """
        self.positive_class = positive_class
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute precision as TP / (TP + FP)."""
        if not predictions or not targets or len(predictions) != len(targets):
            return 0.0
        
        true_positives = 0
        false_positives = 0
        
        for pred, target in zip(predictions, targets):
            pred_positive = self._is_positive(pred)
            target_positive = self._is_positive_target(target)
            
            if pred_positive and target_positive:
                true_positives += 1
            elif pred_positive and not target_positive:
                false_positives += 1
        
        if true_positives + false_positives == 0:
            return 0.0
        
        return true_positives / (true_positives + false_positives)
    
    def get_name(self) -> str:
        return "precision"
    
    def get_description(self) -> str:
        return "Ratio of true positives to total positive predictions"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def _is_positive(self, action: Action) -> bool:
        """Check if action is positive class."""
        if self.positive_class:
            return action.action_type == self.positive_class
        else:
            # Default: consider non-no-op as positive
            return not action.is_no_op()
    
    def _is_positive_target(self, target: Union[Action, Feedback]) -> bool:
        """Check if target is positive class."""
        if isinstance(target, Feedback):
            return target.is_positive()
        else:
            if self.positive_class:
                return target.action_type == self.positive_class
            else:
                return not target.is_no_op()


class RecallMetric(EvaluationMetric):
    """Recall metric for classification tasks."""
    
    def __init__(self, positive_class: str = None):
        """
        Initialize recall metric.
        
        Args:
            positive_class: The class to consider as positive
        """
        self.positive_class = positive_class
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute recall as TP / (TP + FN)."""
        if not predictions or not targets or len(predictions) != len(targets):
            return 0.0
        
        true_positives = 0
        false_negatives = 0
        
        for pred, target in zip(predictions, targets):
            pred_positive = self._is_positive(pred)
            target_positive = self._is_positive_target(target)
            
            if pred_positive and target_positive:
                true_positives += 1
            elif not pred_positive and target_positive:
                false_negatives += 1
        
        if true_positives + false_negatives == 0:
            return 0.0
        
        return true_positives / (true_positives + false_negatives)
    
    def get_name(self) -> str:
        return "recall"
    
    def get_description(self) -> str:
        return "Ratio of true positives to total actual positives"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)
    
    def _is_positive(self, action: Action) -> bool:
        """Check if action is positive class."""
        if self.positive_class:
            return action.action_type == self.positive_class
        else:
            return not action.is_no_op()
    
    def _is_positive_target(self, target: Union[Action, Feedback]) -> bool:
        """Check if target is positive class."""
        if isinstance(target, Feedback):
            return target.is_positive()
        else:
            if self.positive_class:
                return target.action_type == self.positive_class
            else:
                return not target.is_no_op()


class F1ScoreMetric(EvaluationMetric):
    """F1-score metric combining precision and recall."""
    
    def __init__(self, positive_class: str = None):
        """
        Initialize F1-score metric.
        
        Args:
            positive_class: The class to consider as positive
        """
        self.precision_metric = PrecisionMetric(positive_class)
        self.recall_metric = RecallMetric(positive_class)
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute F1-score as 2 * (precision * recall) / (precision + recall)."""
        precision = self.precision_metric.compute(predictions, targets)
        recall = self.recall_metric.compute(predictions, targets)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def get_name(self) -> str:
        return "f1_score"
    
    def get_description(self) -> str:
        return "Harmonic mean of precision and recall"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)


class MeanSquaredErrorMetric(EvaluationMetric):
    """Mean squared error metric for regression tasks."""
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute mean squared error."""
        if not predictions or not targets or len(predictions) != len(targets):
            return float('inf')
        
        squared_errors = []
        for pred, target in zip(predictions, targets):
            pred_value = self._extract_numeric_value(pred)
            target_value = self._extract_numeric_value(target)
            
            if pred_value is not None and target_value is not None:
                squared_errors.append((pred_value - target_value) ** 2)
        
        if not squared_errors:
            return float('inf')
        
        return sum(squared_errors) / len(squared_errors)
    
    def get_name(self) -> str:
        return "mse"
    
    def get_description(self) -> str:
        return "Mean squared error between predictions and targets"
    
    def is_higher_better(self) -> bool:
        return False  # Lower MSE is better
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, float('inf'))
    
    def _extract_numeric_value(self, item: Union[Action, Feedback]) -> Optional[float]:
        """Extract numeric value from action or feedback."""
        if isinstance(item, Feedback) and isinstance(item.value, (int, float)):
            return float(item.value)
        elif isinstance(item, Action) and isinstance(item.data, (int, float)):
            return float(item.data)
        elif isinstance(item, Action) and isinstance(item.data, dict):
            # Try to find numeric value in data dictionary
            for value in item.data.values():
                if isinstance(value, (int, float)):
                    return float(value)
        
        return None


class RewardMetric(EvaluationMetric):
    """Reward metric for reinforcement learning tasks."""
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute total reward from feedback."""
        if not targets:
            return 0.0
        
        total_reward = 0.0
        for target in targets:
            if isinstance(target, Feedback):
                if isinstance(target.value, (int, float)):
                    total_reward += float(target.value)
                elif isinstance(target.value, dict) and "reward" in target.value:
                    total_reward += float(target.value["reward"])
        
        return total_reward
    
    def get_name(self) -> str:
        return "total_reward"
    
    def get_description(self) -> str:
        return "Total reward accumulated over the evaluation period"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))


class AverageRewardMetric(EvaluationMetric):
    """Average reward metric for reinforcement learning tasks."""
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute average reward per step."""
        if not targets:
            return 0.0
        
        total_reward = 0.0
        reward_count = 0
        
        for target in targets:
            if isinstance(target, Feedback):
                if isinstance(target.value, (int, float)):
                    total_reward += float(target.value)
                    reward_count += 1
                elif isinstance(target.value, dict) and "reward" in target.value:
                    total_reward += float(target.value["reward"])
                    reward_count += 1
        
        return total_reward / reward_count if reward_count > 0 else 0.0
    
    def get_name(self) -> str:
        return "average_reward"
    
    def get_description(self) -> str:
        return "Average reward per step"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (float('-inf'), float('inf'))


class SuccessRateMetric(EvaluationMetric):
    """Success rate metric for task completion."""
    
    def compute(self, 
                predictions: List[Action],
                targets: List[Union[Action, Feedback]],
                **kwargs) -> float:
        """Compute success rate based on positive feedback."""
        if not targets:
            return 0.0
        
        successful_steps = 0
        total_steps = 0
        
        for target in targets:
            if isinstance(target, Feedback):
                total_steps += 1
                if target.is_positive():
                    successful_steps += 1
        
        return successful_steps / total_steps if total_steps > 0 else 0.0
    
    def get_name(self) -> str:
        return "success_rate"
    
    def get_description(self) -> str:
        return "Ratio of successful steps to total steps"
    
    def is_higher_better(self) -> bool:
        return True
    
    def get_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)


class EvaluationMetrics:
    """
    Collection of evaluation metrics with computation utilities.
    
    Provides a unified interface for computing multiple metrics
    and aggregating results.
    """
    
    def __init__(self, metrics: Optional[List[EvaluationMetric]] = None):
        """
        Initialize evaluation metrics collection.
        
        Args:
            metrics: List of metrics to include
        """
        self.metrics = metrics or self._default_metrics()
        self._metric_cache: Dict[str, EvaluationMetric] = {}
        
        # Cache metrics by name for fast lookup
        for metric in self.metrics:
            self._metric_cache[metric.get_name()] = metric
    
    def compute_all(self, 
                   predictions: List[Action],
                   targets: List[Union[Action, Feedback]],
                   **kwargs) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            predictions: List of predicted actions
            targets: List of target actions or feedback
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric name to value
        """
        results = {}
        
        for metric in self.metrics:
            try:
                value = metric.compute(predictions, targets, **kwargs)
                results[metric.get_name()] = value
            except Exception as e:
                # Log error and continue
                results[metric.get_name()] = None
        
        return results
    
    def compute_metric(self, 
                      metric_name: str,
                      predictions: List[Action],
                      targets: List[Union[Action, Feedback]],
                      **kwargs) -> Optional[float]:
        """
        Compute a specific metric.
        
        Args:
            metric_name: Name of the metric to compute
            predictions: List of predicted actions
            targets: List of target actions or feedback
            **kwargs: Additional parameters
            
        Returns:
            Metric value or None if metric not found
        """
        if metric_name not in self._metric_cache:
            return None
        
        try:
            return self._metric_cache[metric_name].compute(predictions, targets, **kwargs)
        except Exception:
            return None
    
    def get_metric_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all metrics.
        
        Returns:
            List of metric information dictionaries
        """
        info = []
        
        for metric in self.metrics:
            info.append({
                "name": metric.get_name(),
                "description": metric.get_description(),
                "higher_better": metric.is_higher_better(),
                "range": metric.get_range()
            })
        
        return info
    
    def add_metric(self, metric: EvaluationMetric) -> None:
        """
        Add a new metric to the collection.
        
        Args:
            metric: The metric to add
        """
        self.metrics.append(metric)
        self._metric_cache[metric.get_name()] = metric
    
    def remove_metric(self, metric_name: str) -> bool:
        """
        Remove a metric by name.
        
        Args:
            metric_name: Name of the metric to remove
            
        Returns:
            True if metric was removed
        """
        if metric_name not in self._metric_cache:
            return False
        
        # Remove from cache
        del self._metric_cache[metric_name]
        
        # Remove from list
        self.metrics = [m for m in self.metrics if m.get_name() != metric_name]
        
        return True
    
    def _default_metrics(self) -> List[EvaluationMetric]:
        """Get default set of metrics."""
        return [
            AccuracyMetric(),
            PrecisionMetric(),
            RecallMetric(),
            F1ScoreMetric(),
            MeanSquaredErrorMetric(),
            RewardMetric(),
            AverageRewardMetric(),
            SuccessRateMetric()
        ]
    
    def aggregate_results(self, 
                        results_list: List[Dict[str, float]],
                        aggregation_method: str = "mean") -> Dict[str, float]:
        """
        Aggregate results from multiple evaluations.
        
        Args:
            results_list: List of result dictionaries
            aggregation_method: How to aggregate ("mean", "median", "std")
            
        Returns:
            Aggregated results dictionary
        """
        if not results_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_metrics = set()
        for results in results_list:
            all_metrics.update(results.keys())
        
        # Aggregate each metric
        for metric_name in all_metrics:
            values = []
            for results in results_list:
                if metric_name in results and results[metric_name] is not None:
                    values.append(results[metric_name])
            
            if not values:
                aggregated[metric_name] = None
                continue
            
            if aggregation_method == "mean":
                aggregated[metric_name] = sum(values) / len(values)
            elif aggregation_method == "median":
                sorted_values = sorted(values)
                n = len(sorted_values)
                if n % 2 == 0:
                    aggregated[metric_name] = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
                else:
                    aggregated[metric_name] = sorted_values[n//2]
            elif aggregation_method == "std":
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                aggregated[metric_name] = math.sqrt(variance)
            else:
                aggregated[metric_name] = None
        
        return aggregated
