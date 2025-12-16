"""
UpdateHook implementation for learning and adaptation.

UpdateHook provides mechanisms for updating models, policies,
and other components based on experience and feedback.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from ..interfaces.learner import Learner
from ..primitives import Action, Feedback, Observation, State, Trajectory


class UpdateHook(ABC):
    """
    Abstract base class for update hooks in BugForge.
    
    Update hooks provide mechanisms for updating components
    based on experience, with various triggers and strategies.
    """
    
    @abstractmethod
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if an update should occur for this experience.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if an update should occur
        """
        pass
    
    @abstractmethod
    def execute_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                      learner: Learner,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute the update using the provided learner.
        
        Args:
            experience: The learning experience
            learner: The learner to use for updating
            context: Additional context (optional)
            
        Returns:
            Dictionary of update metrics
        """
        pass
    
    @abstractmethod
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about update behavior.
        
        Returns:
            Dictionary with update statistics
        """
        pass
    
    def reset(self) -> None:
        """Reset update hook state for new sessions."""
        pass


class FrequencyUpdateHook(UpdateHook):
    """
    Update hook that triggers updates at fixed frequency intervals.
    
    Performs updates every N experiences or time intervals.
    """
    
    def __init__(self, update_frequency: int, 
                 max_buffer_size: Optional[int] = None):
        """
        Initialize frequency update hook.
        
        Args:
            update_frequency: Number of experiences between updates
            max_buffer_size: Maximum buffer size for experiences
        """
        self.update_frequency = update_frequency
        self.max_buffer_size = max_buffer_size
        
        self._experience_buffer: List[Union[Trajectory, Tuple[Observation, Action, Feedback]]] = []
        self._experience_count = 0
        self._update_count = 0
        self._total_updates = 0
        self._last_update_time: Optional[datetime] = None
    
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if update should occur based on frequency.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if update frequency has been reached
        """
        self._experience_count += 1
        
        # Add to buffer
        if self.max_buffer_size is None or len(self._experience_buffer) < self.max_buffer_size:
            self._experience_buffer.append(experience)
        else:
            # Replace oldest if buffer is full
            self._experience_buffer[self._experience_count % self.max_buffer_size] = experience
        
        return self._experience_count % self.update_frequency == 0
    
    def execute_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                      learner: Learner,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute update using buffered experiences.
        
        Args:
            experience: The current experience (may not be used)
            learner: The learner to update
            context: Additional context (optional)
            
        Returns:
            Dictionary of update metrics
        """
        # Use buffered experiences for batch update
        if self._experience_buffer:
            metrics = learner.batch_update(self._experience_buffer, context)
        else:
            metrics = learner.update(experience, context)
        
        self._total_updates += 1
        self._last_update_time = datetime.now()
        
        # Clear buffer if it was for this update only
        if self.max_buffer_size is None:
            self._experience_buffer.clear()
        
        return metrics
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get frequency update statistics.
        
        Returns:
            Dictionary with update statistics
        """
        return {
            "update_frequency": self.update_frequency,
            "experience_count": self._experience_count,
            "update_count": self._total_updates,
            "buffer_size": len(self._experience_buffer),
            "max_buffer_size": self.max_buffer_size,
            "last_update_time": self._last_update_time,
            "update_efficiency": self._total_updates / max(1, self._experience_count)
        }
    
    def reset(self) -> None:
        """Reset frequency update hook state."""
        self._experience_buffer.clear()
        self._experience_count = 0
        self._update_count = 0


class ThresholdUpdateHook(UpdateHook):
    """
    Update hook that triggers updates based on performance thresholds.
    
    Performs updates when performance metrics fall below thresholds.
    """
    
    def __init__(self, performance_threshold: float,
                 window_size: int = 100,
                 metric_name: str = "reward"):
        """
        Initialize threshold update hook.
        
        Args:
            performance_threshold: Performance threshold for triggering updates
            window_size: Size of performance window for evaluation
            metric_name: Name of performance metric to track
        """
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self.metric_name = metric_name
        
        self._performance_window: List[float] = []
        self._update_count = 0
        self._last_update_time: Optional[datetime] = None
    
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if update should occur based on performance threshold.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if performance is below threshold
        """
        # Extract performance metric
        metric_value = self._extract_metric(experience)
        
        if metric_value is not None:
            self._performance_window.append(metric_value)
            
            # Keep window size limited
            if len(self._performance_window) > self.window_size:
                self._performance_window.pop(0)
            
            # Check if average performance is below threshold
            if len(self._performance_window) >= self.window_size:
                avg_performance = sum(self._performance_window) / len(self._performance_window)
                return avg_performance < self.performance_threshold
        
        return False
    
    def execute_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                      learner: Learner,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute update when performance is poor.
        
        Args:
            experience: The learning experience
            learner: The learner to update
            context: Additional context (optional)
            
        Returns:
            Dictionary of update metrics
        """
        metrics = learner.update(experience, context)
        
        self._update_count += 1
        self._last_update_time = datetime.now()
        
        return metrics
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get threshold update statistics.
        
        Returns:
            Dictionary with update statistics
        """
        current_performance = (
            sum(self._performance_window) / len(self._performance_window)
            if self._performance_window else None
        )
        
        return {
            "performance_threshold": self.performance_threshold,
            "window_size": self.window_size,
            "metric_name": self.metric_name,
            "current_performance": current_performance,
            "performance_window_size": len(self._performance_window),
            "update_count": self._update_count,
            "last_update_time": self._last_update_time
        }
    
    def _extract_metric(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]]) -> Optional[float]:
        """Extract performance metric from experience."""
        if isinstance(experience, tuple) and len(experience) == 3:
            _, _, feedback = experience
            if feedback and hasattr(feedback, 'value'):
                if isinstance(feedback.value, (int, float)):
                    return float(feedback.value)
                elif isinstance(feedback.value, dict) and self.metric_name in feedback.value:
                    return float(feedback.value[self.metric_name])
        elif isinstance(experience, Trajectory):
            return experience.total_reward()
        
        return None
    
    def reset(self) -> None:
        """Reset threshold update hook state."""
        self._performance_window.clear()
        self._update_count = 0


class AdaptiveUpdateHook(UpdateHook):
    """
    Update hook that adapts update frequency based on learning progress.
    
    Increases or decreases update frequency based on performance trends.
    """
    
    def __init__(self, initial_frequency: int = 10,
                 min_frequency: int = 1,
                 max_frequency: int = 1000,
                 adaptation_factor: float = 2.0,
                 performance_window: int = 50):
        """
        Initialize adaptive update hook.
        
        Args:
            initial_frequency: Starting update frequency
            min_frequency: Minimum update frequency
            max_frequency: Maximum update frequency
            adaptation_factor: Factor for frequency adaptation
            performance_window: Window size for performance evaluation
        """
        self.initial_frequency = initial_frequency
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.adaptation_factor = adaptation_factor
        self.performance_window = performance_window
        
        self._current_frequency = initial_frequency
        self._experience_count = 0
        self._update_count = 0
        self._performance_history: List[float] = []
        self._last_adaptation_time = datetime.now()
    
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if update should occur with adaptive frequency.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if adaptive frequency threshold is reached
        """
        self._experience_count += 1
        
        # Extract and track performance
        performance = self._extract_performance(experience)
        if performance is not None:
            self._performance_history.append(performance)
            if len(self._performance_history) > self.performance_window:
                self._performance_history.pop(0)
            
            # Adapt frequency based on performance trend
            self._adapt_frequency()
        
        return self._experience_count % self._current_frequency == 0
    
    def execute_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                      learner: Learner,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute update with current adaptive frequency.
        
        Args:
            experience: The learning experience
            learner: The learner to update
            context: Additional context (optional)
            
        Returns:
            Dictionary of update metrics
        """
        metrics = learner.update(experience, context)
        
        self._update_count += 1
        
        return metrics
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get adaptive update statistics.
        
        Returns:
            Dictionary with update statistics
        """
        current_performance = (
            sum(self._performance_history) / len(self._performance_history)
            if self._performance_history else None
        )
        
        return {
            "current_frequency": self._current_frequency,
            "initial_frequency": self.initial_frequency,
            "min_frequency": self.min_frequency,
            "max_frequency": self.max_frequency,
            "experience_count": self._experience_count,
            "update_count": self._update_count,
            "current_performance": current_performance,
            "performance_window_size": len(self._performance_history),
            "adaptation_factor": self.adaptation_factor
        }
    
    def _extract_performance(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]]) -> Optional[float]:
        """Extract performance metric from experience."""
        if isinstance(experience, tuple) and len(experience) == 3:
            _, _, feedback = experience
            if feedback and hasattr(feedback, 'value') and isinstance(feedback.value, (int, float)):
                return float(feedback.value)
        elif isinstance(experience, Trajectory):
            return experience.total_reward()
        
        return None
    
    def _adapt_frequency(self) -> None:
        """Adapt update frequency based on performance trend."""
        if len(self._performance_history) < self.performance_window // 2:
            return
        
        # Calculate performance trend
        recent_performance = sum(self._performance_history[-self.performance_window//2:]) / (self.performance_window // 2)
        older_performance = sum(self._performance_history[:self.performance_window//2]) / (self.performance_window // 2)
        
        performance_ratio = recent_performance / older_performance if older_performance != 0 else 1.0
        
        # Adapt frequency
        if performance_ratio < 0.9:  # Performance declining
            # Increase update frequency
            self._current_frequency = max(
                self.min_frequency,
                int(self._current_frequency / self.adaptation_factor)
            )
        elif performance_ratio > 1.1:  # Performance improving
            # Decrease update frequency
            self._current_frequency = min(
                self.max_frequency,
                int(self._current_frequency * self.adaptation_factor)
            )
        
        self._last_adaptation_time = datetime.now()
    
    def reset(self) -> None:
        """Reset adaptive update hook state."""
        self._current_frequency = self.initial_frequency
        self._experience_count = 0
        self._update_count = 0
        self._performance_history.clear()


class CompositeUpdateHook(UpdateHook):
    """
    Update hook that combines multiple update strategies.
    
    Coordinates multiple update hooks with different strategies.
    """
    
    def __init__(self, hooks: List[UpdateHook], 
                 combination_strategy: str = "any"):
        """
        Initialize composite update hook.
        
        Args:
            hooks: List of update hooks to combine
            combination_strategy: How to combine hook decisions ("any", "all", "majority")
        """
        self.hooks = hooks
        self.combination_strategy = combination_strategy
        
        self._hook_statistics: Dict[str, Dict[str, Any]] = {}
        self._total_updates = 0
    
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if update should occur using combination strategy.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if combination strategy indicates update
        """
        decisions = [hook.should_update(experience, context) for hook in self.hooks]
        
        if self.combination_strategy == "any":
            return any(decisions)
        elif self.combination_strategy == "all":
            return all(decisions)
        elif self.combination_strategy == "majority":
            return sum(decisions) > len(decisions) / 2
        else:
            return False
    
    def execute_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                      learner: Learner,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute update using all applicable hooks.
        
        Args:
            experience: The learning experience
            learner: The learner to update
            context: Additional context (optional)
            
        Returns:
            Dictionary of combined update metrics
        """
        all_metrics = {}
        
        for hook in self.hooks:
            if hook.should_update(experience, context):
                hook_metrics = hook.execute_update(experience, learner, context)
                
                # Combine metrics
                for key, value in hook_metrics.items():
                    if key in all_metrics:
                        all_metrics[key] += value
                    else:
                        all_metrics[key] = value
        
        self._total_updates += 1
        
        return all_metrics
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """
        Get composite update statistics.
        
        Returns:
            Dictionary with composite statistics
        """
        hook_stats = {}
        for i, hook in enumerate(self.hooks):
            hook_name = f"hook_{i}_{hook.__class__.__name__}"
            hook_stats[hook_name] = hook.get_update_statistics()
        
        return {
            "combination_strategy": self.combination_strategy,
            "num_hooks": len(self.hooks),
            "total_updates": self._total_updates,
            "hook_statistics": hook_stats
        }
    
    def reset(self) -> None:
        """Reset all update hooks."""
        for hook in self.hooks:
            hook.reset()
        self._total_updates = 0
