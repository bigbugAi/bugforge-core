"""
TrainingLoop implementation for structured learning processes.

TrainingLoop provides infrastructure for organizing and executing
training processes with various strategies and monitoring.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

from ..interfaces.learner import Learner
from ..interfaces.evaluator import Evaluator
from ..primitives import Action, Feedback, Observation, State, Trajectory


class TrainingLoop(ABC):
    """
    Abstract base class for training loops in BugForge.
    
    Training loops provide structured processes for training
    learners with various strategies and monitoring capabilities.
    """
    
    @abstractmethod
    def train(self, 
              learner: Learner,
              training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
              validation_data: Optional[Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]] = None,
              evaluator: Optional[Evaluator] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Execute the training loop.
        
        Args:
            learner: The learner to train
            training_data: Training data
            validation_data: Validation data (optional)
            evaluator: Evaluator for performance monitoring (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results and metrics
        """
        pass
    
    @abstractmethod
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get the current training state.
        
        Returns:
            Dictionary representing training state
        """
        pass
    
    def stop_training(self) -> None:
        """Stop the training process."""
        pass
    
    def pause_training(self) -> None:
        """Pause the training process."""
        pass
    
    def resume_training(self) -> None:
        """Resume the training process."""
        pass


class EpochTrainingLoop(TrainingLoop):
    """
    Training loop that operates in epochs over the entire dataset.
    
    Performs multiple passes over the training data with validation.
    """
    
    def __init__(self, 
                 num_epochs: int = 10,
                 shuffle_data: bool = True,
                 save_best_model: bool = True,
                 early_stopping_patience: Optional[int] = None):
        """
        Initialize epoch-based training loop.
        
        Args:
            num_epochs: Number of training epochs
            shuffle_data: Whether to shuffle training data each epoch
            save_best_model: Whether to save the best performing model
            early_stopping_patience: Patience for early stopping (optional)
        """
        self.num_epochs = num_epochs
        self.shuffle_data = shuffle_data
        self.save_best_model = save_best_model
        self.early_stopping_patience = early_stopping_patience
        
        self._training_state = {
            "current_epoch": 0,
            "total_epochs": num_epochs,
            "training_loss": [],
            "validation_loss": [],
            "best_validation_loss": float('inf'),
            "epochs_since_improvement": 0,
            "is_training": False,
            "should_stop": False,
            "start_time": None,
            "end_time": None
        }
    
    def train(self, 
              learner: Learner,
              training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
              validation_data: Optional[Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]] = None,
              evaluator: Optional[Evaluator] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train using epoch-based approach.
        
        Args:
            learner: The learner to train
            training_data: Training data
            validation_data: Validation data (optional)
            evaluator: Evaluator for performance monitoring (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        self._training_state["is_training"] = True
        self._training_state["start_time"] = datetime.now()
        
        try:
            for epoch in range(self.num_epochs):
                if self._training_state["should_stop"]:
                    break
                
                self._training_state["current_epoch"] = epoch + 1
                
                # Shuffle data if requested
                if self.shuffle_data:
                    import random
                    random.shuffle(training_data)
                
                # Training phase
                epoch_training_metrics = self._train_epoch(learner, training_data)
                
                # Validation phase
                epoch_validation_metrics = {}
                if validation_data:
                    epoch_validation_metrics = self._validate_epoch(learner, validation_data, evaluator)
                
                # Update training state
                self._update_training_state(epoch_training_metrics, epoch_validation_metrics)
                
                # Check early stopping
                if self._should_early_stop():
                    break
            
        finally:
            self._training_state["is_training"] = False
            self._training_state["end_time"] = datetime.now()
        
        return self._get_training_results()
    
    def _train_epoch(self, learner: Learner, 
                    training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {}
        
        for experience in training_data:
            if self._training_state["should_stop"]:
                break
            
            # Update learner
            metrics = learner.update(experience)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
                else:
                    epoch_metrics[key] = value
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(training_data)
        
        return epoch_metrics
    
    def _validate_epoch(self, learner: Learner,
                       validation_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
                       evaluator: Optional[Evaluator] = None) -> Dict[str, float]:
        """Validate for one epoch."""
        if evaluator:
            return evaluator.evaluate(learner, validation_data)
        else:
            # Simple validation using learner's own evaluation
            validation_metrics = {}
            for experience in validation_data:
                # This would depend on the specific learner implementation
                pass
            return validation_metrics
    
    def _update_training_state(self, training_metrics: Dict[str, float], 
                              validation_metrics: Dict[str, float]) -> None:
        """Update training state with epoch results."""
        # Store training loss
        training_loss = training_metrics.get("loss", 0.0)
        self._training_state["training_loss"].append(training_loss)
        
        # Store validation loss
        validation_loss = validation_metrics.get("loss", 0.0)
        self._training_state["validation_loss"].append(validation_loss)
        
        # Check for improvement
        if validation_loss < self._training_state["best_validation_loss"]:
            self._training_state["best_validation_loss"] = validation_loss
            self._training_state["epochs_since_improvement"] = 0
        else:
            self._training_state["epochs_since_improvement"] += 1
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping should be triggered."""
        if self.early_stopping_patience is None:
            return False
        
        return self._training_state["epochs_since_improvement"] >= self.early_stopping_patience
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive training results."""
        duration = None
        if self._training_state["start_time"] and self._training_state["end_time"]:
            duration = (self._training_state["end_time"] - self._training_state["start_time"]).total_seconds()
        
        return {
            "training_state": self._training_state.copy(),
            "duration_seconds": duration,
            "final_training_loss": self._training_state["training_loss"][-1] if self._training_state["training_loss"] else None,
            "final_validation_loss": self._training_state["validation_loss"][-1] if self._training_state["validation_loss"] else None,
            "best_validation_loss": self._training_state["best_validation_loss"],
            "epochs_completed": self._training_state["current_epoch"],
            "early_stopped": self._should_early_stop()
        }
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return self._training_state.copy()
    
    def stop_training(self) -> None:
        """Stop the training process."""
        self._training_state["should_stop"] = True


class OnlineTrainingLoop(TrainingLoop):
    """
    Training loop for online/incremental learning.
    
    Processes experiences as they arrive with continuous learning.
    """
    
    def __init__(self, 
                 update_frequency: int = 1,
                 evaluation_frequency: int = 100,
                 max_experiences: Optional[int] = None):
        """
        Initialize online training loop.
        
        Args:
            update_frequency: How often to update (every N experiences)
            evaluation_frequency: How often to evaluate (every N experiences)
            max_experiences: Maximum number of experiences to process
        """
        self.update_frequency = update_frequency
        self.evaluation_frequency = evaluation_frequency
        self.max_experiences = max_experiences
        
        self._training_state = {
            "experience_count": 0,
            "update_count": 0,
            "evaluation_count": 0,
            "recent_metrics": [],
            "is_training": False,
            "should_stop": False,
            "start_time": None,
            "end_time": None
        }
    
    def train(self, 
              learner: Learner,
              training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
              validation_data: Optional[Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]] = None,
              evaluator: Optional[Evaluator] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train using online approach.
        
        Args:
            learner: The learner to train
            training_data: Training data (processed sequentially)
            validation_data: Validation data (optional)
            evaluator: Evaluator for performance monitoring (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        self._training_state["is_training"] = True
        self._training_state["start_time"] = datetime.now()
        
        try:
            for i, experience in enumerate(training_data):
                if self._training_state["should_stop"]:
                    break
                
                if self.max_experiences and i >= self.max_experiences:
                    break
                
                self._training_state["experience_count"] = i + 1
                
                # Update learner if frequency is reached
                if i % self.update_frequency == 0:
                    metrics = learner.update(experience)
                    self._training_state["update_count"] += 1
                    self._training_state["recent_metrics"].append(metrics)
                    
                    # Keep only recent metrics
                    if len(self._training_state["recent_metrics"]) > 100:
                        self._training_state["recent_metrics"].pop(0)
                
                # Evaluate if frequency is reached
                if i % self.evaluation_frequency == 0 and validation_data and evaluator:
                    eval_metrics = evaluator.evaluate(learner, validation_data)
                    # Store evaluation metrics as needed
        
        finally:
            self._training_state["is_training"] = False
            self._training_state["end_time"] = datetime.now()
        
        return self._get_training_results()
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get online training results."""
        duration = None
        if self._training_state["start_time"] and self._training_state["end_time"]:
            duration = (self._training_state["end_time"] - self._training_state["start_time"]).total_seconds()
        
        # Calculate average recent metrics
        avg_recent_metrics = {}
        if self._training_state["recent_metrics"]:
            for key in self._training_state["recent_metrics"][0].keys():
                values = [m.get(key, 0) for m in self._training_state["recent_metrics"]]
                avg_recent_metrics[key] = sum(values) / len(values)
        
        return {
            "training_state": self._training_state.copy(),
            "duration_seconds": duration,
            "experiences_processed": self._training_state["experience_count"],
            "updates_performed": self._training_state["update_count"],
            "evaluations_performed": self._training_state["evaluation_count"],
            "average_recent_metrics": avg_recent_metrics
        }
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return self._training_state.copy()
    
    def stop_training(self) -> None:
        """Stop the training process."""
        self._training_state["should_stop"] = True


class CurriculumTrainingLoop(TrainingLoop):
    """
    Training loop that implements curriculum learning.
    
    Gradually increases task difficulty during training.
    """
    
    def __init__(self, 
                 curriculum_stages: List[Dict[str, Any]],
                 stage_transition_criteria: str = "performance"):
        """
        Initialize curriculum training loop.
        
        Args:
            curriculum_stages: List of curriculum stage configurations
            stage_transition_criteria: How to determine when to advance stages
        """
        self.curriculum_stages = curriculum_stages
        self.stage_transition_criteria = stage_transition_criteria
        
        self._training_state = {
            "current_stage": 0,
            "total_stages": len(curriculum_stages),
            "stage_metrics": [],
            "stage_start_times": [],
            "is_training": False,
            "should_stop": False,
            "start_time": None,
            "end_time": None
        }
    
    def train(self, 
              learner: Learner,
              training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
              validation_data: Optional[Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]] = None,
              evaluator: Optional[Evaluator] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train using curriculum approach.
        
        Args:
            learner: The learner to train
            training_data: Training data
            validation_data: Validation data (optional)
            evaluator: Evaluator for performance monitoring (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        self._training_state["is_training"] = True
        self._training_state["start_time"] = datetime.now()
        
        try:
            for stage_idx, stage_config in enumerate(self.curriculum_stages):
                if self._training_state["should_stop"]:
                    break
                
                self._training_state["current_stage"] = stage_idx
                self._training_state["stage_start_times"].append(datetime.now())
                
                # Filter data for current stage
                stage_training_data = self._filter_data_for_stage(training_data, stage_config)
                
                # Train on current stage
                stage_results = self._train_stage(learner, stage_training_data, stage_config, validation_data, evaluator)
                
                self._training_state["stage_metrics"].append(stage_results)
                
                # Check if should advance to next stage
                if not self._should_advance_stage(stage_results, stage_config):
                    break
        
        finally:
            self._training_state["is_training"] = False
            self._training_state["end_time"] = datetime.now()
        
        return self._get_training_results()
    
    def _filter_data_for_stage(self, 
                               training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
                               stage_config: Dict[str, Any]) -> Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]:
        """Filter training data for current curriculum stage."""
        # This would depend on specific curriculum implementation
        # For now, return all data
        return training_data
    
    def _train_stage(self, learner: Learner,
                    stage_training_data: Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]],
                    stage_config: Dict[str, Any],
                    validation_data: Optional[Union[List[Trajectory], List[Tuple[Observation, Action, Feedback]]]] = None,
                    evaluator: Optional[Evaluator] = None) -> Dict[str, Any]:
        """Train on a single curriculum stage."""
        # Use epoch-based training for the stage
        num_epochs = stage_config.get("num_epochs", 10)
        
        stage_metrics = {}
        for epoch in range(num_epochs):
            if self._training_state["should_stop"]:
                break
            
            epoch_metrics = {}
            for experience in stage_training_data:
                metrics = learner.update(experience)
                
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                    else:
                        epoch_metrics[key] = value
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(stage_training_data)
            
            stage_metrics[f"epoch_{epoch}"] = epoch_metrics
        
        return stage_metrics
    
    def _should_advance_stage(self, stage_results: Dict[str, Any], 
                             stage_config: Dict[str, Any]) -> bool:
        """Determine if should advance to next curriculum stage."""
        if self.stage_transition_criteria == "performance":
            # Check if performance threshold is met
            target_performance = stage_config.get("target_performance")
            if target_performance is None:
                return True  # No target, always advance
            
            # Get final epoch metrics
            final_epoch_key = f"epoch_{max(int(k.split('_')[1]) for k in stage_results.keys() if k.startswith('epoch_'))}"
            final_metrics = stage_results[final_epoch_key]
            
            performance = final_metrics.get("reward", 0)  # Assume reward metric
            return performance >= target_performance
        
        return True  # Default: always advance
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get curriculum training results."""
        duration = None
        if self._training_state["start_time"] and self._training_state["end_time"]:
            duration = (self._training_state["end_time"] - self._training_state["start_time"]).total_seconds()
        
        return {
            "training_state": self._training_state.copy(),
            "duration_seconds": duration,
            "stages_completed": self._training_state["current_stage"] + 1,
            "total_stages": self._training_state["total_stages"],
            "stage_metrics": self._training_state["stage_metrics"]
        }
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return self._training_state.copy()
    
    def stop_training(self) -> None:
        """Stop the training process."""
        self._training_state["should_stop"] = True
