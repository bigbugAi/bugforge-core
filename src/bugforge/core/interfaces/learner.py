"""
Learner interface - defines the contract for learning and adaptation.

Learners are components that can improve models, policies, or other
components based on feedback and experience.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..primitives import Action, Feedback, Observation, State, Trajectory


class Learner(ABC):
    """
    Abstract base class for all learners in BugForge.
    
    Learners are responsible for improving system components through
    learning from feedback, experience, and data.
    """
    
    @abstractmethod
    def update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update the learning component based on experience.
        
        Args:
            experience: Learning experience (trajectory or single step)
            context: Additional context for learning (optional)
            
        Returns:
            Dictionary of learning metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the learner's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    @abstractmethod
    def get_learning_state(self) -> Dict[str, Any]:
        """
        Get the current learning state.
        
        Returns:
            Dictionary representing the learning state
        """
        pass
    
    def batch_update(self, experiences: List[Union[Trajectory, Tuple[Observation, Action, Feedback]]], 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update based on multiple experiences.
        
        Args:
            experiences: List of learning experiences
            context: Additional context for learning (optional)
            
        Returns:
            Dictionary of aggregated learning metrics
        """
        metrics = {}
        for experience in experiences:
            step_metrics = self.update(experience, context)
            for key, value in step_metrics.items():
                if key in metrics:
                    metrics[key] += value
                else:
                    metrics[key] = value
        
        # Average the metrics
        for key in metrics:
            metrics[key] /= len(experiences)
        
        return metrics
    
    def should_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                     context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if learning should occur for this experience.
        
        Args:
            experience: The learning experience
            context: Additional context (optional)
            
        Returns:
            True if learning should proceed
        """
        return True
    
    def get_update_frequency(self) -> Optional[int]:
        """
        Get the update frequency (updates per N experiences).
        
        Returns:
            Number of experiences between updates, or None for immediate updates
        """
        return None
    
    def reset(self) -> None:
        """
        Reset learner state for new learning sessions.
        
        Called to reset any accumulated learning state for new episodes
        or learning phases.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save learner state to disk.
        
        Args:
            path: Path where to save the learner
        """
        raise NotImplementedError("Learner saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load learner state from disk.
        
        Args:
            path: Path from which to load the learner
        """
        raise NotImplementedError("Learner loading not implemented")
    
    def clone(self) -> "Learner":
        """
        Create a copy of this learner.
        
        Returns:
            A new instance of the same learner with the same parameters
        """
        raise NotImplementedError("Learner cloning not implemented")
    
    def get_learning_rate(self) -> Optional[float]:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate, or None if not applicable
        """
        return None
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate.
        
        Args:
            learning_rate: New learning rate
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(capabilities={self.get_capabilities()})"


class OnlineLearner(Learner):
    """
    Learner that can update incrementally from streaming data.
    
    Supports online learning where updates happen continuously
    as new experiences arrive.
    """
    
    @abstractmethod
    def incremental_update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Perform incremental update from a single experience.
        
        Args:
            experience: Single learning experience
            context: Additional context (optional)
            
        Returns:
            Dictionary of learning metrics
        """
        pass
    
    def update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update using incremental learning.
        
        Args:
            experience: Learning experience
            context: Additional context (optional)
            
        Returns:
            Dictionary of learning metrics
        """
        return self.incremental_update(experience, context)
    
    def get_buffer_size(self) -> Optional[int]:
        """
        Get the size of the experience buffer.
        
        Returns:
            Size of the experience buffer, or None if no buffering
        """
        return None
    
    def clear_buffer(self) -> None:
        """Clear the experience buffer."""
        pass


class OfflineLearner(Learner):
    """
    Learner that trains on batches of pre-collected data.
    
    Supports offline learning where training happens on
    fixed datasets or experience buffers.
    """
    
    @abstractmethod
    def train_batch(self, experiences: List[Union[Trajectory, Tuple[Observation, Action, Feedback]]], 
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train on a batch of experiences.
        
        Args:
            experiences: Batch of learning experiences
            context: Additional context (optional)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def update(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Add experience to buffer (actual training happens in train_batch).
        
        Args:
            experience: Learning experience
            context: Additional context (optional)
            
        Returns:
            Empty metrics (actual training happens in train_batch)
        """
        self.add_to_buffer(experience)
        return {}
    
    @abstractmethod
    def add_to_buffer(self, experience: Union[Trajectory, Tuple[Observation, Action, Feedback]]) -> None:
        """
        Add experience to the training buffer.
        
        Args:
            experience: Learning experience to add
        """
        pass
    
    @abstractmethod
    def train_epoch(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train for one epoch on buffered experiences.
        
        Args:
            context: Additional context (optional)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def get_buffer_size(self) -> int:
        """
        Get the current size of the training buffer.
        
        Returns:
            Number of experiences in the buffer
        """
        return 0
    
    def should_train(self) -> bool:
        """
        Determine if training should occur.
        
        Returns:
            True if training conditions are met
        """
        return self.get_buffer_size() >= self.get_min_buffer_size()
    
    def get_min_buffer_size(self) -> int:
        """
        Get minimum buffer size required for training.
        
        Returns:
            Minimum number of experiences needed
        """
        return 1


class ReinforcementLearner(Learner):
    """
    Learner specialized for reinforcement learning.
    
    Provides reinforcement learning specific functionality including
    value function learning, policy gradients, etc.
    """
    
    @abstractmethod
    def compute_returns(self, trajectory: Trajectory, 
                       gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns for a trajectory.
        
        Args:
            trajectory: The trajectory to process
            gamma: Discount factor
            
        Returns:
            List of discounted returns for each step
        """
        pass
    
    @abstractmethod
    def compute_advantages(self, trajectory: Trajectory, 
                          gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> List[float]:
        """
        Compute advantage estimates using GAE.
        
        Args:
            trajectory: The trajectory to process
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            
        Returns:
            List of advantage estimates for each step
        """
        pass
    
    def get_value_function(self) -> Optional["Model"]:
        """
        Get the value function model.
        
        Returns:
            Value function model, or None if not used
        """
        return None
    
    def set_value_function(self, value_model: "Model") -> None:
        """
        Set the value function model.
        
        Args:
            value_model: Value function model to use
        """
        pass
    
    def get_policy(self) -> Optional["Policy"]:
        """
        Get the policy being learned.
        
        Returns:
            Policy being learned, or None if not applicable
        """
        return None
    
    def set_policy(self, policy: "Policy") -> None:
        """
        Set the policy to learn.
        
        Args:
            policy: Policy to learn
        """
        pass


class SupervisedLearner(Learner):
    """
    Learner specialized for supervised learning.
    
    Provides supervised learning specific functionality including
    classification, regression, and sequence learning.
    """
    
    @abstractmethod
    def predict(self, observation: Observation, 
               context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make predictions for supervised learning tasks.
        
        Args:
            observation: Input observation
            context: Additional context (optional)
            
        Returns:
            Predicted action/output
        """
        pass
    
    @abstractmethod
    def compute_loss(self, prediction: Action, target: Action) -> float:
        """
        Compute loss between prediction and target.
        
        Args:
            prediction: Model prediction
            target: Target action/output
            
        Returns:
            Loss value
        """
        pass
    
    def evaluate(self, observations: List[Observation], 
                targets: List[Action]) -> Dict[str, float]:
        """
        Evaluate performance on a test set.
        
        Args:
            observations: Test observations
            targets: Target actions
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = [self.predict(obs) for obs in observations]
        losses = [self.compute_loss(pred, target) for pred, target in zip(predictions, targets)]
        
        return {
            "loss": sum(losses) / len(losses),
            "num_samples": len(losses)
        }
