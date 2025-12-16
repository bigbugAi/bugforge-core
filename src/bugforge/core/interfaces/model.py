"""
Model interface - defines the contract for predictive models.

Models are components that can make predictions or generate outputs
based on inputs, serving as the core reasoning engine.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Action, Observation, State


class Model(ABC):
    """
    Abstract base class for all models in BugForge.
    
    Models are responsible for generating predictions, actions, or
    other outputs based on input observations and context.
    """
    
    @abstractmethod
    def predict(self, observation: Observation, context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Generate a prediction or action based on an observation.
        
        Args:
            observation: The input observation to process
            context: Additional context for prediction (optional)
            
        Returns:
            Action representing the model's prediction/decision
        """
        pass
    
    @abstractmethod
    def batch_predict(self, observations: List[Observation], 
                     context: Optional[Dict[str, Any]] = None) -> List[Action]:
        """
        Generate predictions for multiple observations.
        
        Args:
            observations: List of input observations
            context: Additional context for predictions (optional)
            
        Returns:
            List of actions corresponding to each observation
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the model's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
    
    def reset(self) -> None:
        """
        Reset model state for new episodes.
        
        Called at the beginning of new episodes to clear any
        accumulated state from previous episodes.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model state to disk.
        
        Args:
            path: Path where to save the model
        """
        raise NotImplementedError("Model saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load model state from disk.
        
        Args:
            path: Path from which to load the model
        """
        raise NotImplementedError("Model loading not implemented")
    
    def clone(self) -> "Model":
        """
        Create a copy of this model.
        
        Returns:
            A new instance of the same model with the same parameters
        """
        raise NotImplementedError("Model cloning not implemented")
    
    def get_parameter_count(self) -> Optional[int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Number of parameters, or None if not applicable
        """
        return None
    
    def get_memory_usage(self) -> Optional[Dict[str, Any]]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage details, or None if not available
        """
        return None
    
    def validate_input(self, observation: Observation) -> bool:
        """
        Validate that an observation is compatible with this model.
        
        Args:
            observation: The observation to validate
            
        Returns:
            True if the observation is valid for this model
        """
        return True
    
    def validate_output(self, action: Action) -> bool:
        """
        Validate that an action is compatible with this model's output format.
        
        Args:
            action: The action to validate
            
        Returns:
            True if the action is valid for this model
        """
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(capabilities={self.get_capabilities()})"


class ProbabilisticModel(Model):
    """
    Model that can provide probability distributions over actions.
    
    Extends the base Model interface to support uncertainty estimation
    and probabilistic reasoning.
    """
    
    @abstractmethod
    def predict_distribution(self, observation: Observation, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Predict probability distribution over possible actions.
        
        Args:
            observation: The input observation
            context: Additional context (optional)
            
        Returns:
            Dictionary mapping action identifiers to probabilities
        """
        pass
    
    @abstractmethod
    def sample_action(self, observation: Observation, 
                     context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Sample an action from the model's probability distribution.
        
        Args:
            observation: The input observation
            context: Additional context (optional)
            
        Returns:
            Sampled action from the probability distribution
        """
        pass
    
    def get_uncertainty(self, observation: Observation, 
                       context: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Get uncertainty estimate for the prediction.
        
        Args:
            observation: The input observation
            context: Additional context (optional)
            
        Returns:
            Uncertainty estimate (e.g., entropy), or None if not available
        """
        return None


class GenerativeModel(Model):
    """
    Model that can generate new content or sequences.
    
    Extends the base Model interface for generative capabilities
    like text generation, sequence prediction, etc.
    """
    
    @abstractmethod
    def generate(self, prompt: Union[str, Observation], 
                max_length: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None) -> Union[str, Action]:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Input prompt or observation
            max_length: Maximum generation length (optional)
            context: Additional context (optional)
            
        Returns:
            Generated content or action
        """
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[Union[str, Observation]], 
                      max_length: Optional[int] = None,
                      context: Optional[Dict[str, Any]] = None) -> List[Union[str, Action]]:
        """
        Generate content for multiple prompts.
        
        Args:
            prompts: List of input prompts or observations
            max_length: Maximum generation length (optional)
            context: Additional context (optional)
            
        Returns:
            List of generated content or actions
        """
        pass
    
    def set_generation_params(self, **params) -> None:
        """
        Set generation parameters.
        
        Args:
            **params: Generation parameters (temperature, top_k, etc.)
        """
        pass
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get current generation parameters.
        
        Returns:
            Dictionary of current generation parameters
        """
        return {}
