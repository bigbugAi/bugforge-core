"""
Policy interface - defines the contract for decision-making policies.

Policies are components that map observations to actions, implementing
the decision logic for intelligent agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..primitives import Action, Observation, State


class Policy(ABC):
    """
    Abstract base class for all policies in BugForge.
    
    Policies are responsible for making decisions by mapping
    observations to actions based on learned or programmed logic.
    """
    
    @abstractmethod
    def decide(self, observation: Observation, 
               context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make a decision based on an observation.
        
        Args:
            observation: The current observation
            context: Additional context for decision making (optional)
            
        Returns:
            Action representing the policy's decision
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> "ActionSpace":
        """
        Get the action space this policy operates in.
        
        Returns:
            ActionSpace defining valid actions
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> "ObservationSpace":
        """
        Get the observation space this policy expects.
        
        Returns:
            ObservationSpace defining valid observations
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the policy's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    def batch_decide(self, observations: List[Observation], 
                    context: Optional[Dict[str, Any]] = None) -> List[Action]:
        """
        Make decisions for multiple observations.
        
        Args:
            observations: List of current observations
            context: Additional context for decision making (optional)
            
        Returns:
            List of actions corresponding to each observation
        """
        return [self.decide(obs, context) for obs in observations]
    
    def evaluate_actions(self, observation: Observation, 
                       actions: List[Action]) -> List[float]:
        """
        Evaluate a set of possible actions for an observation.
        
        Args:
            observation: The current observation
            actions: List of possible actions to evaluate
            
        Returns:
            List of scores/preferences for each action
        """
        return [0.0] * len(actions)
    
    def get_action_probabilities(self, observation: Observation) -> Dict[str, float]:
        """
        Get probability distribution over actions (for stochastic policies).
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary mapping action identifiers to probabilities
        """
        # Default: deterministic policy returns 1.0 for chosen action
        action = self.decide(observation)
        return {action.action_id.hex: 1.0}
    
    def reset(self) -> None:
        """
        Reset policy state for new episodes.
        
        Called at the beginning of new episodes to clear any
        accumulated state from previous episodes.
        """
        pass
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the policy's internal context.
        
        Args:
            context: New context information
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the policy's current internal state.
        
        Returns:
            Dictionary representing the policy's state
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the policy's internal state.
        
        Args:
            state: Dictionary representing the desired state
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save policy state to disk.
        
        Args:
            path: Path where to save the policy
        """
        raise NotImplementedError("Policy saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load policy state from disk.
        
        Args:
            path: Path from which to load the policy
        """
        raise NotImplementedError("Policy loading not implemented")
    
    def clone(self) -> "Policy":
        """
        Create a copy of this policy.
        
        Returns:
            A new instance of the same policy with the same parameters
        """
        raise NotImplementedError("Policy cloning not implemented")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(capabilities={self.get_capabilities()})"


class StochasticPolicy(Policy):
    """
    Policy that can provide probability distributions over actions.
    
    Extends the base Policy interface to support stochastic decision
    making and uncertainty quantification.
    """
    
    @abstractmethod
    def sample_action(self, observation: Observation, 
                     context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Sample an action from the policy's probability distribution.
        
        Args:
            observation: The current observation
            context: Additional context (optional)
            
        Returns:
            Sampled action from the probability distribution
        """
        pass
    
    @abstractmethod
    def get_action_probabilities(self, observation: Observation) -> Dict[str, float]:
        """
        Get probability distribution over actions.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary mapping action identifiers to probabilities
        """
        pass
    
    def get_entropy(self, observation: Observation) -> float:
        """
        Calculate entropy of the action distribution.
        
        Args:
            observation: The current observation
            
        Returns:
            Entropy of the action probability distribution
        """
        probs = list(self.get_action_probabilities(observation).values())
        import math
        return -sum(p * math.log(p + 1e-8) for p in probs if p > 0)


class GreedyPolicy(Policy):
    """
    Policy that always selects the highest-valued action.
    
    Implements deterministic greedy decision making based on
    action value estimates.
    """
    
    @abstractmethod
    def get_action_values(self, observation: Observation) -> Dict[str, float]:
        """
        Get value estimates for all possible actions.
        
        Args:
            observation: The current observation
            
        Returns:
            Dictionary mapping action identifiers to value estimates
        """
        pass
    
    def decide(self, observation: Observation, 
               context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make a greedy decision (select highest-valued action).
        
        Args:
            observation: The current observation
            context: Additional context (optional)
            
        Returns:
            Action with the highest estimated value
        """
        action_values = self.get_action_values(observation)
        if not action_values:
            # Fallback to base implementation
            return super().decide(observation, context)
        
        best_action_id = max(action_values, key=action_values.get)
        # Convert action_id back to Action - this requires implementation-specific logic
        # For now, return a basic action
        from ..primitives import Action
        from datetime import datetime
        return Action(
            data={"action_id": best_action_id},
            action_type="greedy_selection",
            timestamp=datetime.now()
        )


class HierarchicalPolicy(Policy):
    """
    Policy that can make decisions at multiple levels of abstraction.
    
    Supports hierarchical decision making with high-level goals and
    low-level action selection.
    """
    
    @abstractmethod
    def decide_high_level(self, observation: Observation, 
                         context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make a high-level decision.
        
        Args:
            observation: The current observation
            context: Additional context (optional)
            
        Returns:
            High-level action or goal
        """
        pass
    
    @abstractmethod
    def decide_low_level(self, high_level_action: Action, 
                        observation: Observation,
                        context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make a low-level decision to implement a high-level goal.
        
        Args:
            high_level_action: The high-level action/goal
            observation: The current observation
            context: Additional context (optional)
            
        Returns:
            Low-level action to execute
        """
        pass
    
    def decide(self, observation: Observation, 
               context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Make a hierarchical decision.
        
        Args:
            observation: The current observation
            context: Additional context (optional)
            
        Returns:
            Action combining high-level and low-level decisions
        """
        high_level = self.decide_high_level(observation, context)
        return self.decide_low_level(high_level, observation, context)
