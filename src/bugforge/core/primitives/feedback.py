"""
Feedback primitive - represents reward signals and evaluation feedback.

Feedback provides information about the quality of actions or decisions,
including rewards, costs, and other evaluation signals.
"""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4


class FeedbackType(Enum):
    """Types of feedback signals."""
    REWARD = "reward"
    COST = "cost"
    PENALTY = "penalty"
    BONUS = "bonus"
    EVALUATION = "evaluation"
    HUMAN_FEEDBACK = "human_feedback"
    EXTERNAL_SIGNAL = "external_signal"


@dataclass(frozen=True)
class Feedback:
    """
    Represents feedback about actions or decisions.
    
    Feedback provides evaluation information that can be used
    for learning, adaptation, and improvement.
    """
    
    value: Union[float, int, Dict[str, float]]
    """The feedback value (reward, cost, evaluation score, etc.)."""
    
    feedback_type: FeedbackType
    """Type of feedback signal."""
    
    timestamp: datetime
    """When the feedback was received."""
    
    source: str
    """Source of the feedback (environment, human, evaluator, etc.)."""
    
    target_id: Optional[str] = None
    """ID of the action, decision, or episode this feedback targets."""
    
    schema_version: Optional[str] = None
    """Version of the feedback schema for compatibility tracking."""
    
    feedback_id: UUID = None
    """Unique identifier for this feedback."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata about the feedback."""
    
    confidence: Optional[float] = None
    """Confidence in the feedback quality (0.0 to 1.0)."""
    
    def __post_init__(self):
        if self.feedback_id is None:
            object.__setattr__(self, 'feedback_id', uuid4())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def reward(cls, value: float, source: str, **kwargs) -> "Feedback":
        """Create a reward feedback."""
        return cls(value=value, feedback_type=FeedbackType.REWARD, source=source, **kwargs)
    
    @classmethod
    def cost(cls, value: float, source: str, **kwargs) -> "Feedback":
        """Create a cost feedback."""
        return cls(value=value, feedback_type=FeedbackType.COST, source=source, **kwargs)
    
    @classmethod
    def evaluation(cls, scores: Dict[str, float], source: str, **kwargs) -> "Feedback":
        """Create an evaluation feedback with multiple scores."""
        return cls(value=scores, feedback_type=FeedbackType.EVALUATION, source=source, **kwargs)
    
    def with_metadata(self, **metadata) -> "Feedback":
        """Return a new feedback with additional metadata."""
        new_metadata = {**self.metadata, **metadata}
        return Feedback(
            value=self.value,
            feedback_type=self.feedback_type,
            timestamp=self.timestamp,
            source=self.source,
            target_id=self.target_id,
            schema_version=self.schema_version,
            feedback_id=self.feedback_id,
            metadata=new_metadata,
            confidence=self.confidence
        )
    
    def get_score(self, metric: str, default: float = 0.0) -> float:
        """Get a specific score from evaluation feedback."""
        if isinstance(self.value, dict):
            return self.value.get(metric, default)
        return float(self.value) if metric == "total" else default
    
    def is_positive(self) -> bool:
        """Check if feedback is positive (reward/bonus)."""
        return self.feedback_type in [FeedbackType.REWARD, FeedbackType.BONUS]
    
    def is_negative(self) -> bool:
        """Check if feedback is negative (cost/penalty)."""
        return self.feedback_type in [FeedbackType.COST, FeedbackType.PENALTY]
    
    def __str__(self) -> str:
        return f"Feedback(type={self.feedback_type.value}, source={self.source}, value={self.value})"


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __call__(self, state: "State", action: "Action", next_state: "State") -> Feedback:
        """Compute reward for a state-action-next_state transition."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset any internal state for new episodes."""
        pass


class CostFunction(ABC):
    """Abstract base class for cost functions."""
    
    def __call__(self, state: "State", action: "Action", next_state: "State") -> Feedback:
        """Compute cost for a state-action-next_state transition."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset any internal state for new episodes."""
        pass
