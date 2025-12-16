"""
Action primitive - represents decisions and control signals.

Actions are immutable data structures that represent decisions
or control signals produced by models or policies.
"""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Action:
    """
    Represents an action or decision to be executed.
    
    Actions are immutable representations of control signals
    that can be applied to an environment or system.
    """
    
    data: Union[Dict[str, Any], str, bytes, float, int, List[Any]]
    """The action data or parameters."""
    
    action_type: str
    """Type or category of the action."""
    
    timestamp: datetime
    """When the action was generated."""
    
    actor_id: Optional[str] = None
    """Identifier for the agent/model that produced this action."""
    
    schema_version: Optional[str] = None
    """Version of the action schema for compatibility tracking."""
    
    action_id: UUID = None
    """Unique identifier for this action."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata about the action."""
    
    confidence: Optional[float] = None
    """Confidence score for this action (0.0 to 1.0)."""
    
    def __post_init__(self):
        if self.action_id is None:
            object.__setattr__(self, 'action_id', uuid4())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], action_type: str, **kwargs) -> "Action":
        """Create action from dictionary data."""
        return cls(data=data, action_type=action_type, **kwargs)
    
    @classmethod
    def from_scalar(cls, value: Union[float, int, str], action_type: str, **kwargs) -> "Action":
        """Create action from scalar value."""
        return cls(data=value, action_type=action_type, **kwargs)
    
    @classmethod
    def no_op(cls, actor_id: Optional[str] = None) -> "Action":
        """Create a no-operation action."""
        return cls(
            data={},
            action_type="no_op",
            timestamp=datetime.now(),
            actor_id=actor_id
        )
    
    def with_metadata(self, **metadata) -> "Action":
        """Return a new action with additional metadata."""
        new_metadata = {**self.metadata, **metadata}
        return Action(
            data=self.data,
            action_type=self.action_type,
            timestamp=self.timestamp,
            actor_id=self.actor_id,
            schema_version=self.schema_version,
            action_id=self.action_id,
            metadata=new_metadata,
            confidence=self.confidence
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from action data if it's a dictionary."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default
    
    def is_no_op(self) -> bool:
        """Check if this is a no-operation action."""
        return self.action_type == "no_op"
    
    def __str__(self) -> str:
        return f"Action(type={self.action_type}, timestamp={self.timestamp.isoformat()})"


class ActionSpace(ABC):
    """Abstract base class for defining action spaces."""
    
    def validate(self, action: Action) -> bool:
        """Validate that an action belongs to this space."""
        raise NotImplementedError
    
    def sample(self) -> Action:
        """Sample a valid action from this space."""
        raise NotImplementedError
    
    @property
    def shape(self) -> Optional[tuple]:
        """Shape of actions in this space."""
        return None
    
    @property
    def dtype(self) -> Optional[type]:
        """Data type of actions in this space."""
        return None
    
    @property
    def n_actions(self) -> Optional[int]:
        """Number of discrete actions (for discrete spaces)."""
        return None
