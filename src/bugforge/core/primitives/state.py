"""
State primitive - represents system and environmental states.

States are immutable snapshots that capture the complete state
of a system or environment at a given time.
"""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4


@dataclass(frozen=True)
class State:
    """
    Represents the state of a system or environment.
    
    States are immutable snapshots that capture the complete
    state information at a specific point in time.
    """
    
    data: Dict[str, Any]
    """The state data as key-value pairs."""
    
    timestamp: datetime
    """When the state was captured."""
    
    state_id: Optional[str] = None
    """Identifier for this state (e.g., environment state ID)."""
    
    schema_version: Optional[str] = None
    """Version of the state schema for compatibility tracking."""
    
    state_uuid: UUID = None
    """Unique identifier for this state instance."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata about the state."""
    
    def __post_init__(self):
        if self.state_uuid is None:
            object.__setattr__(self, 'state_uuid', uuid4())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def empty(cls, state_id: Optional[str] = None) -> "State":
        """Create an empty state."""
        return cls(
            data={},
            timestamp=datetime.now(),
            state_id=state_id
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "State":
        """Create state from dictionary data."""
        return cls(data=data, **kwargs)
    
    def with_metadata(self, **metadata) -> "State":
        """Return a new state with additional metadata."""
        new_metadata = {**self.metadata, **metadata}
        return State(
            data=self.data,
            timestamp=self.timestamp,
            state_id=self.state_id,
            schema_version=self.schema_version,
            state_uuid=self.state_uuid,
            metadata=new_metadata
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state data."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> "State":
        """Return a new state with updated key-value pair."""
        new_data = {**self.data, key: value}
        return State(
            data=new_data,
            timestamp=self.timestamp,
            state_id=self.state_id,
            schema_version=self.schema_version,
            state_uuid=self.state_uuid,
            metadata=self.metadata
        )
    
    def merge(self, other: "State") -> "State":
        """Merge with another state, with other taking precedence."""
        merged_data = {**self.data, **other.data}
        return State(
            data=merged_data,
            timestamp=max(self.timestamp, other.timestamp),
            state_id=self.state_id or other.state_id,
            schema_version=self.schema_version or other.schema_version,
            metadata={**self.metadata, **other.metadata}
        )
    
    def __str__(self) -> str:
        return f"State(id={self.state_id}, timestamp={self.timestamp.isoformat()})"


class StateSpace(ABC):
    """Abstract base class for defining state spaces."""
    
    def validate(self, state: State) -> bool:
        """Validate that a state belongs to this space."""
        raise NotImplementedError
    
    def sample(self) -> State:
        """Sample a valid state from this space."""
        raise NotImplementedError
    
    @property
    def shape(self) -> Optional[Dict[str, tuple]]:
        """Shape of state variables."""
        return None
    
    @property
    def bounds(self) -> Optional[Dict[str, tuple]]:
        """Bounds for continuous state variables."""
        return None
