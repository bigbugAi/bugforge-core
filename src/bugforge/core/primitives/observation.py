"""
Observation primitive - represents environmental data and sensor inputs.

Observations are immutable data structures that capture the state
of the environment from an agent's perspective.
"""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Observation:
    """
    Represents an observation of the environment.
    
    Observations are immutable snapshots of environmental data
    at a specific point in time.
    """
    
    data: Union[Dict[str, Any], str, bytes, float, int]
    """The raw observation data."""
    
    timestamp: datetime
    """When the observation was captured."""
    
    source: str
    """Identifier for the observation source (sensor, modality, etc.)."""
    
    schema_version: Optional[str] = None
    """Version of the data schema for compatibility tracking."""
    
    observation_id: UUID = None
    """Unique identifier for this observation."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata about the observation."""
    
    def __post_init__(self):
        if self.observation_id is None:
            object.__setattr__(self, 'observation_id', uuid4())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: str, **kwargs) -> "Observation":
        """Create observation from dictionary data."""
        return cls(data=data, source=source, **kwargs)
    
    @classmethod
    def from_scalar(cls, value: Union[float, int, str], source: str, **kwargs) -> "Observation":
        """Create observation from scalar value."""
        return cls(data=value, source=source, **kwargs)
    
    def with_metadata(self, **metadata) -> "Observation":
        """Return a new observation with additional metadata."""
        new_metadata = {**self.metadata, **metadata}
        return Observation(
            data=self.data,
            timestamp=self.timestamp,
            source=self.source,
            schema_version=self.schema_version,
            observation_id=self.observation_id,
            metadata=new_metadata
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from observation data if it's a dictionary."""
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default
    
    def __str__(self) -> str:
        return f"Observation(source={self.source}, timestamp={self.timestamp.isoformat()})"


class ObservationSpace(ABC):
    """Abstract base class for defining observation spaces."""
    
    def validate(self, observation: Observation) -> bool:
        """Validate that an observation belongs to this space."""
        raise NotImplementedError
    
    def sample(self) -> Observation:
        """Sample a valid observation from this space."""
        raise NotImplementedError
    
    @property
    def shape(self) -> Optional[tuple]:
        """Shape of observations in this space."""
        return None
    
    @property
    def dtype(self) -> Optional[type]:
        """Data type of observations in this space."""
        return None
