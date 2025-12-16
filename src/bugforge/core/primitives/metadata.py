"""
Metadata primitive - provides structured metadata for BugForge components.

Metadata captures version information, capabilities, provenance, and
other descriptive information about BugForge components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4


@dataclass(frozen=True)
class Metadata:
    """
    Structured metadata for BugForge components.
    
    Provides version information, capabilities, provenance tracking,
    and other descriptive metadata.
    """
    
    component_name: str
    """Name of the component (model, policy, etc.)."""
    
    version: str
    """Version string (semantic versioning recommended)."""
    
    created_at: datetime = field(default_factory=datetime.now)
    """When this component was created."""
    
    updated_at: Optional[datetime] = None
    """When this component was last updated."""
    
    component_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this component instance."""
    
    description: Optional[str] = None
    """Human-readable description of the component."""
    
    tags: Set[str] = field(default_factory=set)
    """Tags for categorization and search."""
    
    capabilities: Set[str] = field(default_factory=set)
    """Declared capabilities of this component."""
    
    requirements: Dict[str, str] = field(default_factory=dict)
    """Required dependencies and their versions."""
    
    schema_version: Optional[str] = None
    """Version of the metadata schema."""
    
    provenance: Optional[Dict[str, Any]] = None
    """Provenance information (source, training data, etc.)."""
    
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    """Custom metadata fields."""
    
    def __post_init__(self):
        # Convert lists to sets for tags and capabilities
        if isinstance(self.tags, list):
            object.__setattr__(self, 'tags', set(self.tags))
        if isinstance(self.capabilities, list):
            object.__setattr__(self, 'capabilities', set(self.capabilities))
        if self.provenance is None:
            object.__setattr__(self, 'provenance', {})
    
    @classmethod
    def create(cls, component_name: str, version: str, **kwargs) -> "Metadata":
        """Create metadata with required fields."""
        return cls(component_name=component_name, version=version, **kwargs)
    
    def with_updated_version(self, new_version: str) -> "Metadata":
        """Return metadata with updated version and timestamp."""
        return Metadata(
            component_name=self.component_name,
            version=new_version,
            created_at=self.created_at,
            updated_at=datetime.now(),
            component_id=self.component_id,
            description=self.description,
            tags=self.tags,
            capabilities=self.capabilities,
            requirements=self.requirements,
            schema_version=self.schema_version,
            provenance=self.provenance,
            custom_fields=self.custom_fields
        )
    
    def with_capability(self, capability: str) -> "Metadata":
        """Return metadata with added capability."""
        new_capabilities = self.capabilities | {capability}
        return Metadata(
            component_name=self.component_name,
            version=self.version,
            created_at=self.created_at,
            updated_at=datetime.now(),
            component_id=self.component_id,
            description=self.description,
            tags=self.tags,
            capabilities=new_capabilities,
            requirements=self.requirements,
            schema_version=self.schema_version,
            provenance=self.provenance,
            custom_fields=self.custom_fields
        )
    
    def without_capability(self, capability: str) -> "Metadata":
        """Return metadata without specified capability."""
        new_capabilities = self.capabilities - {capability}
        return Metadata(
            component_name=self.component_name,
            version=self.version,
            created_at=self.created_at,
            updated_at=datetime.now(),
            component_id=self.component_id,
            description=self.description,
            tags=self.tags,
            capabilities=new_capabilities,
            requirements=self.requirements,
            schema_version=self.schema_version,
            provenance=self.provenance,
            custom_fields=self.custom_fields
        )
    
    def has_capability(self, capability: str) -> bool:
        """Check if component has a specific capability."""
        return capability in self.capabilities
    
    def has_all_capabilities(self, capabilities: List[str]) -> bool:
        """Check if component has all specified capabilities."""
        return all(cap in self.capabilities for cap in capabilities)
    
    def has_any_capability(self, capabilities: List[str]) -> bool:
        """Check if component has any of the specified capabilities."""
        return any(cap in self.capabilities for cap in capabilities)
    
    def with_tag(self, tag: str) -> "Metadata":
        """Return metadata with added tag."""
        new_tags = self.tags | {tag}
        return Metadata(
            component_name=self.component_name,
            version=self.version,
            created_at=self.created_at,
            updated_at=datetime.now(),
            component_id=self.component_id,
            description=self.description,
            tags=new_tags,
            capabilities=self.capabilities,
            requirements=self.requirements,
            schema_version=self.schema_version,
            provenance=self.provenance,
            custom_fields=self.custom_fields
        )
    
    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get a custom field value."""
        return self.custom_fields.get(key, default)
    
    def with_custom_field(self, key: str, value: Any) -> "Metadata":
        """Return metadata with added/updated custom field."""
        new_custom_fields = {**self.custom_fields, key: value}
        return Metadata(
            component_name=self.component_name,
            version=self.version,
            created_at=self.created_at,
            updated_at=datetime.now(),
            component_id=self.component_id,
            description=self.description,
            tags=self.tags,
            capabilities=self.capabilities,
            requirements=self.requirements,
            schema_version=self.schema_version,
            provenance=self.provenance,
            custom_fields=new_custom_fields
        )
    
    def __str__(self) -> str:
        return f"Metadata({self.component_name} v{self.version})"


class CapabilityDescriptor:
    """Descriptor for component capabilities."""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.parameters: Dict[str, Any] = {}
    
    def with_parameter(self, key: str, value: Any) -> "CapabilityDescriptor":
        """Add a parameter to the capability descriptor."""
        self.parameters[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability descriptor to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters
        }
    
    def __str__(self) -> str:
        return f"Capability({self.name} v{self.version})"
