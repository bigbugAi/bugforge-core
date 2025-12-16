"""
ComponentRegistry implementation for managing system components.

ComponentRegistry provides a centralized registry for discovering,
registering, and managing BugForge components.
"""

from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass
from datetime import datetime
import inspect

from ..interfaces import (
    Model, Policy, Learner, Evaluator, MemoryStore, 
    Retriever, Encoder, UpgradeHook, ModelProvider
)


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: str
    component_class: Type
    description: str
    version: str
    author: str
    tags: List[str]
    dependencies: List[str]
    capabilities: List[str]
    registered_at: datetime
    metadata: Dict[str, Any]


class ComponentRegistry:
    """
    Registry for managing BugForge components.
    
    Provides discovery, registration, and management of components
    with support for dependencies and capabilities.
    """
    
    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[str, ComponentInfo] = {}
        self._components_by_type: Dict[str, List[str]] = {}
        self._components_by_capability: Dict[str, List[str]] = {}
        self._components_by_tag: Dict[str, List[str]] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
    
    def register(self, 
                 component_class: Type,
                 name: Optional[str] = None,
                 description: str = "",
                 version: str = "1.0.0",
                 author: str = "",
                 tags: Optional[List[str]] = None,
                 dependencies: Optional[List[str]] = None,
                 capabilities: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a component class.
        
        Args:
            component_class: The component class to register
            name: Component name (derived from class name if None)
            description: Component description
            version: Component version
            author: Component author
            tags: Component tags for categorization
            dependencies: List of component dependencies
            capabilities: List of component capabilities
            metadata: Additional metadata
            
        Returns:
            Registered component name
        """
        if name is None:
            name = component_class.__name__
        
        # Determine component type
        component_type = self._determine_component_type(component_class)
        
        # Get capabilities from class if not provided
        if capabilities is None and hasattr(component_class, 'get_capabilities'):
            try:
                # Create temporary instance to get capabilities
                temp_instance = component_class()
                capabilities = temp_instance.get_capabilities()
            except Exception:
                capabilities = []
        
        component_info = ComponentInfo(
            name=name,
            component_type=component_type,
            component_class=component_class,
            description=description,
            version=version,
            author=author,
            tags=tags or [],
            dependencies=dependencies or [],
            capabilities=capabilities or [],
            registered_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store component
        self._components[name] = component_info
        
        # Update indices
        self._update_indices(component_info)
        
        return name
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a component.
        
        Args:
            name: Component name to unregister
            
        Returns:
            True if component was unregistered
        """
        if name not in self._components:
            return False
        
        component_info = self._components[name]
        
        # Remove from indices
        self._remove_from_indices(component_info)
        
        # Remove from registry
        del self._components[name]
        
        return True
    
    def get_component(self, name: str) -> Optional[ComponentInfo]:
        """
        Get component information by name.
        
        Args:
            name: Component name
            
        Returns:
            Component information or None if not found
        """
        return self._components.get(name)
    
    def create_component(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered component.
        
        Args:
            name: Component name
            **kwargs: Constructor arguments
            
        Returns:
            Component instance
        """
        component_info = self.get_component(name)
        if not component_info:
            raise ValueError(f"Component '{name}' not found")
        
        # Check dependencies
        self._check_dependencies(component_info)
        
        # Create instance
        return component_info.component_class(**kwargs)
    
    def find_components(self, 
                       component_type: Optional[str] = None,
                       capabilities: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       author: Optional[str] = None) -> List[ComponentInfo]:
        """
        Find components matching criteria.
        
        Args:
            component_type: Filter by component type
            capabilities: Filter by required capabilities
            tags: Filter by tags
            author: Filter by author
            
        Returns:
            List of matching component information
        """
        results = list(self._components.values())
        
        if component_type:
            results = [c for c in results if c.component_type == component_type]
        
        if capabilities:
            results = [c for c in results 
                     if all(cap in c.capabilities for cap in capabilities)]
        
        if tags:
            results = [c for c in results 
                     if any(tag in c.tags for tag in tags)]
        
        if author:
            results = [c for c in results if c.author == author]
        
        return results
    
    def get_components_by_type(self, component_type: str) -> List[str]:
        """
        Get component names by type.
        
        Args:
            component_type: Component type
            
        Returns:
            List of component names
        """
        return self._components_by_type.get(component_type, [])
    
    def get_components_by_capability(self, capability: str) -> List[str]:
        """
        Get component names by capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List of component names
        """
        return self._components_by_capability.get(capability, [])
    
    def get_components_by_tag(self, tag: str) -> List[str]:
        """
        Get component names by tag.
        
        Args:
            tag: Component tag
            
        Returns:
            List of component names
        """
        return self._components_by_tag.get(tag, [])
    
    def list_all_components(self) -> List[str]:
        """
        List all registered component names.
        
        Returns:
            List of all component names
        """
        return list(self._components.keys())
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph.
        
        Returns:
            Dictionary mapping component names to their dependencies
        """
        return self._dependency_graph.copy()
    
    def check_dependencies(self, name: str) -> List[str]:
        """
        Check if component dependencies are satisfied.
        
        Args:
            name: Component name
            
        Returns:
            List of missing dependencies
        """
        component_info = self.get_component(name)
        if not component_info:
            return []
        
        missing = []
        for dep in component_info.dependencies:
            if dep not in self._components:
                missing.append(dep)
        
        return missing
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        type_counts = {}
        for component_type, names in self._components_by_type.items():
            type_counts[component_type] = len(names)
        
        capability_counts = {}
        for capability, names in self._components_by_capability.items():
            capability_counts[capability] = len(names)
        
        tag_counts = {}
        for tag, names in self._components_by_tag.items():
            tag_counts[tag] = len(names)
        
        return {
            "total_components": len(self._components),
            "components_by_type": type_counts,
            "components_by_capability": capability_counts,
            "components_by_tag": tag_counts,
            "total_dependencies": sum(len(c.dependencies) for c in self._components.values()),
            "total_capabilities": sum(len(c.capabilities) for c in self._components.values())
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Export registry data.
        
        Returns:
            Dictionary with registry data
        """
        export_data = {
            "components": {},
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_components": len(self._components)
            }
        }
        
        for name, info in self._components.items():
            export_data["components"][name] = {
                "name": info.name,
                "component_type": info.component_type,
                "description": info.description,
                "version": info.version,
                "author": info.author,
                "tags": info.tags,
                "dependencies": info.dependencies,
                "capabilities": info.capabilities,
                "registered_at": info.registered_at.isoformat(),
                "metadata": info.metadata
            }
        
        return export_data
    
    def _determine_component_type(self, component_class: Type) -> str:
        """Determine component type from class hierarchy."""
        if issubclass(component_class, Model):
            return "model"
        elif issubclass(component_class, Policy):
            return "policy"
        elif issubclass(component_class, Learner):
            return "learner"
        elif issubclass(component_class, Evaluator):
            return "evaluator"
        elif issubclass(component_class, MemoryStore):
            return "memory_store"
        elif issubclass(component_class, Retriever):
            return "retriever"
        elif issubclass(component_class, Encoder):
            return "encoder"
        elif issubclass(component_class, UpgradeHook):
            return "upgrade_hook"
        elif issubclass(component_class, ModelProvider):
            return "model_provider"
        else:
            return "unknown"
    
    def _update_indices(self, component_info: ComponentInfo) -> None:
        """Update all indices with new component."""
        name = component_info.name
        
        # Type index
        if component_info.component_type not in self._components_by_type:
            self._components_by_type[component_info.component_type] = []
        self._components_by_type[component_info.component_type].append(name)
        
        # Capability index
        for capability in component_info.capabilities:
            if capability not in self._components_by_capability:
                self._components_by_capability[capability] = []
            self._components_by_capability[capability].append(name)
        
        # Tag index
        for tag in component_info.tags:
            if tag not in self._components_by_tag:
                self._components_by_tag[tag] = []
            self._components_by_tag[tag].append(name)
        
        # Dependency graph
        self._dependency_graph[name] = component_info.dependencies
    
    def _remove_from_indices(self, component_info: ComponentInfo) -> None:
        """Remove component from all indices."""
        name = component_info.name
        
        # Type index
        if component_info.component_type in self._components_by_type:
            self._components_by_type[component_info.component_type] = [
                n for n in self._components_by_type[component_info.component_type] 
                if n != name
            ]
        
        # Capability index
        for capability in component_info.capabilities:
            if capability in self._components_by_capability:
                self._components_by_capability[capability] = [
                    n for n in self._components_by_capability[capability] 
                    if n != name
                ]
        
        # Tag index
        for tag in component_info.tags:
            if tag in self._components_by_tag:
                self._components_by_tag[tag] = [
                    n for n in self._components_by_tag[tag] 
                    if n != name
                ]
        
        # Dependency graph
        if name in self._dependency_graph:
            del self._dependency_graph[name]
    
    def _check_dependencies(self, component_info: ComponentInfo) -> None:
        """Check if component dependencies are satisfied."""
        missing = self.check_dependencies(component_info.name)
        if missing:
            raise ValueError(f"Missing dependencies for component '{component_info.name}': {missing}")


# Global registry instance
_global_registry = ComponentRegistry()


def register_component(*args, **kwargs) -> Callable:
    """
    Decorator for registering components.
    
    Args:
        *args: Arguments to pass to register()
        **kwargs: Keyword arguments to pass to register()
        
    Returns:
        Decorator function
    """
    def decorator(component_class: Type) -> Type:
        _global_registry.register(component_class, *args, **kwargs)
        return component_class
    
    return decorator


def get_registry() -> ComponentRegistry:
    """
    Get the global component registry.
    
    Returns:
        Global component registry instance
    """
    return _global_registry
