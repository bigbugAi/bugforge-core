"""
UpgradeHook interface - defines the contract for model evolution and migration.

UpgradeHooks are components that can manage the evolution, migration,
and upgrading of models, policies, and other components over time.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..primitives import Metadata


class UpgradeHook(ABC):
    """
    Abstract base class for all upgrade hooks in BugForge.
    
    Upgrade hooks are responsible for managing the evolution and
    migration of components as they are updated or improved.
    """
    
    @abstractmethod
    def can_upgrade(self, component: Any, target_version: str) -> bool:
        """
        Check if a component can be upgraded to a target version.
        
        Args:
            component: The component to check
            target_version: The target version to upgrade to
            
        Returns:
            True if upgrade is possible
        """
        pass
    
    @abstractmethod
    def upgrade(self, component: Any, target_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Upgrade a component to a target version.
        
        Args:
            component: The component to upgrade
            target_version: The target version
            context: Additional upgrade context (optional)
            
        Returns:
            Tuple of (upgraded_component, upgrade_metadata)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the upgrade hook's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    @abstractmethod
    def get_supported_versions(self) -> List[str]:
        """
        Get the list of versions this hook can handle.
        
        Returns:
            List of supported version strings
        """
        pass
    
    def validate_upgrade(self, component: Any, target_version: str) -> Dict[str, Any]:
        """
        Validate that an upgrade would be successful.
        
        Args:
            component: The component to validate
            target_version: The target version
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "can_upgrade": self.can_upgrade(component, target_version),
            "warnings": [],
            "errors": []
        }
        
        if not validation_result["can_upgrade"]:
            validation_result["errors"].append(
                f"Cannot upgrade from current version to {target_version}"
            )
        
        return validation_result
    
    def get_upgrade_path(self, current_version: str, target_version: str) -> List[str]:
        """
        Get the upgrade path from current to target version.
        
        Args:
            current_version: Starting version
            target_version: Target version
            
        Returns:
            List of intermediate versions to upgrade through
        """
        if current_version == target_version:
            return []
        
        # Default: direct upgrade if supported
        if self.can_upgrade(None, target_version):  # None represents any version
            return [target_version]
        
        return []
    
    def rollback(self, component: Any, previous_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Rollback a component to a previous version.
        
        Args:
            component: The component to rollback
            previous_version: The version to rollback to
            context: Additional rollback context (optional)
            
        Returns:
            Tuple of (rolled_back_component, rollback_metadata)
        """
        # Default implementation: treat rollback as upgrade to previous version
        return self.upgrade(component, previous_version, context)
    
    def get_upgrade_metadata(self, component: Any) -> Dict[str, Any]:
        """
        Get upgrade metadata for a component.
        
        Args:
            component: The component to get metadata for
            
        Returns:
            Dictionary with upgrade-related metadata
        """
        return {}
    
    def save(self, path: str) -> None:
        """
        Save upgrade hook state to disk.
        
        Args:
            path: Path where to save the hook
        """
        raise NotImplementedError("UpgradeHook saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load upgrade hook state from disk.
        
        Args:
            path: Path from which to load the hook
        """
        raise NotImplementedError("UpgradeHook loading not implemented")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(versions={self.get_supported_versions()})"


class ModelUpgradeHook(UpgradeHook):
    """
    Upgrade hook specialized for model components.
    
    Handles model-specific upgrade logic including parameter
    migration, architecture changes, and compatibility.
    """
    
    @abstractmethod
    def migrate_parameters(self, old_model: Any, new_model: Any) -> Dict[str, Any]:
        """
        Migrate parameters from old model to new model.
        
        Args:
            old_model: The old model instance
            new_model: The new model instance
            
        Returns:
            Dictionary with migration results
        """
        pass
    
    @abstractmethod
    def validate_compatibility(self, old_model: Any, new_model: Any) -> bool:
        """
        Validate compatibility between old and new models.
        
        Args:
            old_model: The old model instance
            new_model: The new model instance
            
        Returns:
            True if models are compatible
        """
        pass
    
    def upgrade(self, component: Any, target_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Upgrade a model with parameter migration.
        
        Args:
            component: The model to upgrade
            target_version: The target version
            context: Additional upgrade context (optional)
            
        Returns:
            Tuple of (upgraded_model, upgrade_metadata)
        """
        # Create new model instance
        new_model = self._create_model(target_version, context)
        
        # Migrate parameters
        migration_result = self.migrate_parameters(component, new_model)
        
        upgrade_metadata = {
            "old_version": getattr(component, "version", "unknown"),
            "new_version": target_version,
            "migration_result": migration_result,
            "upgrade_timestamp": context.get("timestamp") if context else None
        }
        
        return new_model, upgrade_metadata
    
    @abstractmethod
    def _create_model(self, version: str, context: Optional[Dict[str, Any]]) -> Any:
        """
        Create a new model instance of the specified version.
        
        Args:
            version: The version to create
            context: Additional creation context
            
        Returns:
            New model instance
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get model-specific capabilities."""
        return ["parameter_migration", "architecture_upgrade", "model_compatibility"]


class PolicyUpgradeHook(UpgradeHook):
    """
    Upgrade hook specialized for policy components.
    
    Handles policy-specific upgrade logic including action space
    changes, decision logic updates, and behavior preservation.
    """
    
    @abstractmethod
    def migrate_policy_logic(self, old_policy: Any, new_policy: Any) -> Dict[str, Any]:
        """
        Migrate policy logic from old to new policy.
        
        Args:
            old_policy: The old policy instance
            new_policy: The new policy instance
            
        Returns:
            Dictionary with migration results
        """
        pass
    
    @abstractmethod
    def validate_action_space_compatibility(self, old_policy: Any, new_policy: Any) -> bool:
        """
        Validate action space compatibility between policies.
        
        Args:
            old_policy: The old policy instance
            new_policy: The new policy instance
            
        Returns:
            True if action spaces are compatible
        """
        pass
    
    def upgrade(self, component: Any, target_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Upgrade a policy with logic migration.
        
        Args:
            component: The policy to upgrade
            target_version: The target version
            context: Additional upgrade context (optional)
            
        Returns:
            Tuple of (upgraded_policy, upgrade_metadata)
        """
        # Create new policy instance
        new_policy = self._create_policy(target_version, context)
        
        # Migrate policy logic
        migration_result = self.migrate_policy_logic(component, new_policy)
        
        upgrade_metadata = {
            "old_version": getattr(component, "version", "unknown"),
            "new_version": target_version,
            "migration_result": migration_result,
            "upgrade_timestamp": context.get("timestamp") if context else None
        }
        
        return new_policy, upgrade_metadata
    
    @abstractmethod
    def _create_policy(self, version: str, context: Optional[Dict[str, Any]]) -> Any:
        """
        Create a new policy instance of the specified version.
        
        Args:
            version: The version to create
            context: Additional creation context
            
        Returns:
            New policy instance
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get policy-specific capabilities."""
        return ["policy_migration", "action_space_compatibility", "behavior_preservation"]


class DataUpgradeHook(UpgradeHook):
    """
    Upgrade hook specialized for data format migration.
    
    Handles data schema changes, format conversions, and
    compatibility maintenance for stored data.
    """
    
    @abstractmethod
    def migrate_data_schema(self, old_data: Any, old_schema: str, new_schema: str) -> Any:
        """
        Migrate data from old schema to new schema.
        
        Args:
            old_data: The data in old schema format
            old_schema: The old schema identifier
            new_schema: The new schema identifier
            
        Returns:
            Data in new schema format
        """
        pass
    
    @abstractmethod
    def validate_schema_compatibility(self, old_schema: str, new_schema: str) -> bool:
        """
        Validate compatibility between data schemas.
        
        Args:
            old_schema: The old schema identifier
            new_schema: The new schema identifier
            
        Returns:
            True if schemas are compatible
        """
        pass
    
    def upgrade(self, component: Any, target_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Upgrade data format.
        
        Args:
            component: The data to upgrade
            target_version: The target version (schema)
            context: Additional upgrade context (optional)
            
        Returns:
            Tuple of (upgraded_data, upgrade_metadata)
        """
        # Get current schema
        current_schema = getattr(component, "schema_version", "unknown")
        
        # Migrate data schema
        upgraded_data = self.migrate_data_schema(component, current_schema, target_version)
        
        upgrade_metadata = {
            "old_schema": current_schema,
            "new_schema": target_version,
            "upgrade_timestamp": context.get("timestamp") if context else None
        }
        
        return upgraded_data, upgrade_metadata
    
    def get_capabilities(self) -> List[str]:
        """Get data-specific capabilities."""
        return ["schema_migration", "format_conversion", "data_compatibility"]


class CompositeUpgradeHook(UpgradeHook):
    """
    Upgrade hook that combines multiple upgrade strategies.
    
    Coordinates multiple specialized upgrade hooks to handle
    complex upgrade scenarios involving multiple component types.
    """
    
    @abstractmethod
    def add_hook(self, hook: UpgradeHook, priority: int = 0) -> None:
        """
        Add an upgrade hook to the composite.
        
        Args:
            hook: The upgrade hook to add
            priority: Priority for execution order (higher = earlier)
        """
        pass
    
    @abstractmethod
    def remove_hook(self, hook: UpgradeHook) -> None:
        """
        Remove an upgrade hook from the composite.
        
        Args:
            hook: The upgrade hook to remove
        """
        pass
    
    @abstractmethod
    def get_hooks(self) -> List[Tuple[UpgradeHook, int]]:
        """
        Get all registered hooks with their priorities.
        
        Returns:
            List of (hook, priority) tuples
        """
        pass
    
    def can_upgrade(self, component: Any, target_version: str) -> bool:
        """
        Check if any hook can upgrade the component.
        
        Args:
            component: The component to check
            target_version: The target version
            
        Returns:
            True if any hook can upgrade
        """
        for hook, _ in self.get_hooks():
            if hook.can_upgrade(component, target_version):
                return True
        return False
    
    def upgrade(self, component: Any, target_version: str, 
                context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Upgrade using the first applicable hook.
        
        Args:
            component: The component to upgrade
            target_version: The target version
            context: Additional upgrade context (optional)
            
        Returns:
            Tuple of (upgraded_component, upgrade_metadata)
        """
        # Sort hooks by priority (highest first)
        hooks = sorted(self.get_hooks(), key=lambda x: x[1], reverse=True)
        
        for hook, priority in hooks:
            if hook.can_upgrade(component, target_version):
                return hook.upgrade(component, target_version, context)
        
        raise ValueError(f"No hook can upgrade component to version {target_version}")
    
    def get_capabilities(self) -> List[str]:
        """Get combined capabilities from all hooks."""
        all_caps = set()
        for hook, _ in self.get_hooks():
            all_caps.update(hook.get_capabilities())
        return list(all_caps)
    
    def get_supported_versions(self) -> List[str]:
        """Get combined supported versions from all hooks."""
        all_versions = set()
        for hook, _ in self.get_hooks():
            all_versions.update(hook.get_supported_versions())
        return list(all_versions)
