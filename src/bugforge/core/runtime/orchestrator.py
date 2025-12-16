"""
SystemOrchestrator implementation for coordinating BugForge components.

SystemOrchestrator provides high-level orchestration of models,
policies, learners, and other components in intelligent systems.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
import time

from ..interfaces import (
    Model, Policy, Learner, Evaluator, MemoryStore,
    Retriever, Encoder, UpgradeHook
)
from ..primitives import Observation, Action, State, Feedback, Trajectory
from .component_registry import ComponentRegistry
from .config_manager import ConfigManager
from .logging_utils import StructuredLogger
from .monitoring import SystemMonitor


class SystemState(Enum):
    """System orchestration states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentInstance:
    """Component instance information."""
    name: str
    component: Any
    component_type: str
    config: Dict[str, Any]
    created_at: datetime
    last_used: datetime


class SystemOrchestrator:
    """
    Orchestrator for coordinating BugForge system components.
    
    Provides high-level coordination, lifecycle management,
    and integration of system components.
    """
    
    def __init__(self, 
                 config: Optional[ConfigManager] = None,
                 registry: Optional[ComponentRegistry] = None,
                 logger: Optional[StructuredLogger] = None):
        """
        Initialize system orchestrator.
        
        Args:
            config: Configuration manager
            registry: Component registry
            logger: Logger for orchestration events
        """
        self.config = config or ConfigManager()
        self.registry = registry or ComponentRegistry()
        self.logger = logger or StructuredLogger("orchestrator")
        self.monitor = SystemMonitor()
        
        self.state = SystemState.INITIALIZING
        self.components: Dict[str, ComponentInstance] = {}
        self.component_connections: Dict[str, List[str]] = {}
        self.system_lock = threading.RLock()
        
        # System metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "uptime_seconds": 0.0
        }
        
        self.start_time = datetime.now()
    
    def initialize(self) -> None:
        """Initialize the system orchestrator."""
        with self.system_lock:
            self.logger.info("Initializing system orchestrator")
            
            try:
                # Load configuration
                self._load_system_config()
                
                # Initialize core components
                self._initialize_core_components()
                
                # Setup component connections
                self._setup_component_connections()
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                self.state = SystemState.RUNNING
                self.logger.info("System orchestrator initialized successfully")
                
            except Exception as e:
                self.state = SystemState.ERROR
                self.logger.error("Failed to initialize system orchestrator", error=e)
                raise
    
    def shutdown(self) -> None:
        """Shutdown the system orchestrator."""
        with self.system_lock:
            self.logger.info("Shutting down system orchestrator")
            
            self.state = SystemState.STOPPING
            
            try:
                # Stop monitoring
                self.monitor.stop_monitoring()
                
                # Shutdown components
                self._shutdown_components()
                
                self.state = SystemState.STOPPED
                self.logger.info("System orchestrator shutdown completed")
                
            except Exception as e:
                self.state = SystemState.ERROR
                self.logger.error("Error during shutdown", error=e)
    
    def add_component(self, 
                     name: str,
                     component_type: str,
                     config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a component to the system.
        
        Args:
            name: Component name
            component_type: Type of component
            config: Component configuration
            
        Returns:
            Component instance name
        """
        with self.system_lock:
            if name in self.components:
                raise ValueError(f"Component '{name}' already exists")
            
            self.logger.info(f"Adding component: {name} ({component_type})")
            
            # Create component instance
            component = self._create_component(component_type, config or {})
            
            instance = ComponentInstance(
                name=name,
                component=component,
                component_type=component_type,
                config=config or {},
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            self.components[name] = instance
            self.component_connections[name] = []
            
            self.logger.info(f"Component '{name}' added successfully")
            
            return name
    
    def remove_component(self, name: str) -> bool:
        """
        Remove a component from the system.
        
        Args:
            name: Component name
            
        Returns:
            True if component was removed
        """
        with self.system_lock:
            if name not in self.components:
                return False
            
            self.logger.info(f"Removing component: {name}")
            
            # Check dependencies
            dependents = self._get_component_dependents(name)
            if dependents:
                raise ValueError(f"Cannot remove component '{name}' - has dependents: {dependents}")
            
            # Remove connections
            for connected_name in self.component_connections[name]:
                if connected_name in self.component_connections:
                    self.component_connections[connected_name] = [
                        n for n in self.component_connections[connected_name] 
                        if n != name
                    ]
            
            # Remove component
            del self.components[name]
            del self.component_connections[name]
            
            self.logger.info(f"Component '{name}' removed successfully")
            
            return True
    
    def connect_components(self, source: str, target: str) -> None:
        """
        Connect two components.
        
        Args:
            source: Source component name
            target: Target component name
        """
        with self.system_lock:
            if source not in self.components or target not in self.components:
                raise ValueError("Both components must exist")
            
            if target not in self.component_connections[source]:
                self.component_connections[source].append(target)
            
            self.logger.info(f"Connected components: {source} -> {target}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component instance.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        with self.system_lock:
            if name in self.components:
                instance = self.components[name]
                instance.last_used = datetime.now()
                return instance.component
            
            return None
    
    def process_observation(self, observation: Observation) -> Action:
        """
        Process an observation through the system.
        
        Args:
            observation: Input observation
            
        Returns:
            Generated action
        """
        if self.state != SystemState.RUNNING:
            raise RuntimeError("System is not running")
        
        start_time = time.time()
        
        try:
            with self.logger.context(operation="process_observation"):
                # Get policy component
                policy = self._get_policy_component()
                if not policy:
                    raise RuntimeError("No policy component available")
                
                # Process observation
                action = policy.decide(observation)
                
                # Update metrics
                self._update_metrics(True, time.time() - start_time)
                
                self.logger.info(f"Processed observation, generated action: {action.action_type}")
                
                return action
                
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error("Failed to process observation", error=e)
            raise
    
    def update_system(self, experience: Union[Trajectory, tuple]) -> Dict[str, float]:
        """
        Update system based on experience.
        
        Args:
            experience: Learning experience
            
        Returns:
            Update metrics
        """
        if self.state != SystemState.RUNNING:
            raise RuntimeError("System is not running")
        
        try:
            with self.logger.context(operation="update_system"):
                # Get learner component
                learner = self._get_learner_component()
                if not learner:
                    self.logger.warning("No learner component available")
                    return {}
                
                # Update learner
                metrics = learner.update(experience)
                
                self.logger.info(f"System updated with metrics: {metrics}")
                
                return metrics
                
        except Exception as e:
            self.logger.error("Failed to update system", error=e)
            raise
    
    def evaluate_system(self, test_data: List[Any]) -> Dict[str, float]:
        """
        Evaluate system performance.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Evaluation metrics
        """
        if self.state != SystemState.RUNNING:
            raise RuntimeError("System is not running")
        
        try:
            with self.logger.context(operation="evaluate_system"):
                # Get evaluator component
                evaluator = self._get_evaluator_component()
                if not evaluator:
                    self.logger.warning("No evaluator component available")
                    return {}
                
                # Get model component for evaluation
                model = self._get_model_component()
                if not model:
                    raise RuntimeError("No model component available")
                
                # Evaluate system
                metrics = evaluator.evaluate(model, test_data)
                
                self.logger.info(f"System evaluation completed: {metrics}")
                
                return metrics
                
        except Exception as e:
            self.logger.error("Failed to evaluate system", error=e)
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            System status dictionary
        """
        with self.system_lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "state": self.state.value,
                "uptime_seconds": uptime,
                "components": {
                    name: {
                        "type": instance.component_type,
                        "created_at": instance.created_at.isoformat(),
                        "last_used": instance.last_used.isoformat()
                    }
                    for name, instance in self.components.items()
                },
                "connections": self.component_connections,
                "metrics": self.metrics.copy(),
                "health": self.monitor.get_health_status(),
                "system_info": self.monitor.get_system_info()
            }
    
    def _load_system_config(self) -> None:
        """Load system configuration."""
        # Load component configurations
        component_configs = self.config.get_section("components", {})
        
        for component_name, config in component_configs.items():
            component_type = config.get("type")
            if component_type:
                self.add_component(component_name, component_type, config)
    
    def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        # Initialize default components if not configured
        if not self.components:
            self.logger.info("No components configured, using defaults")
            # Could add default component initialization here
    
    def _setup_component_connections(self) -> None:
        """Setup component connections based on configuration."""
        connections_config = self.config.get_section("connections", {})
        
        for source, targets in connections_config.items():
            if isinstance(targets, str):
                targets = [targets]
            
            for target in targets:
                try:
                    self.connect_components(source, target)
                except ValueError as e:
                    self.logger.warning(f"Failed to connect {source} -> {target}: {e}")
    
    def _create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """Create a component instance."""
        # Find component class in registry
        components = self.registry.find_components(component_type=component_type)
        
        if not components:
            raise ValueError(f"No component found for type: {component_type}")
        
        # Use first available component
        component_info = components[0]
        
        # Create instance with configuration
        return self.registry.create_component(component_info.name, **config)
    
    def _shutdown_components(self) -> None:
        """Shutdown all components."""
        for name, instance in self.components.items():
            try:
                if hasattr(instance.component, "shutdown"):
                    instance.component.shutdown()
                elif hasattr(instance.component, "close"):
                    instance.component.close()
            except Exception as e:
                self.logger.warning(f"Error shutting down component '{name}': {e}")
    
    def _get_component_by_type(self, component_type: str) -> Optional[Any]:
        """Get first component of specified type."""
        for instance in self.components.values():
            if instance.component_type == component_type:
                return instance.component
        return None
    
    def _get_policy_component(self) -> Optional[Policy]:
        """Get policy component."""
        return self._get_component_by_type("policy")
    
    def _get_model_component(self) -> Optional[Model]:
        """Get model component."""
        return self._get_component_by_type("model")
    
    def _get_learner_component(self) -> Optional[Learner]:
        """Get learner component."""
        return self._get_component_by_type("learner")
    
    def _get_evaluator_component(self) -> Optional[Evaluator]:
        """Get evaluator component."""
        return self._get_component_by_type("evaluator")
    
    def _get_component_dependents(self, name: str) -> List[str]:
        """Get components that depend on the specified component."""
        dependents = []
        for source, targets in self.component_connections.items():
            if name in targets:
                dependents.append(source)
        return dependents
    
    def _update_metrics(self, success: bool, response_time: float) -> None:
        """Update system metrics."""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        total = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        self.metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()


# Global orchestrator instance
_global_orchestrator = SystemOrchestrator()


def get_orchestrator() -> SystemOrchestrator:
    """
    Get global system orchestrator.
    
    Returns:
        Global orchestrator instance
    """
    return _global_orchestrator
