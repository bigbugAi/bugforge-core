"""
BugForge Core Library

A foundational, model-agnostic intelligence library for building
adaptive systems with support for various learning paradigms,
memory architectures, and runtime orchestration.

Core Components:
- Primitives: Observation, Action, State, Feedback, Trajectory, Metadata
- Interfaces: Model, Policy, Learner, Evaluator, MemoryStore, Retriever, Encoder, UpgradeHook
- Memory: ShortTermMemory, LongTermMemory, EpisodicMemory, SemanticMemory, WorkingMemory
- Learning: UpdateHook, TrainingLoop, EvaluationMetrics
- Runtime: SystemOrchestrator, ComponentRegistry, ConfigManager, LoggingUtils, SystemMonitor
"""

__version__ = "0.1.0"
__author__ = "BugForge Team"

# Core primitives
from .primitives import (
    Observation,
    Action,
    State,
    Feedback,
    Trajectory,
    Metadata,
)

# Core interfaces
from .interfaces import (
    Model,
    Policy,
    Learner,
    Evaluator,
    MemoryStore,
    Retriever,
    Encoder,
    UpgradeHook,
    ModelProvider,
    LocalModelArtifact,
)

# Memory implementations
from .memory import (
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
)

# Learning components
from .learning import (
    UpdateHook,
    TrainingLoop,
    EvaluationMetrics,
)

# Runtime components
from .runtime import (
    SystemOrchestrator,
    ComponentRegistry,
    ConfigManager,
    LoggingUtils,
    SystemMonitor,
)

# Utility functions
from .runtime.component_registry import register_component, get_registry
from .runtime.config_manager import define_config, get_config, config, set_config
from .runtime.logging_utils import get_performance_tracker, track_performance
from .runtime.monitoring import start_monitoring, stop_monitoring, record_metric, get_health_status
from .runtime.orchestrator import get_orchestrator

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core primitives
    "Observation",
    "Action", 
    "State",
    "Feedback",
    "Trajectory",
    "Metadata",
    
    # Core interfaces
    "Model",
    "Policy",
    "Learner",
    "Evaluator",
    "MemoryStore",
    "Retriever",
    "Encoder",
    "UpgradeHook",
    "ModelProvider",
    "LocalModelArtifact",
    
    # Memory implementations
    "ShortTermMemory",
    "LongTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    
    # Learning components
    "UpdateHook",
    "TrainingLoop",
    "EvaluationMetrics",
    
    # Runtime components
    "SystemOrchestrator",
    "ComponentRegistry",
    "ConfigManager",
    "LoggingUtils",
    "SystemMonitor",
    
    # Utility functions
    "register_component",
    "get_registry",
    "define_config",
    "get_config",
    "config",
    "set_config",
    "get_performance_tracker",
    "track_performance",
    "start_monitoring",
    "stop_monitoring",
    "record_metric",
    "get_health_status",
    "get_orchestrator",
]
