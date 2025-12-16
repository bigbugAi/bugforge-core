# BugForge Core

**BugForge Core** is foundational library for BugForge ecosystem — providing **core primitives, abstractions, and utilities** used to build learning, reasoning, and adaptive intelligence systems.

It is designed to be **model-agnostic**, **domain-agnostic**, and **extensible**, serving as shared backbone for higher-level BugForge modules such as finance, agents, memory, and robotics.

---

## Purpose

BugForge Core exists to solve one problem well:

> Provide clean, composable building blocks for intelligence systems that **learn, adapt, and evolve**.

This repository intentionally avoids domain-specific logic and focuses on **reusable intelligence infrastructure**.

---

## Core Architecture

### Package Structure

```
src/bugforge/core/
├── primitives/          # Core data structures
├── interfaces/          # Abstract interfaces  
├── memory/             # Memory implementations
├── learning/           # Learning components
├── runtime/            # Runtime orchestration
└── __init__.py        # Main exports
```

---

## Core Primitives

### Observation
Represents environmental data and sensor inputs:
```python
observation = Observation(
    data={"sensor_values": [1.0, 2.0, 3.0]},
    timestamp=datetime.now(),
    source="sensor_array",
    metadata={"quality": "high"}
)
```

### Action  
Represents decisions and control signals:
```python
action = Action(
    action_type="move_forward",
    parameters={"speed": 1.5, "duration": 2.0},
    confidence=0.95
)
```

### State
Represents system and environmental states:
```python
state = State(
    system_state={"position": (10, 20), "battery": 0.8},
    environment_state={"temperature": 22.5},
    timestamp=datetime.now()
)
```

### Feedback
Represents reward signals and evaluation feedback:
```python
feedback = Feedback(
    value=0.8,
    feedback_type="reward",
    source="environment",
    metadata={"episode_step": 42}
)
```

### Trajectory
Represents sequences of observations, actions, and feedback:
```python
trajectory = Trajectory(
    steps=[
        TrajectoryStep(observation, action, feedback, state)
        for step in episode_steps
    ],
    trajectory_id=uuid4()
)
```

---

## Core Interfaces

### Model Interface
Abstract interface for predictive models:
```python
class Model(ABC):
    @abstractmethod
    def predict(self, observation: Observation) -> Action:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        pass
```

### Policy Interface
Decision-making interface:
```python
class Policy(ABC):
    @abstractmethod
    def decide(self, observation: Observation) -> Action:
        pass
    
    @abstractmethod
    def update_policy(self, experience: Any) -> None:
        pass
```

### Learner Interface
Learning and adaptation interface:
```python
class Learner(ABC):
    @abstractmethod
    def update(self, experience: Any) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_learning_state(self) -> Dict[str, Any]:
        pass
```

### Evaluator Interface
Performance evaluation interface:
```python
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: Model, test_data: List[Any]) -> Dict[str, float]:
        pass
```

---

## Memory Implementations

### ShortTermMemory
Temporary storage with time-based expiration:
```python
memory = ShortTermMemory(
    max_capacity=1000,
    max_age_seconds=300,
    eviction_policy="lru"
)
```

### LongTermMemory
Persistent storage with search capabilities:
```python
memory = LongTermMemory(
    storage_path="./memory/",
    auto_save=True,
    index_search=True
)
```

### EpisodicMemory
Specialized storage for complete episodes:
```python
memory = EpisodicMemory(
    max_episodes=1000,
    similarity_threshold=0.5
)
```

### SemanticMemory
Embedding-based storage with semantic search:
```python
memory = SemanticMemory(
    encoder=text_encoder,
    similarity_threshold=0.7
)
```

### WorkingMemory
Limited-capacity active processing memory:
```python
memory = WorkingMemory(
    capacity=7,  # Miller's magical number
    eviction_policy="lru"
)
```

---

## Learning Components

### UpdateHook
Flexible update mechanisms:
```python
# Frequency-based updates
hook = FrequencyUpdateHook(update_frequency=10)

# Performance-based updates  
hook = ThresholdUpdateHook(performance_threshold=0.8)

# Adaptive updates
hook = AdaptiveUpdateHook(initial_frequency=5)
```

### TrainingLoop
Structured training processes:
```python
# Epoch-based training
trainer = EpochTrainingLoop(
    num_epochs=100,
    early_stopping_patience=10
)

# Online learning
trainer = OnlineTrainingLoop(
    update_frequency=1,
    evaluation_frequency=100
)

# Curriculum learning
trainer = CurriculumTrainingLoop(
    curriculum_stages=[
        {"num_epochs": 10, "difficulty": "easy"},
        {"num_epochs": 20, "difficulty": "medium"},
        {"num_epochs": 30, "difficulty": "hard"}
    ]
)
```

### EvaluationMetrics
Comprehensive performance metrics:
```python
metrics = EvaluationMetrics([
    AccuracyMetric(),
    PrecisionMetric(),
    RecallMetric(),
    F1ScoreMetric(),
    RewardMetric()
])

results = metrics.compute_all(predictions, targets)
```

---

## Runtime Components

### SystemOrchestrator
High-level system coordination:
```python
orchestrator = SystemOrchestrator()
orchestrator.initialize()

# Add components
orchestrator.add_component("policy", "policy", config)
orchestrator.add_component("learner", "learner", config)

# Connect components
orchestrator.connect_components("policy", "learner")

# Process observations
action = orchestrator.process_observation(observation)
```

### ComponentRegistry
Component discovery and management:
```python
registry = ComponentRegistry()

# Register component
registry.register(MyModel, name="my_model", description="Custom model")

# Find components
models = registry.find_components(component_type="model")

# Create instance
model = registry.create_component("my_model")
```

### ConfigManager
Centralized configuration management:
```python
config = ConfigManager("config.yaml")

# Define schema
config.define_schema("learning_rate", ConfigSchema(
    type="float",
    default=0.001,
    min_value=0.0,
    max_value=1.0,
    env_var="BUGFORGE_LEARNING_RATE"
))

# Use configuration
lr = config.get("learning_rate")
```

### LoggingUtils
Structured logging with performance tracking:
```python
logger = LoggingUtils.create_logger("my_system")

# Context logging
with logger.context(component="policy", episode=42):
    logger.info("Processing observation")
    
# Performance tracking
with logger.timer("decision_making"):
    action = policy.decide(observation)
```

### SystemMonitor
Health monitoring and metrics collection:
```python
monitor = SystemMonitor()

# Add custom health check
monitor.add_health_check(CustomHealthCheck("database", check_db_connection))

# Start monitoring
monitor.start_monitoring(interval_seconds=30)

# Get health status
status = monitor.get_health_status()
```

---

## Hugging Face Integration

### ModelProvider Interface
Abstract model downloading and management:
```python
class HuggingFaceModelProvider(ModelProvider):
    def download_model(self, model_id: str) -> LocalModelArtifact:
        pass
    
    def list_available_models(self) -> List[str]:
        pass
```

### LocalModelArtifact
Local model representation:
```python
artifact = LocalModelArtifact(
    model_path="./models/model.pt",
    config_path="./models/config.json",
    metadata={"model_id": "gpt2", "version": "1.0"}
)
```

---

## Design Principles

- **Minimal but powerful**  
- **Composable over monolithic**  
- **Explicit over magic**  
- **Interfaces first, implementations later**  
- **Built for extension, not lock-in**

BugForge Core is intentionally boring — because everything built on top of it shouldn't be.

---

## Quick Start

### Installation
```bash
pip install bugforge-core
```

### Basic Usage
```python
from bugforge.core import (
    Observation, Action, Feedback,
    SystemOrchestrator, ConfigManager
)

# Create system
orchestrator = SystemOrchestrator()
orchestrator.initialize()

# Process observation
observation = Observation(data={"input": "hello world"})
action = orchestrator.process_observation(observation)

# Update system
feedback = Feedback(value=1.0, feedback_type="reward")
orchestrator.update_system((observation, action, feedback))
```

### Custom Component
```python
from bugforge.core import Model, register_component

@register_component(name="my_model", description="Custom model")
class MyModel(Model):
    def predict(self, observation: Observation) -> Action:
        # Custom prediction logic
        return Action(action_type="custom_action")
    
    def get_capabilities(self) -> List[str]:
        return ["prediction", "custom_logic"]
```

---

## API Reference

Complete API documentation is available at:
- [Core Primitives](docs/primitives.md)
- [Interfaces](docs/interfaces.md) 
- [Memory Components](docs/memory.md)
- [Learning Components](docs/learning.md)
- [Runtime Components](docs/runtime.md)

---

## Contributing

BugForge Core follows the [BugForge contribution guidelines](CONTRIBUTING.md).

Key principles:
- **Interfaces first** - All new features start with abstract interfaces
- **Model-agnostic** - No hard-coded model architectures
- **Domain-agnostic** - No domain-specific logic in core
- **Extensible** - Easy to extend without modifying core

---

## License

BugForge Core is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## Related Projects

- **BugForge Agents** - Agent framework built on Core
- **BugForge Memory** - Advanced memory systems built on Core  
- **BugForge Finance** - Financial intelligence systems built on Core
- **BugForge Robotics** - Robotics control systems built on Core

---

**BugForge Core** - The foundation for intelligent systems that learn, adapt, and evolve. 
