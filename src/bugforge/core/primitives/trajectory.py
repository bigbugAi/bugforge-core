"""
Trajectory primitive - represents sequences of observations, actions, and states.

Trajectories capture the temporal evolution of intelligence operations,
providing the foundation for learning from experience.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from .observation import Observation
from .action import Action
from .state import State
from .feedback import Feedback


@dataclass(frozen=True)
class TrajectoryStep:
    """
    Represents a single step in a trajectory.
    
    A step contains the observation, action, resulting state, and feedback
    at a specific point in time.
    """
    
    observation: Observation
    """The observation at this step."""
    
    action: Action
    """The action taken at this step."""
    
    next_state: State
    """The state after the action was executed."""
    
    feedback: Optional[Feedback] = None
    """Feedback received for this step."""
    
    step_index: int = 0
    """Index of this step in the trajectory."""
    
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Additional metadata about this step."""
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    def with_feedback(self, feedback: Feedback) -> "TrajectoryStep":
        """Return a new step with feedback added."""
        return TrajectoryStep(
            observation=self.observation,
            action=self.action,
            next_state=self.next_state,
            feedback=feedback,
            step_index=self.step_index,
            metadata=self.metadata
        )
    
    def __str__(self) -> str:
        return f"Step({self.step_index}: {self.observation.source} -> {self.action.action_type})"


@dataclass(frozen=True)
class Trajectory:
    """
    Represents a sequence of steps (trajectory or episode).
    
    Trajectories capture the complete temporal evolution of interactions,
    providing the foundation for learning from experience.
    """
    
    steps: List[TrajectoryStep]
    """Sequence of steps in the trajectory."""
    
    trajectory_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this trajectory."""
    
    start_time: datetime = field(default_factory=datetime.now)
    """When the trajectory started."""
    
    end_time: Optional[datetime] = None
    """When the trajectory ended."""
    
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Additional metadata about the trajectory."""
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def empty(cls, trajectory_id: Optional[UUID] = None) -> "Trajectory":
        """Create an empty trajectory."""
        return cls(
            steps=[],
            trajectory_id=trajectory_id or uuid4()
        )
    
    def add_step(self, step: TrajectoryStep) -> "Trajectory":
        """Add a step to the trajectory."""
        new_step = TrajectoryStep(
            observation=step.observation,
            action=step.action,
            next_state=step.next_state,
            feedback=step.feedback,
            step_index=len(self.steps),
            metadata=step.metadata
        )
        new_steps = self.steps + [new_step]
        return Trajectory(
            steps=new_steps,
            trajectory_id=self.trajectory_id,
            start_time=self.start_time,
            end_time=datetime.now() if len(new_steps) > 0 else None,
            metadata=self.metadata
        )
    
    def get_step(self, index: int) -> Optional[TrajectoryStep]:
        """Get a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def get_last_step(self) -> Optional[TrajectoryStep]:
        """Get the last step in the trajectory."""
        if self.steps:
            return self.steps[-1]
        return None
    
    def get_observations(self) -> List[Observation]:
        """Get all observations in the trajectory."""
        return [step.observation for step in self.steps]
    
    def get_actions(self) -> List[Action]:
        """Get all actions in the trajectory."""
        return [step.action for step in self.steps]
    
    def get_states(self) -> List[State]:
        """Get all states in the trajectory."""
        return [step.next_state for step in self.steps]
    
    def get_feedback(self) -> List[Feedback]:
        """Get all feedback in the trajectory."""
        return [step.feedback for step in self.steps if step.feedback is not None]
    
    def total_reward(self) -> float:
        """Calculate total reward from feedback."""
        total = 0.0
        for feedback in self.get_feedback():
            if feedback.is_positive():
                total += float(feedback.value)
            elif feedback.is_negative():
                total -= float(feedback.value)
        return total
    
    def length(self) -> int:
        """Get the length of the trajectory."""
        return len(self.steps)
    
    def is_empty(self) -> bool:
        """Check if the trajectory is empty."""
        return len(self.steps) == 0
    
    def with_metadata(self, **metadata) -> "Trajectory":
        """Return a new trajectory with additional metadata."""
        new_metadata = {**self.metadata, **metadata}
        return Trajectory(
            steps=self.steps,
            trajectory_id=self.trajectory_id,
            start_time=self.start_time,
            end_time=self.end_time,
            metadata=new_metadata
        )
    
    def __str__(self) -> str:
        return f"Trajectory(id={self.trajectory_id.hex[:8]}, steps={len(self.steps)})"


class TrajectoryBuffer:
    """Buffer for storing and managing trajectories."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self.trajectories: List[Trajectory] = []
        self.current_trajectory: Optional[Trajectory] = None
    
    def start_new_trajectory(self) -> Trajectory:
        """Start a new trajectory."""
        self.current_trajectory = Trajectory.empty()
        return self.current_trajectory
    
    def add_step(self, step: TrajectoryStep) -> Optional[Trajectory]:
        """Add a step to the current trajectory."""
        if self.current_trajectory is None:
            self.start_new_trajectory()
        
        self.current_trajectory = self.current_trajectory.add_step(step)
        return self.current_trajectory
    
    def end_trajectory(self) -> Optional[Trajectory]:
        """End the current trajectory and add it to the buffer."""
        if self.current_trajectory is not None:
            if self.max_size and len(self.trajectories) >= self.max_size:
                self.trajectories.pop(0)  # Remove oldest
            self.trajectories.append(self.current_trajectory)
            completed = self.current_trajectory
            self.current_trajectory = None
            return completed
        return None
    
    def get_trajectories(self, max_count: Optional[int] = None) -> List[Trajectory]:
        """Get trajectories from the buffer."""
        if max_count:
            return self.trajectories[-max_count:]
        return self.trajectories.copy()
    
    def clear(self) -> None:
        """Clear all trajectories from the buffer."""
        self.trajectories.clear()
        self.current_trajectory = None
    
    def size(self) -> int:
        """Get the number of stored trajectories."""
        return len(self.trajectories)
