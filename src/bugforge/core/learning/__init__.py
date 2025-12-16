"""
Learning abstractions for BugForge intelligence systems.

This module provides learning components, update hooks, and
training infrastructure for adaptive intelligence systems.
"""

from .update_hook import UpdateHook
from .training_loop import TrainingLoop
from .evaluation_metrics import EvaluationMetrics

__all__ = [
    "UpdateHook",
    "TrainingLoop",
    "EvaluationMetrics",
]
