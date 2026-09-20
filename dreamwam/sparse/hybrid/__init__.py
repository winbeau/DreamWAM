"""Opt-in visual temporal sparsity; planning imports have no tensor dependencies."""

from .config import HybridConfig
from .schedule import StepContext, StepDecision, StepPlan, compile_plan

__all__ = ["HybridConfig", "StepContext", "StepDecision", "StepPlan", "compile_plan"]
