"""Milestone C5 outer Bayesian optimization over experiment settings theta.

BO chooses settings theta; the C4 inner QP still chooses channel weights w.
Importing this package does not import torch; BoTorch loads on first BO step.
"""

from .benchmark import (
    BenchmarkResult,
    CampaignOutcome,
    StrategySummary,
    plot_benchmark,
    run_benchmark,
    summarize_campaign,
)
from .evaluation import (
    C5_PARETO_SETTINGS,
    EvaluationRecord,
    FoldScore,
    SimulatedExperiment,
    score_experiment,
)
from .loop import (
    BASELINE_CONFIG,
    STRATEGIES,
    CampaignConfig,
    CampaignResult,
    NoiseEstimate,
    estimate_drift,
    estimate_noise,
    run_campaign,
)
from .space import C5_DESIGN_SPACE, C5_REFERENCE_SETTINGS, DesignSpace, Parameter

__all__ = [
    "BASELINE_CONFIG",
    "BenchmarkResult",
    "C5_DESIGN_SPACE",
    "C5_PARETO_SETTINGS",
    "C5_REFERENCE_SETTINGS",
    "CampaignConfig",
    "CampaignOutcome",
    "CampaignResult",
    "DesignSpace",
    "EvaluationRecord",
    "FoldScore",
    "NoiseEstimate",
    "Parameter",
    "STRATEGIES",
    "SimulatedExperiment",
    "StrategySummary",
    "estimate_drift",
    "estimate_noise",
    "plot_benchmark",
    "run_benchmark",
    "run_campaign",
    "score_experiment",
    "summarize_campaign",
]
