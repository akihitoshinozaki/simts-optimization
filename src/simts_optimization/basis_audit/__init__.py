"""C3 tools for held-out, noise-aware basis auditing."""

from .audit import (
    BasisAuditConfig,
    C3BasisAudit,
    audit_basis_expansion,
    audit_mock_experiment,
)
from .bootstrap import BootstrapInterval, BootstrapSummary
from .projection import ProjectionMetrics, evaluate_held_out_projection
from .subspace import SubspaceMetrics, principal_angles_degrees
from .svd import NoiseWhitenedSVD, noise_whitened_residual_svd
from .transforms import rotate_iq_pairs

__all__ = [
    "BasisAuditConfig",
    "BootstrapInterval",
    "BootstrapSummary",
    "C3BasisAudit",
    "NoiseWhitenedSVD",
    "ProjectionMetrics",
    "SubspaceMetrics",
    "audit_basis_expansion",
    "audit_mock_experiment",
    "evaluate_held_out_projection",
    "noise_whitened_residual_svd",
    "principal_angles_degrees",
    "rotate_iq_pairs",
]
