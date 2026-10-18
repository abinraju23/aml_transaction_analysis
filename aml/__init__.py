"""Rule-based transaction monitoring, measured honestly.

The deliverable of this package is not a detector. It is the measurement of what
a detector costs: alert volume, analyst hours, missed cases, and the threshold
decision that trades one against the other.
"""

from .data import BookConfig, generate_book, load_public_dataset
from .evaluate import (
    CostModel,
    confusion,
    missed_case_analysis,
    per_rule_table,
    portfolio_summary,
    threshold_sweep,
)
from .rules import Thresholds, run_all_rules

__all__ = [
    "BookConfig",
    "generate_book",
    "load_public_dataset",
    "Thresholds",
    "run_all_rules",
    "CostModel",
    "confusion",
    "per_rule_table",
    "portfolio_summary",
    "threshold_sweep",
    "missed_case_analysis",
]
