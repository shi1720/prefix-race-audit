"""Randomized prefix audits for response-suppression interventions."""
from .core import Record, estimate_record, replay, sample_depth, targets, coefficients
from .design import solve_design, exchangeable_coefficients, pooled_allocation
