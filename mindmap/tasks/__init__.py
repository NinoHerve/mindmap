"""
Generic Task/run_parallel_pipeline framework shared across analysis domains.
"""

from .base import (
    Task,
    run_parallel_pipeline,
    read_ave,
    read_cov,
    read_simulator,
)

__all__ = [
    "Task",
    "run_parallel_pipeline",
    "read_ave",
    "read_cov",
    "read_simulator",
]
