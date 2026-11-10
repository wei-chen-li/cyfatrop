"""NumPy callbacks for Fatrop's structured optimal control solver."""

from ._native import BLASFEO_COMMIT, FATROP_COMMIT, FATROP_VERSION, Status
from .abstract import OcpAbstract, TrajOptAbstract
from ._solver import Result, Solver

__version__ = "0.1.0"
__all__ = [
    "Solver", "Result", "Status", "OcpAbstract", "TrajOptAbstract",
    "FATROP_VERSION", "FATROP_COMMIT", "BLASFEO_COMMIT"
]
