from enum import Enum
from typing import Any

FATROP_VERSION: str
FATROP_COMMIT: str
BLASFEO_COMMIT: str

class Status(Enum):
    Success = 0
    MaxIterExceeded = 1
    StopAtAcceptablePoint = 2
    LocalInfeasibility = 3
    LineSearchFailed = 4
    ErrorInStepComputation = 5
    Unknown = 6

class NativeSolver:
    def __init__(self, problem: Any, options: dict[str, Any]) -> None: ...
    def solve(self) -> dict[str, Any]: ...
