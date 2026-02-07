from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from core.schema import ParamSpec

@dataclass
class ComponentState:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

class SignalComponent:
    TYPE: str = "base"
    NAME: str = "База"


    @staticmethod
    def schema() -> list[ParamSpec]:
        return []

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {s.key: s.default for s in cls.schema()}

    def __init__(self, state: ComponentState):
        self.state = state

    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        """Return component contribution for chunk [t0, t0+n/fs)."""
        return np.zeros(n, dtype=np.float32)

    def update_params(self, new_params: Dict[str, Any]):
        self.state.params.update(new_params)
