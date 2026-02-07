from __future__ import annotations
from typing import Dict, Any
import numpy as np
from core.wavelet_result import WaveletResult
from core.schema import Schema

class IWaveletPlugin:
    PLUGIN_META: dict

    def get_parameters_schema(self) -> Schema:
        raise NotImplementedError

    def transform(self, x: np.ndarray, fs: float, params: Dict[str, Any]) -> WaveletResult:
        raise NotImplementedError
