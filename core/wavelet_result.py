from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class WaveletResult:
    image: np.ndarray        # 2D (rows=y, cols=time)
    y_axis: np.ndarray       # scales or frequencies
    x_axis: np.ndarray       # time axis (sec)
    label_y: str             # "Hz" or "scale" or "band/node"
    meta: Dict[str, Any]
