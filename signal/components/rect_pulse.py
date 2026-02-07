import numpy as np
from core.schema import ParamSpec
from .base import SignalComponent, ComponentState

class RectPulseComponent(SignalComponent):
    TYPE = "rect_pulse"
    NAME = "Прямоугольные импульсы"

    @staticmethod
    def schema():
        return [
            ParamSpec(
                key="amplitude",
                label="Амплитуда",
                type="float",
                default=1.0,
                min=0.0,
                max=10.0,
                step=0.01,
                description="Амплитуда прямоугольных импульсов.",
                examples=["A=1.0 — базовый уровень", "A=3.0 — более выраженные импульсы"],
            ),
            ParamSpec(
                key="width_sec",
                label="Длительность импульса (с)",
                type="float",
                default=0.02,
                min=0.0005,
                max=2.0,
                step=0.0005,
                description="Ширина одного импульса по времени.",
                examples=["0.02 с — короткий импульс", "0.2 с — широкий импульс"],
            ),
            ParamSpec(
                key="period_sec",
                label="Период повторения (с)",
                type="float",
                default=0.3,
                min=0.001,
                max=10.0,
                step=0.001,
                description="Интервал между началами импульсов. Частота повторения = 1/period.",
                examples=["0.3 с → ~3.33 Гц", "1.0 с → 1 Гц"],
            ),
            ParamSpec(
                key="start_time_sec",
                label="Старт (с)",
                type="float",
                default=0.0,
                min=0.0,
                max=60.0,
                step=0.01,
                description="Смещение по времени, начиная с которого запускается последовательность импульсов.",
                examples=["0 — сразу", "1.0 — импульсы начинаются через 1 секунду"],
            ),
        ]


    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        if not self.state.enabled:
            return np.zeros(n, dtype=np.float32)

        A = float(self.state.params.get("amplitude", 1.0))
        w = float(self.state.params.get("width_sec", 0.02))
        p = float(self.state.params.get("period_sec", 0.3))
        s = float(self.state.params.get("start_time_sec", 0.0))

        t = t0 + np.arange(n, dtype=np.float32) / float(fs)
        tt = t - s
        y = np.zeros(n, dtype=np.float32)
        mask = tt >= 0
        if np.any(mask) and p > 0:
            phase = np.mod(tt[mask], p)
            y[mask] = (phase < w).astype(np.float32) * A
        return y
