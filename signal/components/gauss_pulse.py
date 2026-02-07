import numpy as np
from core.schema import ParamSpec
from .base import SignalComponent, ComponentState

class GaussPulseComponent(SignalComponent):
    TYPE = "gauss_pulse"
    NAME = "Гауссовы импульсы"

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
                description="Амплитуда гауссова импульса.",
                examples=["A=1.0 — базовый уровень", "A=3.0 — выраженный импульс"],
            ),
            ParamSpec(
                key="sigma_sec",
                label="Ширина σ (с)",
                type="float",
                default=0.01,
                min=0.0002,
                max=2.0,
                step=0.0002,
                description="Параметр σ определяет длительность/ширину гауссова импульса по времени.",
                examples=["0.01 с — короткий импульс", "0.1 с — более широкий импульс"],
            ),
            ParamSpec(
                key="center_time_sec",
                label="Центр (с)",
                type="float",
                default=1.0,
                min=0.0,
                max=60.0,
                step=0.01,
                description="Время центра первого импульса (максимум функции).",
                examples=["1.0 с — импульс около 1 секунды", "0.2 с — ранний импульс"],
            ),
            ParamSpec(
                key="repetition_period_sec",
                label="Период повторения (с, 0 = выкл.)",
                type="float",
                default=0.0,
                min=0.0,
                max=10.0,
                step=0.01,
                description="Период повторения импульсов. При 0 формируется одиночный импульс.",
                examples=["0 — одиночный импульс", "0.5 — импульсы каждые 0.5 с"],
            ),
        ]



    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        if not self.state.enabled:
            return np.zeros(n, dtype=np.float32)

        A = float(self.state.params.get("amplitude", 1.0))
        sigma = float(self.state.params.get("sigma_sec", 0.01))
        c0 = float(self.state.params.get("center_time_sec", 1.0))
        rp = float(self.state.params.get("repetition_period_sec", 0.0))

        t = t0 + np.arange(n, dtype=np.float32) / float(fs)
        y = np.zeros(n, dtype=np.float32)

        centers = [c0]
        if rp and rp > 0:
            # generate nearby centers around chunk
            chunk_mid = t0 + (n / fs) * 0.5
            k0 = int((chunk_mid - c0) / rp)
            centers = [c0 + (k0 + dk) * rp for dk in range(-2, 3) if c0 + (k0 + dk) * rp >= 0]

        for c in centers:
            y += A * np.exp(-0.5 * ((t - c) / sigma) ** 2).astype(np.float32)
        return y
