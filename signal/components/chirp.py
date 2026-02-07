import numpy as np
from scipy.signal import chirp
from core.schema import ParamSpec
from .base import SignalComponent, ComponentState

class ChirpComponent(SignalComponent):
    TYPE = "chirp"
    NAME = "Чирп-свип (линейная развертка)"

    @staticmethod
    def schema():
        return [
            ParamSpec(
                key="amplitude",
                label="Амплитуда",
                type="float",
                default=0.8,
                min=0.0,
                max=10.0,
                step=0.01,
                description="Амплитуда свип-сигнала (линейная частотная развертка).",
                examples=["A=0.8 — базовый уровень", "A=2.0 — выраженный свип"],
            ),
            ParamSpec(
                key="f0",
                label="Начальная частота f0 (Гц)",
                type="float",
                default=10.0,
                min=0.0,
                max=10_000_000.0,
                step=0.1,
                description="Частота в начале развертки (t=0 относительно старта).",
                examples=["f0=10 Гц — низкий старт", "f0=100 Гц — средний", "до fs/2 — высокочастотный"],
            ),
            ParamSpec(
                key="f1",
                label="Конечная частота f1 (Гц)",
                type="float",
                default=200.0,
                min=0.0,
                max=10_000_000.0,
                step=0.1,
                description="Частота в конце развертки (t=duration).",
                examples=["f1=200 Гц — умеренный диапазон", "f1=50k Гц — высокочастотный чирп"],
            ),
            ParamSpec(
                key="duration_sec",
                label="Длительность (с)",
                type="float",
                default=2.0,
                min=0.01,
                max=60.0,
                step=0.01,
                description="Длительность свип-участка. Вне интервала формируется нулевой сигнал.",
                examples=["2.0 с — типичный тест", "0.5 с — короткий свип"],
            ),
            ParamSpec(
                key="start_time_sec",
                label="Старт (с)",
                type="float",
                default=0.0,
                min=0.0,
                max=60.0,
                step=0.01,
                description="Время начала свип-участка (смещение по времени).",
                examples=["0 — старт сразу", "1.0 — старт через 1 секунду"],
            ),
        ]


    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        if not self.state.enabled:
            return np.zeros(n, dtype=np.float32)

        A = float(self.state.params.get("amplitude", 0.8))
        f0 = float(self.state.params.get("f0", 10.0))
        f1 = float(self.state.params.get("f1", 200.0))
        dur = float(self.state.params.get("duration_sec", 2.0))
        st = float(self.state.params.get("start_time_sec", 0.0))

        t = t0 + np.arange(n, dtype=np.float32) / float(fs)
        y = np.zeros(n, dtype=np.float32)

        mask = (t >= st) & (t <= st + dur)
        if np.any(mask):
            tt = (t[mask] - st).astype(np.float64)
            y[mask] = (A * chirp(tt, f0=f0, f1=f1, t1=dur, method="linear")).astype(np.float32)
        return y
