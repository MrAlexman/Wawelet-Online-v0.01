import numpy as np
from core.schema import ParamSpec
from .base import SignalComponent, ComponentState

class SineComponent(SignalComponent):
    TYPE = "sine"
    NAME = "Синусоида"

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
                description="Амплитуда синусоидальной составляющей A·sin(2πft+φ).",
                examples=["A=1.0 — базовый уровень", "A=0.2 — слабая составляющая"],
            ),
            ParamSpec(
                key="frequency",
                label="Частота (Гц)",
                type="float",
                default=5.0,
                min=0.0,
                max=10_000_000.0,
                step=0.1,
                description="Частота синусоиды. Отражается как горизонтальная полоса на CWT-скейлограмме.",
                examples=["f=6 Гц — низкая частота", "f=30 Гц — средняя", "до fs/2 — высокочастотные"],
            ),
            ParamSpec(
                key="phase",
                label="Фаза (рад)",
                type="float",
                default=0.0,
                min=-10.0,
                max=10.0,
                step=0.01,
                description="Начальная фаза φ (в радианах). На спектр/скейлограмму влияет слабо, важна для временного вида.",
                examples=["0 — без сдвига", "π/2 ≈ 1.57 — сдвиг на четверть периода"],
            ),
            ParamSpec(
                key="dc",
                label="Смещение (DC)",
                type="float",
                default=0.0,
                min=-5.0,
                max=5.0,
                step=0.01,
                description="Постоянная составляющая (смещение вверх/вниз).",
                examples=["dc=0 — без смещения", "dc=0.5 — сигнал выше нуля"],
            ),
            ParamSpec(
                key="smooth_ms",
                label="Сглаживание изменения (мс)",
                type="int",
                default=150,
                min=0,
                max=500,
                step=10,
                description="Плавное изменение параметров амплитуды/частоты/фазы/смещения для уменьшения скачков.",
                examples=["0 — изменения мгновенные", "150–300 — плавный переход"],
            ),
        ]


    def __init__(self, state: ComponentState):
        super().__init__(state)
        # smoothing state
        self._cur = dict(state.params)
        self._target = dict(state.params)

    def update_params(self, new_params):
        self._target.update(new_params)
        super().update_params(new_params)

    def _smooth(self, key: str, alpha: float):
        c = float(self._cur.get(key, 0.0))
        t = float(self._target.get(key, c))
        self._cur[key] = c + alpha * (t - c)

    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        if not self.state.enabled:
            return np.zeros(n, dtype=np.float32)

        smooth_ms = int(self.state.params.get("smooth_ms", 150))
        if smooth_ms <= 0:
            self._cur = dict(self._target)

        # alpha per chunk
        alpha = 1.0 if smooth_ms <= 0 else min(1.0, (n / fs) / (smooth_ms / 1000.0))

        self._smooth("amplitude", alpha)
        self._smooth("frequency", alpha)
        self._smooth("phase", alpha)
        self._smooth("dc", alpha)

        A = float(self._cur.get("amplitude", 1.0))
        f = float(self._cur.get("frequency", 5.0))
        ph = float(self._cur.get("phase", 0.0))
        dc = float(self._cur.get("dc", 0.0))

        t = t0 + np.arange(n, dtype=np.float32) / float(fs)
        y = dc + A * np.sin(2 * np.pi * f * t + ph)
        return y.astype(np.float32)
