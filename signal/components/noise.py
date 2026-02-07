import numpy as np
from core.schema import ParamSpec
from .base import SignalComponent, ComponentState

class NoiseComponent(SignalComponent):
    TYPE = "noise"
    NAME = "Шум (гауссов)"

    @staticmethod
    def schema():
        return [
            ParamSpec(
                key="mean",
                label="Среднее (μ)",
                type="float",
                default=0.0,
                min=-5.0,
                max=5.0,
                step=0.01,
                description="Математическое ожидание белого гауссова шума.",
                examples=["0 — шум вокруг нуля", "0.2 — добавляет постоянное смещение через шум"],
            ),
            ParamSpec(
                key="sigma",
                label="СКО (σ)",
                type="float",
                default=0.2,
                min=0.0,
                max=5.0,
                step=0.01,
                description="Среднеквадратическое отклонение. Определяет «силу» шума.",
                examples=["0.05 — слабый шум", "0.5 — сильный шум"],
            ),
            ParamSpec(
                key="seed",
                label="Seed (0 = случайно)",
                type="int",
                default=0,
                min=0,
                max=10_000,
                step=1,
                description="Зерно генератора случайных чисел. При seed≠0 шум становится воспроизводимым.",
                examples=["0 — разные реализации на каждом запуске", "123 — одинаковый шум при одинаковых настройках"],
            ),
        ]


    def __init__(self, state: ComponentState):
        super().__init__(state)
        self._rng = np.random.default_rng(None)
        self._last_seed: int | None = None

    def generate(self, t0: float, n: int, fs: float) -> np.ndarray:
        if not self.state.enabled:
            return np.zeros(n, dtype=np.float32)

        mean = float(self.state.params.get("mean", 0.0))
        sigma = float(self.state.params.get("sigma", 0.2))
        seed = int(self.state.params.get("seed", 0))
        if seed != self._last_seed:
            self._last_seed = seed
            self._rng = np.random.default_rng(seed if seed != 0 else None)

        return self._rng.normal(mean, sigma, size=n).astype(np.float32)
