import numpy as np
import pywt

from core.schema import ParamSpec
from core.wavelet_result import WaveletResult
from wavelets.base import IWaveletPlugin

PLUGIN_META = {
    "id": "builtin:cwt_morlet",
    "name": "CWT: Morlet / |коэфф.| (частотная шкала)",
    "type": "CWT",
    "version": "1.3",
    "description": "Непрерывное вейвлет-преобразование (PyWavelets.cwt). Ось Y формируется в Гц.",
}


def _freq_grid(f_min: float, f_max: float, n: int, spacing: str) -> np.ndarray:
    f_min = float(max(f_min, 1e-6))
    f_max = float(max(f_max, f_min * 1.001))
    n = int(max(2, n))
    spacing = str(spacing).lower()

    if spacing == "log":
        return np.geomspace(f_min, f_max, n).astype(np.float32)
    return np.linspace(f_min, f_max, n).astype(np.float32)


def _scales_from_freqs(wavelet_name: str, freqs_hz: np.ndarray, fs: float) -> np.ndarray:
    """
    Для PyWavelets: freq = scale2frequency(wavelet, scale) * fs
    => scale = scale2frequency^{-1}(freq/fs). Для CWT в PyWavelets удобнее:
    scale = central_frequency(wavelet) * fs / freq
    """
    w = pywt.ContinuousWavelet(wavelet_name)
    cf = float(pywt.central_frequency(w))
    freqs_hz = np.asarray(freqs_hz, dtype=np.float32)
    scales = (cf * float(fs)) / np.maximum(freqs_hz, 1e-6)
    return np.maximum(scales, 1e-6).astype(np.float32)


class WaveletPlugin(IWaveletPlugin):
    def get_parameters_schema(self):
        return [
            ParamSpec(
                key="wavelet",
                label="Вейвлет",
                type="enum",
                default="morl",
                choices=["morl", "mexh", "gaus1", "gaus2", "cgau1", "shan1-1.5"],
                description="Тип непрерывного вейвлета, используемый в CWT.",
                examples=["morl — универсальный вариант", "mexh — хорошо для импульсов"],
            ),
            ParamSpec(
                key="magnitude",
                label="Модуль коэффициентов",
                type="enum",
                default="abs",
                choices=["abs", "power"],
                description="Отображаемая величина: |CWT| или |CWT|^2.",
                examples=["abs — амплитуда", "power — энергия (квадрат модуля)"],
            ),
            ParamSpec(
                key="normalize",
                label="Нормализация",
                type="enum",
                default="none",
                choices=["none", "max", "zscore"],
                description="Нормализация 2D-карты коэффициентов перед отображением.",
                examples=["none — без нормализации", "max — деление на максимум"],
            ),
            ParamSpec(
                key="f_min",
                label="Мин. частота (Гц)",
                type="float",
                default=5.0,
                min=0.01,
                max=10_000_000.0,
                step=0.1,
                description="Нижняя граница частотной сетки для CWT.",
                examples=["5–10 Гц — медленные процессы", "20 Гц — если низ не интересен"],
            ),
            ParamSpec(
                key="f_max",
                label="Макс. частота (Гц)",
                type="float",
                default=300.0,
                min=0.01,
                max=10_000_000.0,
                step=0.1,
                description="Верхняя граница частотной сетки (ограничивается Найквистом).",
                examples=["300 Гц — быстрые детали", "fs/2 — максимально возможная"],
            ),
            ParamSpec(
                key="n_freqs",
                label="Число частотных бинов",
                type="int",
                default=128,
                min=8,
                max=4096,
                step=8,
                description="Количество строк скейлограммы по оси частот. Больше — детальнее, но тяжелее по CPU.",
                examples=["64 — быстрее", "256–512 — высокая детализация", "1024+ — мощные ПК"],
            ),
            ParamSpec(
                key="freq_spacing",
                label="Шкала частот",
                type="enum",
                default="linear",
                choices=["linear", "log"],
                description="Распределение частот: линейно или логарифмически.",
                examples=["linear — равномерная шкала", "log — детализация внизу"],
            ),
        ]

    def transform(self, x: np.ndarray, fs: float, params: dict) -> WaveletResult:
        x = np.require(np.asarray(x, dtype=np.float32), requirements="C")
        fs = float(fs)

        wname = str(params.get("wavelet", "morl"))
        magnitude = str(params.get("magnitude", "abs"))
        normalize = str(params.get("normalize", "none"))

        # --- частотная сетка ---
        f_min = float(params.get("f_min", 5.0))
        f_max = float(params.get("f_max", min(fs / 2.0, 300.0)))
        f_max = min(f_max, fs / 2.0 - 1e-6)

        n_freqs = int(params.get("n_freqs", 128))
        spacing = str(params.get("freq_spacing", "linear"))

        freqs_target = _freq_grid(f_min, f_max, n_freqs, spacing)  # ascending


        # freqs_target — ascending (low -> high)
        # scales по формуле central_frequency * fs / f будут descending (high -> low)
        scales_desc = _scales_from_freqs(wname, freqs_target, fs)

        # pywt.cwt обычно удобнее с ascending scales
        scales_asc = scales_desc[::-1]
        coefs, _ = pywt.cwt(x, scales_asc, wname, sampling_period=1.0 / fs)

        # coefs сейчас соответствуют scales_asc (low freq -> high freq),
        # поэтому разворачиваем обратно так, чтобы ось Y шла low->high как freqs_target
        coefs = coefs[::-1, :]

        img = np.abs(coefs) if magnitude == "abs" else (np.abs(coefs) ** 2)

        if normalize == "max":
            m = float(np.nanmax(img)) if img.size and np.nanmax(img) > 0 else 1.0
            if m > 0:
                img = img / m
        elif normalize == "zscore":
            mu = float(np.nanmean(img))
            sd = float(np.nanstd(img))
            if sd > 1e-9:
                img = (img - mu) / sd

        x_axis = np.arange(len(x), dtype=np.float32) / fs

        return WaveletResult(
            image=img.astype(np.float32),
            y_axis=freqs_target.astype(np.float32),  # линейные частоты (Гц)
            x_axis=x_axis.astype(np.float32),
            label_y="Hz",
            meta={
                "mode": "CWT",
                "wavelet": wname,
                "magnitude": magnitude,
                "normalize": normalize,
                "freq_spacing": str(params.get("freq_spacing", "linear")),
            },
        )
