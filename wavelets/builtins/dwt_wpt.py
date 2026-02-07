import numpy as np
import pywt
from core.schema import ParamSpec
from core.wavelet_result import WaveletResult
from wavelets.base import IWaveletPlugin

PLUGIN_META = {
    "id": "builtin:dwt_wpt",
    "name": "DWT/WPT: детали и узлы (PyWavelets)",
    "type": "DWT/WPT",
    "version": "1.1",
    "description": "Дискретное разложение (DWT) и пакетное разложение (WPT). Визуализация матрицы коэффициентов по уровням/узлам.",
}


def _stretch_to_len(v: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if len(v) == n:
        return v
    if len(v) <= 1:
        return np.zeros(n, dtype=np.float32)
    x0 = np.linspace(0, 1, len(v), dtype=np.float32)
    x1 = np.linspace(0, 1, n, dtype=np.float32)
    return np.interp(x1, x0, v).astype(np.float32)

class WaveletPlugin(IWaveletPlugin):
    def get_parameters_schema(self):
        return [
            ParamSpec(
                key="mode",
                label="Режим",
                type="enum",
                default="WPT",
                choices=["DWT", "WPT"],
                description="Тип разложения: DWT (уровни деталей) или WPT (узлы пакетов).",
                examples=["DWT — компактно по уровням", "WPT — больше детализации по полосам"],
            ),

            ParamSpec(
                key="wavelet",
                label="Семейство вейвлета",
                type="enum",
                default="db4",
                choices=["haar", "db2", "db4", "sym4", "coif1"],
                description="Базовый дискретный вейвлет для DWT/WPT.",
                examples=["haar — простой и быстрый", "db4 — универсальный вариант"],
            ),

            ParamSpec(
                key="maxlevel",
                label="Макс. уровень (ограничение)",
                type="int",
                default=5,
                min=1,
                max=14,
                step=1,
                description="Верхняя граница уровня разложения. Для WPT фактический уровень задаётся отдельно.",
                examples=["5 — типичное значение для окна 2–6 сек", "8–12 — для длинных окон и мощных ПК"],
            ),

            ParamSpec(
                key="show_approx",
                label="DWT: включить аппроксимацию (A)",
                type="bool",
                default=False,
                description="Добавление строки аппроксимации A(L) к матрице DWT (помимо деталей D).",
                examples=["Полезно для просмотра низкочастотной составляющей"],
            ),

            ParamSpec(
                key="wpt_level",
                label="WPT: уровень",
                type="int",
                default=4,
                min=1,
                max=14,
                step=1,
                description="Уровень пакетного разложения WPT. Ограничивается значением maxlevel.",
                examples=["4 — базовый вариант", "6–10 — больше полос, выше нагрузка"],
            ),

            ParamSpec(
                key="wpt_nodes",
                label="WPT: узлы (через запятую, пусто = все)",
                type="str",
                default="",
                description="Список путей узлов WPT для отображения. При пустом значении отображаются все узлы уровня.",
                examples=["aa, ad, da, dd", "Пусто — все узлы на уровне"],
            ),

            ParamSpec(
                key="wpt_select",
                label="WPT: отбор узлов",
                type="enum",
                default="all",
                choices=["all", "top_energy"],
                description="Стратегия выбора узлов: все узлы или топ по энергии.",
                examples=["all — полный уровень", "top_energy — наиболее энергетические полосы"],
            ),

            ParamSpec(
                key="top_k",
                label="WPT: число узлов (top-energy)",
                type="int",
                default=8,
                min=1,
                max=256,
                step=1,
                description="Сколько узлов оставить при выборе по энергии (wpt_select=top_energy).",
                examples=["8 — типично", "32–64 — детализация на мощных ПК"],
            ),

            ParamSpec(
                key="magnitude",
                label="Модуль коэффициентов",
                type="enum",
                default="abs",
                choices=["abs", "power"],
                description="Преобразование коэффициентов в амплитудную карту.",
                examples=["abs — |coef|", "power — |coef|² (энергия)"],
            ),

            ParamSpec(
                key="normalize",
                label="Нормализация",
                type="enum",
                default="none",
                choices=["none", "max", "zscore"],
                description="Нормализация матрицы коэффициентов для сравнимой визуализации.",
                examples=["none — без изменений", "max — деление на максимум", "zscore — (x-μ)/σ"],
            ),
        ]


    def transform(self, x: np.ndarray, fs: float, params: dict) -> WaveletResult:
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        if n < 16:
            img = np.zeros((1, max(n, 1)), dtype=np.float32)
            return WaveletResult(img, np.array([0], dtype=np.float32), np.linspace(0, n / fs, max(n, 1)), "level/node", {})

        wname = str(params.get("wavelet", "db4"))
        mode = str(params.get("mode", "WPT"))
        maxlevel = int(params.get("maxlevel", 5))
        mag_mode = str(params.get("magnitude", "abs"))
        norm = str(params.get("normalize", "none"))

        if mode == "DWT":
            coeffs = pywt.wavedec(x, wname, level=maxlevel)
            # coeffs: [cA_L, cD_L, cD_{L-1}, ... cD_1]
            rows = []
            labels = []
            show_approx = bool(params.get("show_approx", False))
            if show_approx:
                cA = _stretch_to_len(coeffs[0], n)
                rows.append(cA)
                labels.append(f"A{maxlevel}")

            # details
            for i, cD in enumerate(coeffs[1:], start=1):
                level = maxlevel - (i - 1)
                v = _stretch_to_len(cD, n)
                rows.append(v)
                labels.append(f"D{level}")

            img = np.vstack(rows).astype(np.float32)
            if mag_mode == "abs":
                img = np.abs(img)
            else:
                img = np.abs(img) ** 2

            y_axis = np.arange(img.shape[0], dtype=np.float32)
            x_axis = np.arange(n, dtype=np.float32) / float(fs)
            meta = {"mode": "DWT", "wavelet": wname, "labels": labels, "level": maxlevel}

        else:
            wpt_level = int(params.get("wpt_level", 4))
            wpt_level = min(wpt_level, maxlevel)
            wp = pywt.WaveletPacket(data=x, wavelet=wname, mode="symmetric", maxlevel=wpt_level)

            nodes_str = str(params.get("wpt_nodes", "")).strip()
            all_nodes = [n_.path for n_ in wp.get_level(wpt_level, order="freq")]

            selected = all_nodes
            if nodes_str:
                wanted = [s.strip() for s in nodes_str.split(",") if s.strip()]
                selected = [p for p in all_nodes if p in wanted]

            select_mode = str(params.get("wpt_select", "all"))
            if select_mode == "top_energy":
                energies = []
                for p in selected:
                    v = wp[p].data
                    energies.append((p, float(np.sum(np.asarray(v) ** 2))))
                energies.sort(key=lambda t: t[1], reverse=True)
                k = int(params.get("top_k", 8))
                selected = [p for p, _ in energies[:k]]

            rows = []
            labels = []
            for p in selected:
                v = wp[p].data
                v = _stretch_to_len(v, n)
                rows.append(v)
                labels.append(p)

            if not rows:
                rows = [np.zeros(n, dtype=np.float32)]
                labels = ["(none)"]

            img = np.vstack(rows).astype(np.float32)
            if mag_mode == "abs":
                img = np.abs(img)
            else:
                img = np.abs(img) ** 2

            y_axis = np.arange(img.shape[0], dtype=np.float32)
            x_axis = np.arange(n, dtype=np.float32) / float(fs)
            meta = {"mode": "WPT", "wavelet": wname, "level": wpt_level, "labels": labels}

        # normalize (nan* for consistency with CWT and robustness to outliers)
        if norm == "max":
            m = float(np.nanmax(img)) if img.size and np.nanmax(img) > 0 else 1.0
            img = img / m
        elif norm == "zscore":
            mu = float(np.nanmean(img))
            sd = float(np.nanstd(img))
            if sd > 1e-9:
                img = (img - mu) / sd

        return WaveletResult(
            image=img.astype(np.float32),
            y_axis=y_axis,
            x_axis=x_axis,
            label_y="Уровень/узел",
            meta=meta
        )
