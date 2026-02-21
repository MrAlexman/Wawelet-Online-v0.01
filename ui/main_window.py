from __future__ import annotations

# --- standard library ---
import json
import os
import logging
import time

import numpy as np

# --- Qt / PySide6 ---
from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, Slot, QRectF, QTimer, QEvent
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QCheckBox,
    QStatusBar, QDialog, QScrollArea,
    QSplitter, QSizePolicy
)

# --- pyqtgraph ---
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

# --- project core ---
from core.ring_buffer import RingBuffer
from core.realtime_clock import RealtimeClock
from core.plugin_manager import PluginManager
from core.params_model import ParamsModel

# --- signal generation ---
from signal.signal_engine import SignalEngine
from signal.components import BUILTIN_COMPONENTS

# --- UI widgets ---
from ui.widgets.component_editor import ComponentEditor
from ui.widgets.param_form import ParamForm




class _NoWheelUnlessFocused(QObject):
    """Ignore mouse wheel unless widget has focus (prevents accidental changes)."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if hasattr(obj, "hasFocus") and not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)

log = logging.getLogger("MainWindow")

# ---------------- Workers ----------------

class GeneratorWorker(QObject):
    chunk_ready = Signal(object, float, float)  # x(np.ndarray), t0, fs
    started = Signal()
    stopped = Signal()

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._running = False

    @Slot()
    def start(self):
        self._running = True
        self.started.emit()

        while self._running:
            fs, n = self.engine.get_global()
            dt = max(0.001, n / fs)

            # Если PAUSE — ничего не генерируем и не шлем в UI (замораживаем картинку)
            if self.engine.is_paused():
                time.sleep(0.05)
                continue

            x, t0 = self.engine.generate_chunk()
            self.chunk_ready.emit(x, t0, fs)

            time.sleep(dt)

        self.stopped.emit()

    @Slot()
    def stop(self):
        self._running = False

# Cap samples passed to wavelet to avoid OOM/hangs (CWT is O(n * n_freqs)).
# Use last N samples when window is larger. 4M supports ~2s @ 2 MHz or ~20s @ 200 kHz.
MAX_WAVELET_SAMPLES = 2**22

class WaveletWorker(QObject):
    result_ready = Signal(object)  # WaveletResult
    status = Signal(str)

    def __init__(self, plugin_manager, params, ring):
        super().__init__()
        self.pm = plugin_manager
        self.params = params
        self.ring = ring
        self._running = False

    @Slot()
    def start(self):
        self._running = True

        while self._running:
            snap = self.params.snapshot()
            plugin_id = snap.get("wavelet_plugin_id", "builtin:cwt_morlet")
            fs = float(snap.get("fs", 2000.0))
            window_sec = float(snap.get("view_window_sec", 4.0))
            n = min(MAX_WAVELET_SAMPLES, int(max(16, window_sec * fs)))
            fps = max(1, int(snap.get("scalogram_fps", 8)))

            try:
                x = self.ring.get_last(n)
                info = self.pm.get_wavelet(plugin_id)
                if not info:
                    self.status.emit(f"Wavelet plugin not found: {plugin_id}")
                else:
                    plugin = info.plugin_obj
                    wparams = snap.get("wavelet_params", {})
                    res = plugin.transform(x, fs, dict(wparams))
                    self.result_ready.emit(res)
            except Exception as e:
                self.status.emit(f"Wavelet error: {e}")

            time.sleep(1.0 / float(fps))

    @Slot()
    def stop(self):
        self._running = False


# ---------------- Main Window ----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Вейвлет-преобразование в реальном времени")
        self.resize(1400, 800)
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.pm = PluginManager(self.project_root)
        self.pm.reload_all()

        self.clock = RealtimeClock()
        self.engine = SignalEngine(self.clock)

        # default components
        self.engine.add_component("noise", {"sigma": 0.15}, enabled=True)
        self.engine.add_component("sine", {"frequency": 6.0, "amplitude": 1.0}, enabled=True)
        self.engine.add_component("sine", {"frequency": 30.0, "amplitude": 0.4}, enabled=True)

        # global params model (thread-safe snapshot)
        self.params = ParamsModel({
            "fs": 2000.0,
            "chunk_size": 256,
            "view_window_sec": 4.0,
            "wavelet_plugin_id": "builtin:cwt_morlet",
            "wavelet_params": {},   # filled after plugin selection
            "scalogram_fps": 8,
        })

        self.ring = RingBuffer(capacity=int(self.params.get("fs") * 60))  # keep 60 sec history (high‑freq)

        self._no_wheel_filter = _NoWheelUnlessFocused(self)
        self._build_ui()

        self._apply_plugin_default_params()

        self._refresh_component_table()
        self._wire_threads()

        self._ui_timer = QTimer()
        self._ui_timer.setTimerType(Qt.PreciseTimer)
        self._ui_timer.timeout.connect(self._ui_update)
        self._ui_timer.start(50)  # ~20 FPS

        self._paused = True
        self._latest_chunk = np.zeros(0, dtype=np.float32)
        self._latest_fs = float(self.params.get("fs", 2000.0))

        self._last_wavelet = None


    def _tt(self, title: str, body: str, examples: list[str] | None = None) -> str:
        ex = ""
        if examples:
            items = "".join([f"<li><code>{e}</code></li>" for e in examples])
            ex = f"<br><b>Примеры:</b><ul>{items}</ul>"
        return f"<b>{title}</b><br>{body}{ex}"

    # ---------- UI building ----------

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)

        # Горизонтальный splitter: слева графики, справа панель управления
        self.main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.main_splitter, 1)

        # Left: plots (как QWidget, чтобы добавлять в splitter)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(8, 8, 8, 8)
        left.setSpacing(8)

        self.main_splitter.addWidget(left_widget)

        pg.setConfigOptions(antialias=False)  # useOpenGL=True можно включить для ускорения, но бывает нестабильно

        # Time plot
        self.time_plot = pg.PlotWidget(title="Сигнал во времени")
        self.time_curve = self.time_plot.plot([], [])
        self.time_plot.showGrid(x=True, y=True, alpha=0.2)
        left.addWidget(self.time_plot, 1)

        # Scalogram (Plot + Colorbar)
        self.scalo_glw = pg.GraphicsLayoutWidget()
        self.scalo_plot = self.scalo_glw.addPlot(title="Скейлограмма / карта коэффициентов")
        self.scalo_img = pg.ImageItem()
        self.scalo_plot.addItem(self.scalo_img)
        self.scalo_plot.setLabel("left", "Частота / уровень")
        self.scalo_plot.setLabel("bottom", "Время (с)")
        self.scalo_plot.showGrid(x=True, y=True, alpha=0.2)

        # Colorbar (HistogramLUTItem)
        self.scalo_hist = pg.HistogramLUTItem()
        self.scalo_hist.setImageItem(self.scalo_img)
        self.scalo_hist.gradient.loadPreset("viridis")
        
        # --- сохранить состояние градиента "по умолчанию" (ticks/цвета) ---
        try:
            self._default_gradient_state = self.scalo_hist.gradient.saveState()
        except Exception:
            self._default_gradient_state = None

        self.scalo_glw.addItem(self.scalo_hist)

        # layout tuning: plot wider than colorbar
        self.scalo_glw.ci.layout.setColumnStretchFactor(0, 10)
        self.scalo_glw.ci.layout.setColumnStretchFactor(1, 2)

        left.addWidget(self.scalo_glw, 2)

        # Crosshair
        self._ch_v = pg.InfiniteLine(angle=90, movable=False)
        self._ch_h = pg.InfiniteLine(angle=0, movable=False)
        self.scalo_plot.addItem(self._ch_v, ignoreBounds=True)
        self.scalo_plot.addItem(self._ch_h, ignoreBounds=True)

        # last scalogram mapping state
        self._last_img_disp = None       # (time, rows) float32
        self._last_x0 = 0.0
        self._last_x1 = 1.0
        self._last_y_axis = None         # freqs or scales or labels array
        self._last_label_y = ""
        self._last_meta = {}

        # Mouse move handler (rate-limited)
        self._mouse_proxy = pg.SignalProxy(
            self.scalo_plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_scalo_mouse_moved
        )


        # Right: controls (SCROLLABLE)

        # Контент правой панели
        right_content = QWidget()
        right = QVBoxLayout(right_content)
        right.setContentsMargins(8, 8, 8, 8)
        right.setSpacing(8)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)

        # Если какой-то элемент шире панели — появляется горизонтальный скролл, а не “обрезка”
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setWidget(right_content)

        # Минимальная ширина панели управления (максимум НЕ ограничиваем)
        right_scroll.setMinimumWidth(360)
        right_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Добавляем правую панель в splitter (а не в root layout)
        self.main_splitter.addWidget(right_scroll)

        # Поведение splitter
        self.main_splitter.setStretchFactor(0, 3)  # графики
        self.main_splitter.setStretchFactor(1, 1)  # панель управления
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)



        # Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("СТАРТ (сброс)")
        self.btn_play = QPushButton("ПУСК")
        self.btn_reload = QPushButton("Перезагрузить плагины")
        
        self.btn_start.setToolTip(self._tt(
            "СТАРТ (сброс)",
            "Сбрасывает время и очищает буфер. Запускает генерацию заново.",
            ["Запуск с очисткой буфера и сбросом времени"]
        ))
        self.btn_play.setToolTip(self._tt(
            "ПУСК / ПАУЗА",
            "Пауза фиксирует текущие данные на экране. Пуск включает обновление в реальном времени.",
            ["Пауза: фиксирует отображение", "Пуск: режим реального времени"]
        ))

        self.btn_reload.setToolTip(self._tt(
            "Перезагрузить плагины",
            "Перечитывает плагины вейвлетов/компонентов без закрытия приложения.",
            ["Применяется после изменения файлов плагинов"]
        ))
                
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_reload)
        right.addLayout(btn_row)

        # Global params
        right.addWidget(QLabel("Параметры генерации"))
        g_row = QHBoxLayout()
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(100, 2_000_000)
        self.spin_fs.setValue(float(self.params.get("fs")))
        
        self.spin_fs.installEventFilter(self._no_wheel_filter)
        self.spin_fs.setFocusPolicy(Qt.StrongFocus)
        self.spin_fs.setKeyboardTracking(False)
        
        self.spin_fs.setDecimals(1)
        self.spin_fs.setSingleStep(100)
        
        self.spin_fs.setToolTip(self._tt(
            "Частота дискретизации (fs)",
            "Сколько отсчётов в секунду. Влияет на частотную шкалу скейлограммы и точность анализа.",
            ["fs=2000 — тесты", "fs=50k–500k — высокочастотные измерения"]
        ))

        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(64, 131072)
        self.spin_chunk.setValue(int(self.params.get("chunk_size")))
        
        self.spin_chunk.installEventFilter(self._no_wheel_filter)
        self.spin_chunk.setFocusPolicy(Qt.StrongFocus)
        self.spin_chunk.setKeyboardTracking(False)

        
        self.spin_chunk.setSingleStep(256)
        
        self.spin_chunk.setToolTip(self._tt(
            "Размер чанка (chunk)",
            "Сколько отсчётов генерируется за один шаг. Больше — меньше нагрузка на UI, но больше задержка.",
            ["256–2048 — типично", "8192+ — для высоких fs на мощных ПК"]
        ))

        

        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.1, 300.0)
        self.spin_window.setValue(float(self.params.get("view_window_sec")))
        
        self.spin_window.installEventFilter(self._no_wheel_filter)
        self.spin_window.setFocusPolicy(Qt.StrongFocus)
        self.spin_window.setKeyboardTracking(False)

        
        self.spin_window.setDecimals(2)
        self.spin_window.setSingleStep(0.5)
        
        self.spin_window.setToolTip(self._tt(
            "Окно просмотра (сек)",
            "Сколько секунд последних данных показываем и анализируем в скейлограмме.",
            ["2–6 сек — эксперименты", "10–60 сек — длинные записи", "до 300 сек — мощные ПК"]
        ))


        g_row.addWidget(QLabel("fs"))
        g_row.addWidget(self.spin_fs)
        g_row.addWidget(QLabel("чанк"))
        g_row.addWidget(self.spin_chunk)
        g_row.addWidget(QLabel("окно (с)"))
        g_row.addWidget(self.spin_window)
        right.addLayout(g_row)

        fps_row = QHBoxLayout()
        self.spin_sfps = QSpinBox()
        self.spin_sfps.setRange(1, 120)
        self.spin_sfps.setValue(int(self.params.get("scalogram_fps")))
        
        self.spin_sfps.installEventFilter(self._no_wheel_filter)
        self.spin_sfps.setFocusPolicy(Qt.StrongFocus)
        self.spin_sfps.setKeyboardTracking(False)

        self.spin_sfps.setToolTip(self._tt(
            "FPS скейлограммы",
            "Как часто пересчитывать вейвлет-анализ. Выше FPS — плавнее картинка, выше нагрузка на CPU.",
            ["5–15 FPS — типично", "30–60 FPS — мощные ПК", "до 120 — высокочастотный мониторинг"]
        ))
        
        fps_row.addWidget(QLabel("FPS скейлограммы"))
        fps_row.addWidget(self.spin_sfps)
        right.addLayout(fps_row)
        
        # Scalogram UX controls
        self.cb_auto_levels = QCheckBox("Автоконтраст скейлограммы")
        
        self.cb_auto_levels.setToolTip(self._tt(
            "Автоконтраст",
            "Автоматически подбирает контраст (levels) при каждом обновлении скейлограммы.",
            ["Рекомендуется отключать при «скачущей» яркости"]
        ))

        self.cb_auto_levels.setChecked(True)
        right.addWidget(self.cb_auto_levels)

        self.btn_reset_levels = QPushButton("Сбросить уровни (контраст)")
        
        self.btn_reset_levels.setToolTip(self._tt(
            "Сброс уровней",
            "Ставит контраст по 1–99 перцентилям текущей картинки.",
            ["Применять после отключения автоконтраста"]
        ))

        
        right.addWidget(self.btn_reset_levels)
        self.btn_reset_levels.clicked.connect(self._reset_scalo_levels)


        # Wavelet plugin dropdown
        right.addWidget(QLabel("Вейвлет-плагин"))
        self.combo_wavelet = QComboBox()
        
        self.combo_wavelet.setToolTip(self._tt(
            "Выбор вейвлет-анализа",
            "CWT показывает карту «частота (Гц) ↔ время». "
            "DWT/WPT показывает уровни/узлы разложения (не в Гц).",
            [
                "CWT: синус 30 Гц → полоса около 30 Гц",
                "WPT: удобно для импульсов и пачек"
            ]
        ))



        
        right.addWidget(self.combo_wavelet)
        
        lbl_wp = QLabel("Параметры выбранного анализа")
        lbl_wp.setToolTip(self._tt(
            "Параметры анализа",
            "Набор параметров зависит от выбранного плагина. Подсказки доступны при наведении.",
            ["CWT: частотная сетка и отображение |коэфф.|", "DWT/WPT: уровни и узлы разложения"]
        ))
        right.addWidget(lbl_wp)

        self.combo_wavelet.installEventFilter(self._no_wheel_filter)
        self.combo_wavelet.setFocusPolicy(Qt.StrongFocus)


        self.wavelet_form_holder = QWidget()
        self.wavelet_form_layout = QVBoxLayout(self.wavelet_form_holder)
        self.wavelet_form_layout.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.wavelet_form_holder, 1)

        # =========================
        # Components (resizable) + Presets (resizable) via splitter
        # =========================

        # --- Components widget ---
        comp_widget = QWidget()
        comp_layout = QVBoxLayout(comp_widget)
        comp_layout.setContentsMargins(0, 0, 0, 0)
        comp_layout.setSpacing(6)

        comp_layout.addWidget(QLabel("Компоненты сигнала"))

        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["Тип", "Вкл", "Правка", "Удалить", "Параметры"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # ВАЖНО: таблица должна уметь расширяться
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tbl.setMinimumHeight(220)

        comp_layout.addWidget(self.tbl)

        add_row = QHBoxLayout()
        self.btn_add_noise = QPushButton("+ Шум")
        self.btn_add_sine = QPushButton("+ Синус")
        self.btn_add_rect = QPushButton("+ Прямоуг. импульсы")
        self.btn_add_gauss = QPushButton("+ Гаусс-импульсы")
        self.btn_add_chirp = QPushButton("+ Чирп")
        
        self.btn_add_noise.setToolTip(self._tt(
            "Шум",
            "Добавляет белый гауссов шум. Удобно для проверки устойчивости анализа.",
            ["sigma=0.1 — лёгкий шум", "sigma=0.5 — сильный шум"]
        ))
        self.btn_add_sine.setToolTip(self._tt(
            "Синус",
            "Добавляет синусоиду A·sin(2πft+φ).",
            ["f=30 Hz, A=1.0 — базовый тест", "Две синусоиды: 6 Hz и 30 Hz"]
        ))
        self.btn_add_rect.setToolTip(self._tt(
            "Прямоугольные импульсы",
            "Пачка прямоугольных импульсов (pulse train). Хорошо видна локализация по времени.",
            ["period=0.5s, width=0.05s"]
        ))
        self.btn_add_gauss.setToolTip(self._tt(
            "Гаусс-импульсы",
            "Гладкие импульсы. Полезно для проверки формы пятна на скейлограмме.",
            ["sigma=0.02s, period=0.5s"]
        ))
        self.btn_add_chirp.setToolTip(self._tt(
            "Чирп",
            "Тон с изменяющейся частотой (тест time-frequency локализации).",
            ["f0=10 Hz → f1=120 Hz за 2 секунды"]
        ))
        
        
        for b in [self.btn_add_noise, self.btn_add_sine, self.btn_add_rect, self.btn_add_gauss, self.btn_add_chirp]:
            add_row.addWidget(b)
        comp_layout.addLayout(add_row)

        # --- Presets/Export widget ---
        pres_widget = QWidget()
        pres_layout = QVBoxLayout(pres_widget)
        pres_layout.setContentsMargins(0, 0, 0, 0)
        pres_layout.setSpacing(6)

        pres_layout.addWidget(QLabel("Пресеты и экспорт"))


        p_row = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить пресет")
        self.btn_save = QPushButton("Сохранить пресет")
        p_row.addWidget(self.btn_load)
        p_row.addWidget(self.btn_save)
        pres_layout.addLayout(p_row)

        # Кнопки экспорта (как у тебя было)
        self.btn_export_csv = QPushButton("Экспорт сигнала (CSV/NPY)")
        self.btn_export_png = QPushButton("Экспорт скейлограммы (PNG)")
        pres_layout.addWidget(self.btn_export_csv)
        pres_layout.addWidget(self.btn_export_png)

        # --- Splitter: верх = components, низ = presets/export ---
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.addWidget(comp_widget)
        self.right_splitter.addWidget(pres_widget)

        self.right_splitter.setStretchFactor(0, 3)  # components больше
        self.right_splitter.setStretchFactor(1, 1)  # presets меньше
        self.right_splitter.setCollapsible(0, False)
        self.right_splitter.setCollapsible(1, False)

        # Добавляем splitter в правую панель
        right.addWidget(self.right_splitter, 1)
        
        right.addStretch(1)

        # Wire UI signals
        self.btn_start.clicked.connect(self.on_start)
        self.btn_play.clicked.connect(self.on_play_pause)
        self.btn_reload.clicked.connect(self.on_reload_plugins)

        self.spin_fs.valueChanged.connect(self.on_global_changed)
        self.spin_chunk.valueChanged.connect(self.on_global_changed)
        self.spin_window.valueChanged.connect(self.on_global_changed)
        self.spin_sfps.valueChanged.connect(self.on_scalo_fps_changed)

        self.btn_add_noise.clicked.connect(lambda: self.on_add_component("noise"))
        self.btn_add_sine.clicked.connect(lambda: self.on_add_component("sine"))
        self.btn_add_rect.clicked.connect(lambda: self.on_add_component("rect_pulse"))
        self.btn_add_gauss.clicked.connect(lambda: self.on_add_component("gauss_pulse"))
        self.btn_add_chirp.clicked.connect(lambda: self.on_add_component("chirp"))

        self.btn_load.clicked.connect(self.on_load_preset)
        self.btn_save.clicked.connect(self.on_save_preset)
        self.btn_export_csv.clicked.connect(self.on_export_signal)
        self.btn_export_png.clicked.connect(self.on_export_png)

        self._fill_wavelet_combo()

        self.combo_wavelet.currentIndexChanged.connect(self.on_wavelet_changed)

        # mouse interactions for time plot (zoom/pan are built-in)
        self.time_plot.setMouseEnabled(x=True, y=False)
        self.scalo_plot.setMouseEnabled(x=True, y=True)
        
        self.scalo_plot.getViewBox().sigRangeChanged.connect(lambda *_: self._refresh_scalo_y_ticks())




    def _fill_wavelet_combo(self):
        self.combo_wavelet.clear()
        items = self.pm.list_wavelets()
        for info in items:
            self.combo_wavelet.addItem(f"{info.meta.get('name')}  [{info.meta.get('type')}]", userData=info.id)

        # set current to model value
        cur_id = self.params.get("wavelet_plugin_id")
        idx = 0
        for i in range(self.combo_wavelet.count()):
            if self.combo_wavelet.itemData(i) == cur_id:
                idx = i
                break
        self.combo_wavelet.setCurrentIndex(idx)

    def _apply_plugin_default_params(self):
        plugin_id = self.params.get("wavelet_plugin_id")
        info = self.pm.get_wavelet(plugin_id)
        if not info:
            return
        schema = info.plugin_obj.get_parameters_schema()
        defaults = {s.key: s.default for s in schema}
        self.params.set("wavelet_params", defaults)
        self._rebuild_wavelet_form(schema, defaults)

    def _rebuild_wavelet_form(self, schema, initial):
        # clear
        while self.wavelet_form_layout.count():
            item = self.wavelet_form_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.wavelet_form = ParamForm(schema, initial=initial, parent=self.wavelet_form_holder)
        self.wavelet_form_layout.addWidget(self.wavelet_form)

        # connect change events (lightweight: poll via timer is also ok; here we bind common widgets)
        for key, (_, w) in self.wavelet_form.widgets.items():
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.on_wavelet_params_changed)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self.on_wavelet_params_changed)
            elif hasattr(w, "textChanged"):
                w.textChanged.connect(self.on_wavelet_params_changed)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.on_wavelet_params_changed)

    # ---------- Threads ----------

    def _wire_threads(self):
        # Generator thread
        self.gen_thread = QThread(self)
        self.gen_worker = GeneratorWorker(self.engine)
        self.gen_worker.moveToThread(self.gen_thread)

        # ВАЖНО: запускаем worker.start() когда поток стартует
        self.gen_thread.started.connect(self.gen_worker.start)
        self.gen_worker.chunk_ready.connect(self.on_chunk)

        # Wavelet thread
        self.wav_thread = QThread(self)
        self.wav_worker = WaveletWorker(self.pm, self.params, self.ring)
        self.wav_worker.moveToThread(self.wav_thread)

        self.wav_thread.started.connect(self.wav_worker.start)
        self.wav_worker.result_ready.connect(self.on_wavelet_result)
        self.wav_worker.status.connect(self.statusbar.showMessage)

        self.gen_thread.start()
        self.wav_thread.start()

        # initially paused
        self.engine.pause()
        self._paused = True


    # ---------- Slots ----------

    def closeEvent(self, event: QCloseEvent):
        self._ui_timer.stop()
        self.gen_worker.stop()
        self.wav_worker.stop()
        self.gen_thread.quit()
        self.wav_thread.quit()
        if not self.gen_thread.wait(3000):
            log.warning("Generator thread did not finish in time")
        if not self.wav_thread.wait(3000):
            log.warning("Wavelet thread did not finish in time")
        event.accept()
        super().closeEvent(event)

    @Slot()
    def on_start(self):
        # reset clock + ring
        self.engine.reset()
        self.ring.clear()
        self.statusbar.showMessage("Старт: сброс времени и очистка буфера", 2000)

        # start in play mode
        self.engine.play()
        self._paused = False
        self.btn_play.setText("ПАУЗА")

    @Slot()
    def on_play_pause(self):
        if self._paused:
            self.engine.play()
            self._paused = False
            self.btn_play.setText("ПАУЗА")
            self.statusbar.showMessage("ПУСК", 1500)
        else:
            self.engine.pause()
            self._paused = True
            self.btn_play.setText("ПУСК")
            self.statusbar.showMessage("ПАУЗА", 1500)

    @Slot()
    def on_reload_plugins(self):
        self.pm.reload_all()
        self._fill_wavelet_combo()
        self._apply_plugin_default_params()
        self.statusbar.showMessage("Плагины перезагружены", 2000)

    @Slot()
    def on_global_changed(self):
        fs = float(self.spin_fs.value())
        chunk = int(self.spin_chunk.value())
        window = float(self.spin_window.value())

        self.params.set("fs", fs)
        self.params.set("chunk_size", chunk)
        self.params.set("view_window_sec", window)

        # update engine and ring capacity (60 s history for high‑freq)
        self.engine.set_global(fs=fs, chunk_size=chunk)
        self.ring = RingBuffer(capacity=int(fs * 60))
        # rebind ring for wavelet worker
        self.wav_worker.ring = self.ring

        self.statusbar.showMessage("Параметры генерации обновлены", 1000)

    @Slot()
    def on_scalo_fps_changed(self):
        fps = int(self.spin_sfps.value())
        self.params.set("scalogram_fps", fps)
        self.statusbar.showMessage(f"FPS скейлограммы: {fps}", 1000)



    @Slot()
    def on_wavelet_changed(self):
        plugin_id = self.combo_wavelet.currentData()
        if not plugin_id:
            return
        self.params.set("wavelet_plugin_id", plugin_id)
        info = self.pm.get_wavelet(plugin_id)
        if not info:
            return
        schema = info.plugin_obj.get_parameters_schema()
        defaults = {s.key: s.default for s in schema}
        self.params.set("wavelet_params", defaults)
        self._rebuild_wavelet_form(schema, defaults)
        self.statusbar.showMessage(f"Выбран анализ: {info.meta.get('name')}", 1500)

    @Slot()
    def on_wavelet_params_changed(self):
        vals = self.wavelet_form.values()
        self.params.set("wavelet_params", vals)

    @Slot(object, float, float)
    def on_chunk(self, x, t0, fs):
        self._latest_chunk = np.asarray(x, dtype=np.float32)
        self._latest_fs = float(fs)
        self.ring.append(self._latest_chunk)

    @Slot(object)
    def on_wavelet_result(self, res):
        self._last_wavelet = res
        self._update_scalogram(res)

    # ---------- UI update loop ----------

    def _ui_update(self):
        fs = self._latest_fs
        window = float(self.params.get("view_window_sec", 4.0))
        n = int(max(16, window * fs))
        x = self.ring.get_last(n)
        t = np.arange(len(x), dtype=np.float32) / fs

        self.time_curve.setData(t, x)
        self.time_plot.setLabel("bottom", "Время (с)")
        self.time_plot.setLabel("left", "Амплитуда")

    def _refresh_scalo_y_ticks(self):
        """Пересчитать подписи оси Y для скейлограммы по текущему viewRange."""
        if self._last_img_disp is None:
            return

        img_disp = self._last_img_disp  # shape: (time, rows)
        rows = int(img_disp.shape[1])

        if self._last_label_y == "Hz" and (self._last_y_axis is not None):
            freqs = np.asarray(self._last_y_axis, dtype=np.float32)
            if len(freqs) == rows and rows >= 2:
                vb = self.scalo_plot.getViewBox()
                (_, y_rng) = vb.viewRange()
                y_min, y_max = y_rng[0], y_rng[1]

                i0 = int(max(0, np.floor(y_min)))
                i1 = int(min(rows - 1, np.ceil(y_max)))
                if i1 <= i0:
                    i0, i1 = 0, rows - 1

                visible_rows = max(1, i1 - i0 + 1)
                n_ticks = int(np.clip(visible_rows / 10, 6, 14))
                idxs = np.linspace(i0, i1, n_ticks).astype(int)

                ticks = []
                for i in idxs:
                    f = float(freqs[i])
                    if f >= 1000:
                        label = f"{f/1000:.3f}k"
                    elif f >= 10:
                        label = f"{f:.1f}"
                    else:
                        label = f"{f:.3f}"
                    ticks.append((float(i), label))

                self.scalo_plot.getAxis("left").setTicks([ticks])
                return

        # fallback: без тиков (DWT/WPT или нет данных)
        self.scalo_plot.getAxis("left").setTicks([])

    def _update_scalogram(self, res):
        img = np.asarray(res.image, dtype=np.float32)
        if img.ndim != 2:
            return

        # Логика: res.image = (rows = freq/scale/node, cols = time)
        # Для ImageItem удобнее хранить (x=time, y=row_index) => транспонируем.
        img_disp = img.T  # shape: (time, rows)
        
        self._last_img_disp = img_disp
        self._last_label_y = res.label_y
        self._last_meta = res.meta or {}

        
        auto_levels = bool(self.cb_auto_levels.isChecked())
        self.scalo_img.setImage(img_disp, autoLevels=auto_levels)

        # Если автоконтраст выключен — держим стрелки (region) синхронно с levels картинки
        if not auto_levels:
            try:
                lv = self.scalo_img.levels  # (min,max) или None
                if lv is not None:
                    self.scalo_hist.region.setRegion((float(lv[0]), float(lv[1])))
            except Exception:
                pass

        # X-axis = time (сек)
        x_axis = np.asarray(res.x_axis, dtype=np.float32)
        if len(x_axis) >= 2:
            x0 = float(x_axis[0])
            x1 = float(x_axis[-1])
        else:
            x0, x1 = 0.0, float(img_disp.shape[0])
        
        self._last_x0 = x0
        self._last_x1 = x1
        self._last_y_axis = np.asarray(res.y_axis, dtype=np.float32) if res.y_axis is not None else None


        # Y-axis = индекс строки (0..rows)
        rows = img.shape[0]
        y0, y1 = 0.0, float(rows)

        w = max(1e-9, x1 - x0)
        h = max(1e-9, y1 - y0)

        # Привязываем изображение к координатам (time, row_index)
        self.scalo_img.setRect(QRectF(x0, y0, w, h))

        # Подписи
        self.scalo_plot.setLabel("bottom", "Время (с)")

        # Если это CWT с Hz — ставим тики оси Y по реальным частотам
        if res.label_y == "Hz":
            self.scalo_plot.setLabel("left", "Частота (Гц)")

            freqs = np.asarray(res.y_axis, dtype=np.float32)
            if len(freqs) == rows and rows >= 2:
                # Берем текущий видимый диапазон по Y (индексы строк)
                vb = self.scalo_plot.getViewBox()
                (x_rng, y_rng) = vb.viewRange()
                y_min, y_max = y_rng[0], y_rng[1]

                i0 = int(max(0, np.floor(y_min)))
                i1 = int(min(rows - 1, np.ceil(y_max)))
                if i1 <= i0:
                    i0, i1 = 0, rows - 1

                # Чем сильнее зум — тем больше подписей можно показывать
                visible_rows = max(1, i1 - i0 + 1)
                n_ticks = int(np.clip(visible_rows / 10, 6, 14))  # 6..14 тиков

                idxs = np.linspace(i0, i1, n_ticks).astype(int)

                ticks = []
                for i in idxs:
                    f = float(freqs[i])
                    if f >= 1000:
                        label = f"{f/1000:.3f}k"
                    elif f >= 10:
                        label = f"{f:.1f}"
                    else:
                        label = f"{f:.3f}"
                    ticks.append((float(i), label))

                self.scalo_plot.getAxis("left").setTicks([ticks])
            else:
                self.scalo_plot.getAxis("left").setTicks([])
        else:
            # DWT/WPT: ось Y = индекс строки (узел/уровень)
            self.scalo_plot.setLabel("left", res.label_y)
            self.scalo_plot.getAxis("left").setTicks([])

        # Для DWT/WPT — подсказка в статусе
        meta = res.meta or {}
        if "labels" in meta:
            labels = meta["labels"]
            self.statusbar.showMessage(
                f"{meta.get('mode')} {meta.get('wavelet')} | rows: {len(labels)} | {labels[:6]}{' …' if len(labels)>6 else ''}",
                1500
            )

    # ---------- Components UI ----------

    def _reset_scalo_levels(self):
        """Сброс контраста + восстановление оси Y (тики как при запуске)."""
        if self._last_img_disp is None:
            return

        # 1) контраст
        img = self._last_img_disp
        vmin = float(np.nanpercentile(img, 1))
        vmax = float(np.nanpercentile(img, 99))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        self.scalo_img.setLevels((vmin, vmax))
        
        # Важно: при ручном сбросе фиксируем уровни и отключаем автоконтраст,
        # иначе следующий кадр снова переставит стрелки.
        self.cb_auto_levels.setChecked(False)

        # 2.1) восстановить "стрелки" (ticks) на цветовой шкале как при старте
        try:
            if getattr(self, "_default_gradient_state", None) is not None:
                self.scalo_hist.gradient.restoreState(self._default_gradient_state)
            else:
                # запасной вариант: просто вернуть пресет
                self.scalo_hist.gradient.loadPreset("viridis")
        except Exception:
            pass

        # 2) маркеры HistogramLUT (если есть)
        try:
            self.scalo_hist.region.setRegion((vmin, vmax))
        except Exception:
            pass

        # 3) вернуть диапазон по Y "как при запуске" (все строки 0..rows)
        rows = int(img.shape[1])  # img shape: (time, rows)
        vb = self.scalo_plot.getViewBox()
        vb.setRange(yRange=(0.0, float(rows)), padding=0.0)


    def _on_scalo_mouse_moved(self, evt):
        """Crosshair + readout in status bar."""
        pos = evt[0]
        if not self.scalo_plot.sceneBoundingRect().contains(pos):
            return

        vb = self.scalo_plot.getViewBox()
        p = vb.mapSceneToView(pos)
        x = float(p.x())
        y = float(p.y())

        # Move crosshair
        self._ch_v.setPos(x)
        self._ch_h.setPos(y)

        if self._last_img_disp is None:
            self.statusbar.showMessage(f"t={x:.3f}s, y={y:.2f}", 0)
            return

        img = self._last_img_disp
        n_time = img.shape[0]
        n_rows = img.shape[1]

        # map x (seconds) -> time index
        x0, x1 = self._last_x0, self._last_x1
        if x1 <= x0 or n_time <= 1:
            it = 0
        else:
            it = int(round((x - x0) / (x1 - x0) * (n_time - 1)))
        it = max(0, min(n_time - 1, it))

        # map y -> row index
        ir = int(round(y))
        ir = max(0, min(n_rows - 1, ir))

        val = float(img[it, ir])

        # human y value
        label_y = self._last_label_y
        y_text = f"row={ir}"
        if label_y == "Hz" and self._last_y_axis is not None and len(self._last_y_axis) == n_rows:
            f = float(self._last_y_axis[ir])
            y_text = f"f={f:.2f} Hz"
        elif self._last_meta and "labels" in self._last_meta:
            labels = self._last_meta.get("labels", [])
            if isinstance(labels, list) and ir < len(labels):
                y_text = str(labels[ir])

        self.statusbar.showMessage(f"t={x:.4f}s | {y_text} | value={val:.5g}", 0)


    def _refresh_component_table(self):
        snap = self.engine.snapshot_components()
        self.tbl.setRowCount(len(snap))

        for r, c in enumerate(snap):
            self.tbl.setItem(r, 0, QTableWidgetItem(c["type"]))

            chk = QCheckBox()
            chk.setChecked(bool(c["enabled"]))
            chk.setProperty("comp_idx", r)
            chk.toggled.connect(self.on_component_toggled)
            self.tbl.setCellWidget(r, 1, chk)

            btn_edit = QPushButton("Правка")
            btn_edit.clicked.connect(lambda _, rr=r: self.on_edit_component(rr))
            self.tbl.setCellWidget(r, 2, btn_edit)

            btn_del = QPushButton("Удалить")
            btn_del.clicked.connect(lambda _, rr=r: self.on_delete_component(rr))
            self.tbl.setCellWidget(r, 3, btn_del)

            info = f"{c['name']} | " + ", ".join([f"{k}={v}" for k, v in list(c["params"].items())[:4]])
            self.tbl.setItem(r, 4, QTableWidgetItem(info))

    @Slot(bool)
    def on_component_toggled(self, checked: bool):
        chk = self.sender()
        if chk is None:
            return
        idx = chk.property("comp_idx")
        if idx is None:
            return
        self.engine.set_component_enabled(int(idx), bool(checked))
        # опционально: короткий статус, чтобы видеть факт обработки
        self.statusbar.showMessage(
            f"Компонент #{int(idx)}: {'вкл' if checked else 'выкл'}",
            800
        )

    @Slot()
    def on_add_component(self, comp_type: str):
        cls = BUILTIN_COMPONENTS[comp_type]
        # defaults
        initial = {s.key: s.default for s in cls.schema()}
        dlg = ComponentEditor(f"Добавить: {cls.NAME}", cls, initial, parent=self)
        if dlg.exec() == QDialog.Accepted:
            params = dlg.get_params()
            self.engine.add_component(comp_type, params=params, enabled=True)
            self._refresh_component_table()

    @Slot()
    def on_edit_component(self, row: int):
        snap = self.engine.snapshot_components()
        if not (0 <= row < len(snap)):
            return
        comp = snap[row]
        cls = BUILTIN_COMPONENTS[comp["type"]]
        dlg = ComponentEditor(f"Изменить: {cls.NAME}", cls, dict(comp["params"]), parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.engine.update_component_params(row, dlg.get_params())
            self._refresh_component_table()

    @Slot()
    def on_delete_component(self, row: int):
        self.engine.remove_component(row)
        self._refresh_component_table()

    # ---------- Presets / Export ----------

    def _preset_snapshot(self):
        return {
            "globals": {
                "fs": float(self.params.get("fs")),
                "chunk_size": int(self.params.get("chunk_size")),
                "view_window_sec": float(self.params.get("view_window_sec")),
                "scalogram_fps": int(self.params.get("scalogram_fps")),
            },
            "wavelet": {
                "plugin_id": str(self.params.get("wavelet_plugin_id")),
                "params": dict(self.params.get("wavelet_params", {})),
            },
            "components": self.engine.snapshot_components(),
        }

    def _apply_preset(self, data: dict):
        g = data.get("globals", {})
        self.spin_fs.setValue(float(g.get("fs", 2000.0)))
        self.spin_chunk.setValue(int(g.get("chunk_size", 256)))
        self.spin_window.setValue(float(g.get("view_window_sec", 4.0)))
        self.spin_sfps.setValue(int(g.get("scalogram_fps", 8)))

        w = data.get("wavelet", {})
        # На случай старых пресетов: удалить устаревшие ключи режима совместимости (scales)
        try:
            wp = w.get("params", {})
            if isinstance(wp, dict):
                wp.pop("use_legacy", None)
                wp.pop("scales_min", None)
                wp.pop("scales_max", None)
                wp.pop("n_scales", None)
        except Exception:
            pass
        pid = w.get("plugin_id", "builtin:cwt_morlet")
        # set combo index by plugin id
        for i in range(self.combo_wavelet.count()):
            if self.combo_wavelet.itemData(i) == pid:
                self.combo_wavelet.setCurrentIndex(i)
                break
        self.params.set("wavelet_params", dict(w.get("params", {})))
        # push into form if exists
        if hasattr(self, "wavelet_form"):
            # brute force: rebuild for correct schema
            info = self.pm.get_wavelet(pid)
            if info:
                schema = info.plugin_obj.get_parameters_schema()
                self._rebuild_wavelet_form(schema, dict(w.get("params", {})))

        # components (thread-safe replace)
        self.engine.replace_components(data.get("components", []))

        self._refresh_component_table()
        self.statusbar.showMessage("Пресет загружен", 2000)


    @Slot()
    def on_load_preset(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Загрузить пресет", os.path.join(self.project_root, "presets"), "JSON (*.json)")
        if not fn:
            return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_preset(data)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить пресет:\n{e}")

    @Slot()
    def on_save_preset(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить пресет", os.path.join(self.project_root, "presets"), "JSON (*.json)")
        if not fn:
            return
        try:
            data = self._preset_snapshot()
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.statusbar.showMessage(f"Пресет сохранён: {os.path.basename(fn)}", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить пресет:\n{e}")

    @Slot()
    def on_export_signal(self):
        fs = float(self.params.get("fs", 2000.0))
        window = float(self.params.get("view_window_sec", 4.0))
        n = int(max(16, window * fs))
        x = self.ring.get_last(n)

        fn, flt = QFileDialog.getSaveFileName(self, "Экспорт сигнала", self.project_root, "CSV (*.csv);;NPY (*.npy)")
        if not fn:
            return
        try:
            if fn.lower().endswith(".npy") or flt.startswith("NPY"):
                np.save(fn, x)
            else:
                t = np.arange(len(x), dtype=np.float32) / fs
                arr = np.column_stack([t, x])
                np.savetxt(fn, arr, delimiter=",", header="t_sec,x", comments="")
            self.statusbar.showMessage("Сигнал экспортирован", 2000)
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Экспорт не выполнен:\n{e}")

    @Slot()
    def on_export_png(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Экспорт скейлограммы (PNG)", self.project_root, "PNG (*.png)")
        if not fn:
            return
        try:
            pix = self.scalo_glw.grab()
            pix.save(fn, "PNG")
            self.statusbar.showMessage("Скейлограмма сохранена (PNG)", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Экспорт не выполнен:\n{e}")


