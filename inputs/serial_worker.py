from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
import serial
from serial.tools import list_ports


class SerialWorker(QObject):
    """Reads samples from serial port and emits chunks compatible with MainWindow.on_chunk."""

    chunk_ready = Signal(object, float, float)  # x(np.ndarray), t0, fs
    status = Signal(str)

    def __init__(self, params_model: Any):
        super().__init__()
        self.params = params_model
        self._running = False
        self._ser: serial.Serial | None = None
        self._cfg_sig: tuple[Any, ...] | None = None
        self._raw = bytearray()
        self._sample_index = 0

    @staticmethod
    def available_ports() -> list[str]:
        return [p.device for p in list_ports.comports()]

    def _cfg_from_snapshot(self, snap: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(snap.get("serial_port", "")).strip(),
            int(snap.get("serial_baudrate", 310680)),
            int(snap.get("serial_bytesize", 8)),
            str(snap.get("serial_parity", "N")),
            float(snap.get("serial_stopbits", 1.0)),
            float(snap.get("serial_timeout", 0.01)),
        )

    def _reopen_if_needed(self, cfg: tuple[Any, ...]) -> bool:
        if cfg == self._cfg_sig and self._ser is not None and self._ser.is_open:
            return True

        self._close_serial()

        port, baudrate, bytesize, parity, stopbits, timeout = cfg
        if not port:
            self.status.emit("Serial: порт не выбран")
            self._cfg_sig = cfg
            return False

        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout,
            )
            self._cfg_sig = cfg
            self._raw.clear()
            self.status.emit(f"Serial: открыт {port} @ {baudrate}")
            return True
        except Exception as e:
            self._ser = None
            self._cfg_sig = cfg
            self.status.emit(f"Serial open error ({port}): {e}")
            return False

    def _close_serial(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
        self._ser = None

    @Slot()
    def start(self):
        self._running = True
        while self._running:
            snap = self.params.snapshot()
            if str(snap.get("input_source", "generator")) != "serial":
                time.sleep(0.05)
                continue

            if bool(snap.get("paused", True)):
                time.sleep(0.03)
                continue

            cfg = self._cfg_from_snapshot(snap)
            if not self._reopen_if_needed(cfg):
                time.sleep(0.2)
                continue

            fs = float(snap.get("fs", 2000.0))
            chunk_size = max(16, int(snap.get("chunk_size", 256)))
            fmt = str(snap.get("serial_format", "int16_le"))
            scale = float(snap.get("serial_scale", 1.0))

            bytes_per_sample = 2 if fmt == "int16_le" else 2
            read_size = max(64, chunk_size * bytes_per_sample)

            try:
                assert self._ser is not None
                waiting = self._ser.in_waiting
                raw = self._ser.read(waiting if waiting > 0 else read_size)
                if not raw:
                    continue

                self._raw.extend(raw)
                ready = (len(self._raw) // bytes_per_sample) * bytes_per_sample
                if ready < chunk_size * bytes_per_sample:
                    continue

                payload = memoryview(self._raw)[:ready]
                if fmt == "int16_le":
                    data_i16 = np.frombuffer(payload, dtype="<i2")
                    x = data_i16.astype(np.float32)
                else:
                    data_i16 = np.frombuffer(payload, dtype="<i2")
                    x = data_i16.astype(np.float32)

                if scale != 1.0:
                    x *= scale

                t0 = self._sample_index / fs
                self._sample_index += int(x.size)
                self.chunk_ready.emit(x, t0, fs)

                del self._raw[:ready]
            except Exception as e:
                self.status.emit(f"Serial read error: {e}")
                self._close_serial()
                time.sleep(0.2)

        self._close_serial()

    @Slot()
    def stop(self):
        self._running = False
