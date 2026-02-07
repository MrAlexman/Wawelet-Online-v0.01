from __future__ import annotations
import numpy as np
import threading
from typing import List, Dict, Any
from core.realtime_clock import RealtimeClock
from signal.components.base import SignalComponent, ComponentState
from signal.components import BUILTIN_COMPONENTS

# Max chunk size (must match UI spin_chunk max). Pre-allocated buffer for zero allocs per chunk.
MAX_CHUNK = 131072

class SignalEngine:
    def __init__(self, clock: RealtimeClock):
        self.clock = clock
        self._lock = threading.Lock()
        self.components: List[SignalComponent] = []

        # global generator params (defaults)
        self.fs = 2000.0
        self.chunk_size = 256
        self.amplitude_clip = 0.0  # 0 => off

        self._paused = True
        self._chunk_buf = np.zeros(MAX_CHUNK, dtype=np.float32)

    def set_global(self, fs: float | None = None, chunk_size: int | None = None, amplitude_clip: float | None = None):
        with self._lock:
            if fs is not None:
                self.fs = float(fs)
            if chunk_size is not None:
                self.chunk_size = int(chunk_size)
            if amplitude_clip is not None:
                self.amplitude_clip = float(amplitude_clip)

    def get_global(self) -> tuple[float, int]:
        """Thread-safe: (fs, chunk_size). Use from worker to avoid races."""
        with self._lock:
            return (float(self.fs), int(self.chunk_size))

    def play(self): 
        with self._lock:
            self._paused = False

    def pause(self):
        with self._lock:
            self._paused = True

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def reset(self):
        with self._lock:
            self.clock.reset()

    def add_component(self, comp_type: str, params: Dict[str, Any] | None = None, enabled: bool = True) -> int:
        cls = BUILTIN_COMPONENTS[comp_type]
        state = ComponentState(enabled=enabled, params={})
        # defaults
        for s in cls.schema():
            state.params[s.key] = s.default
        if params:
            state.params.update(params)
        obj = cls(state)
        with self._lock:
            self.components.append(obj)
            return len(self.components) - 1

    def remove_component(self, idx: int):
        with self._lock:
            if 0 <= idx < len(self.components):
                self.components.pop(idx)

    def set_component_enabled(self, idx: int, enabled: bool):
        with self._lock:
            if 0 <= idx < len(self.components):
                self.components[idx].state.enabled = bool(enabled)

    def update_component_params(self, idx: int, params: Dict[str, Any]):
        with self._lock:
            if 0 <= idx < len(self.components):
                self.components[idx].update_params(params)

    def snapshot_components(self):
        with self._lock:
            snap = []
            for c in self.components:
                snap.append({
                    "type": c.TYPE,
                    "name": c.NAME,
                    "enabled": c.state.enabled,
                    "params": dict(c.state.params),
                })
            return snap

    def replace_components(self, components_snap: List[Dict[str, Any]]):
        """Thread-safe: clear and rebuild from preset-style list."""
        with self._lock:
            self.components.clear()
            for c in components_snap:
                comp_type = c.get("type")
                if comp_type not in BUILTIN_COMPONENTS:
                    continue
                cls = BUILTIN_COMPONENTS[comp_type]
                state = ComponentState(
                    enabled=bool(c.get("enabled", True)),
                    params=dict(c.get("params", {})),
                )
                for s in cls.schema():
                    if s.key not in state.params:
                        state.params[s.key] = s.default
                self.components.append(cls(state))

    def generate_chunk(self) -> tuple[np.ndarray, float]:
        """Return (chunk, t0). If paused -> zeros."""
        with self._lock:
            fs = float(self.fs)
            n = int(self.chunk_size)
            paused = self._paused
            comps = list(self.components)
            clip = float(self.amplitude_clip)

        t0 = self.clock.sample_index() / fs

        if paused:
            return np.zeros(0, dtype=np.float32), t0
        n = min(n, MAX_CHUNK)
        x = self._chunk_buf[:n]
        x.fill(0.0)
        for c in comps:
            x += c.generate(t0=t0, n=n, fs=fs)

        if clip and clip > 0:
            np.clip(x, -clip, clip, out=x)

        self.clock.advance(n)
        return x.copy(), t0
