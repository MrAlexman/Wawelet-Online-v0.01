import time
import threading

class RealtimeClock:
    """Keeps logical time for chunk-based generation."""
    def __init__(self):
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._samples = 0

    def reset(self):
        with self._lock:
            self._t0 = time.perf_counter()
            self._samples = 0

    def advance(self, n_samples: int):
        with self._lock:
            self._samples += int(n_samples)

    def sample_index(self) -> int:
        with self._lock:
            return int(self._samples)

    def uptime_sec(self) -> float:
        with self._lock:
            return time.perf_counter() - self._t0
