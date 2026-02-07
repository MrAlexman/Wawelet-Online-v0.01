import numpy as np
import threading

class RingBuffer:
    """Thread-safe ring buffer for last N samples."""
    def __init__(self, capacity: int, dtype=np.float32):
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=dtype)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._buf[:] = 0
            self._write = 0
            self._filled = 0

    def append(self, x: np.ndarray):
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                self._buf[:] = x[-self.capacity:]
                self._write = 0
                self._filled = self.capacity
                return

            end = self._write + n
            if end <= self.capacity:
                self._buf[self._write:end] = x
            else:
                k = self.capacity - self._write
                self._buf[self._write:] = x[:k]
                self._buf[:end - self.capacity] = x[k:]
            self._write = end % self.capacity
            self._filled = min(self.capacity, self._filled + n)

    def get_last(self, n: int) -> np.ndarray:
        n = int(n)
        with self._lock:
            n = min(n, self._filled)
            if n <= 0:
                return np.zeros(0, dtype=self._buf.dtype)
            start = (self._write - n) % self.capacity
            if start < self._write:
                return self._buf[start:self._write].copy()
            return np.concatenate([self._buf[start:].copy(), self._buf[:self._write].copy()])
