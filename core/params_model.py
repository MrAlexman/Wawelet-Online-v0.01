import threading
from typing import Any, Dict
from PySide6.QtCore import QObject, Signal

class ParamsModel(QObject):
    changed = Signal(str)  # key changed

    def __init__(self, initial: Dict[str, Any] | None = None):
        super().__init__()
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = dict(initial or {})

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value
        self.changed.emit(key)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # shallow copy ok (we store primitives/list[str])
            return dict(self._data)
