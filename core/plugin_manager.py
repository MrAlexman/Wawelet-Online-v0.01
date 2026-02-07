import importlib
import logging
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Optional, Any

log = logging.getLogger("PluginManager")

@dataclass
class PluginInfo:
    id: str
    meta: dict
    module: ModuleType
    plugin_obj: Any

class PluginManager:
    """
    Loads wavelet plugins from:
      - wavelets.builtins.*
      - plugins/wavelets/*.py
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.wavelet_plugins: Dict[str, PluginInfo] = {}

        # Ensure project root is on sys.path
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

    def reload_all(self):
        self.wavelet_plugins.clear()
        self._load_builtin_wavelets()
        self._load_external_wavelets()

    def _load_builtin_wavelets(self):
        builtin_mods = [
            ("wavelets.builtins.cwt_morlet", "builtin:cwt_morlet"),
            ("wavelets.builtins.dwt_wpt", "builtin:dwt_wpt"),
        ]
        for modname, pid in builtin_mods:
            self._load_wavelet_module(modname, plugin_id=pid)


    def _load_external_wavelets(self):
        folder = os.path.join(self.project_root, "plugins", "wavelets")
        if not os.path.isdir(folder):
            return
        for fn in os.listdir(folder):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            modname = f"plugins.wavelets.{fn[:-3]}"
            self._load_wavelet_module(modname, plugin_id=f"plugin:{modname}")

    def _load_wavelet_module(self, modname: str, plugin_id: str):
        try:
            if modname in sys.modules:
                mod = importlib.reload(sys.modules[modname])
            else:
                mod = importlib.import_module(modname)

            plugin_cls = getattr(mod, "WaveletPlugin", None)
            meta = getattr(mod, "PLUGIN_META", None)
            if plugin_cls is None or meta is None:
                raise ValueError("Plugin must define PLUGIN_META and WaveletPlugin class")

            obj = plugin_cls()
            pid = meta.get("id", plugin_id)
            # alias: allow access by fallback plugin_id too
            if pid != plugin_id:
                self.wavelet_plugins[plugin_id] = PluginInfo(id=pid, meta=meta, module=mod, plugin_obj=obj)
            
            self.wavelet_plugins[pid] = PluginInfo(id=pid, meta=meta, module=mod, plugin_obj=obj)
            log.info("Loaded wavelet plugin: %s (%s)", meta.get("name"), pid)
        except Exception as e:
            log.exception("Failed to load wavelet plugin %s: %s", modname, e)

    def list_wavelets(self) -> List[PluginInfo]:
        return list(self.wavelet_plugins.values())

    def get_wavelet(self, plugin_id: str) -> Optional[PluginInfo]:
        return self.wavelet_plugins.get(plugin_id)
