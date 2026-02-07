from typing import Dict, Any

from PySide6.QtCore import Qt, QEvent, QObject
from PySide6.QtWidgets import (
    QWidget, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QComboBox,
    QAbstractSpinBox, QLabel
)

from core.schema import ParamSpec


class _NoWheelUnlessFocused(QObject):
    """Ignore mouse-wheel unless the widget has focus (prevents accidental changes)."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if hasattr(obj, "hasFocus") and not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)


class ParamForm(QWidget):
    """Auto-generated editor from Schema."""
    def __init__(self, schema: list[ParamSpec], initial: Dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.schema = schema
        self.widgets: Dict[str, Any] = {}
        self.layout = QFormLayout(self)
        initial = initial or {}
        self._no_wheel_filter = _NoWheelUnlessFocused(self)        

        for s in schema:
            w = self._make_widget(s, initial.get(s.key, s.default))
            self.widgets[s.key] = (s, w)

            # tooltip from schema (neutral style)
            w.setToolTip(self._build_tooltip(s))

            lbl = QLabel(s.label)
            lbl.setToolTip(self._build_tooltip(s))
            self.layout.addRow(lbl, w)

    def _build_tooltip(self, spec: ParamSpec) -> str:
        """
        Tooltip in neutral style (no 'you', no imperative).
        Uses spec.description/spec.examples if present.
        """
        title = f"<b>{spec.label}</b>"

        desc = getattr(spec, "description", None) or ""
        examples = getattr(spec, "examples", None) or []

        html = title
        if desc:
            html += f"<br>{desc}"

        if examples:
            items = "".join([f"<li><code>{e}</code></li>" for e in examples])
            html += f"<br><b>Примеры:</b><ul>{items}</ul>"

        # small technical footer (optional)
        # html += f"<br><span style='color:gray'>{spec.key}</span>"
        # Техническая строка: тип/диапазон (коротко)
        tech_parts = []
        tech_parts.append(f"тип: {spec.type}")

        if spec.type in ("float", "int"):
            if spec.min is not None or spec.max is not None:
                tech_parts.append(f"диапазон: {spec.min if spec.min is not None else '−∞'}…{spec.max if spec.max is not None else '+∞'}")
            if spec.step is not None:
                tech_parts.append(f"шаг: {spec.step}")

        if spec.type == "enum" and spec.choices:
            tech_parts.append(f"вариантов: {len(spec.choices)}")

        html += "<br><span style='color:gray'>" + " · ".join(tech_parts) + "</span>"
        return html

    def _apply_no_wheel(self, w: QWidget) -> None:
        w.installEventFilter(self._no_wheel_filter)
        w.setFocusPolicy(Qt.StrongFocus)
        if isinstance(w, QAbstractSpinBox):
            w.setKeyboardTracking(False)


    def _make_widget(self, spec: ParamSpec, value):
        t = spec.type
        if t == "float":
            w = QDoubleSpinBox()
            w.setDecimals(6)
            if spec.min is not None: w.setMinimum(float(spec.min))
            if spec.max is not None: w.setMaximum(float(spec.max))
            if spec.step is not None: w.setSingleStep(float(spec.step))
            w.setValue(float(value))
            self._apply_no_wheel(w)
            return w

        if t == "int":
            w = QSpinBox()
            if spec.min is not None: w.setMinimum(int(spec.min))
            if spec.max is not None: w.setMaximum(int(spec.max))
            if spec.step is not None: w.setSingleStep(int(spec.step))
            w.setValue(int(value))
            self._apply_no_wheel(w)
            return w

        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            self._apply_no_wheel(w)
            return w

            
        if t == "enum":
            w = QComboBox()
            choices = spec.choices or []
            w.addItems(choices)
            if value in choices:
                w.setCurrentText(str(value))
            self._apply_no_wheel(w)
            return w
            
        if t == "str":
            w = QLineEdit()
            w.setText(str(value))
            self._apply_no_wheel(w)
            return w



        # fallback
        w = QLineEdit()
        w.setText(str(value))
        self._apply_no_wheel(w)
        return w


    def values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, (s, w) in self.widgets.items():
            if s.type == "float":
                out[k] = float(w.value())
            elif s.type == "int":
                out[k] = int(w.value())
            elif s.type == "bool":
                out[k] = bool(w.isChecked())
            elif s.type == "enum":
                out[k] = str(w.currentText())
            elif s.type == "str":
                out[k] = str(w.text())
            else:
                out[k] = str(w.text())
        return out
