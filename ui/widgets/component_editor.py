from __future__ import annotations
from typing import Dict, Any, Type
from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QLabel, QPushButton
from signal.components.base import SignalComponent
from ui.widgets.param_form import ParamForm

class ComponentEditor(QDialog):
    def __init__(self, title: str, comp_cls: Type[SignalComponent], initial_params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        schema = comp_cls.schema()

        layout = QVBoxLayout(self)
        lbl = QLabel(comp_cls.NAME)
        # нейтральная подсказка, если компонент имеет DESCRIPTION
        desc = getattr(comp_cls, "DESCRIPTION", "")
        if desc:
            lbl.setToolTip(f"<b>{comp_cls.NAME}</b><br>{desc}")
        layout.addWidget(lbl)
        self.form = ParamForm(schema, initial=initial_params, parent=self)
        layout.addWidget(self.form)

        bb = QDialogButtonBox()
        btn_ok = QPushButton("ОК")
        btn_cancel = QPushButton("Отмена")

        bb.addButton(btn_ok, QDialogButtonBox.AcceptRole)
        bb.addButton(btn_cancel, QDialogButtonBox.RejectRole)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        layout.addWidget(bb)

    def get_params(self) -> Dict[str, Any]:
        return self.form.values()
