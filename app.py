import sys
from PySide6.QtWidgets import QApplication
from core.logger import setup_logging
from ui.main_window import MainWindow

def run_app():
    setup_logging()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
