"""
TextureLab Desktop – Entry Point

Standalone pavement texture analysis application.
"""
import sys
import os

# Add parent directory to path so engine/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


def main():
    # High-DPI support
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("TextureLab Desktop")
    app.setOrganizationName("TextureLab")
    app.setApplicationVersion("2.0.0")

    # Apply dark theme
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except ImportError:
        # Fallback dark palette
        from PyQt6.QtGui import QPalette, QColor
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 40))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 45))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 55))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(99, 110, 250))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
