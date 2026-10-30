"""`python -m workbench [file or session folder]`: start the Chatter Workbench.

Order matters: formats are registered and settings loaded before the window exists, the theme is applied before
the first paint, and settings are saved after the window closes (it stores its layout on close).
"""
import sys

from PySide6.QtWidgets import QApplication

from workbench.app.theme import apply_theme
from workbench.app.window import MainWindow
from workbench.core import registry, settings, steps


def main() -> int:
    if "--self-test" in sys.argv:
        from workbench.selftest import main as self_test
        return self_test()
    app = QApplication(sys.argv)
    app.setApplicationName("Chatter Workbench")
    problems = registry.load_formats() + steps.load_ops()
    s = settings.load()
    apply_theme(app, s["theme"])
    win = MainWindow(s, problems)
    win.show()
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if files:
        win.open_path(files[0])
    code = app.exec()
    try:
        settings.save(s)
    except OSError as e:
        print(f"settings not saved: {e}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
