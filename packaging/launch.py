"""Entry point for the installed app (PyInstaller starts here): the same as `python -m workbench`."""
import sys

from workbench.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
