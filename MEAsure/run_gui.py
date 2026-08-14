from __future__ import annotations

import sys
from pathlib import Path

# Spyder may keep a different working directory from the script's folder.
# Resolve sibling imports from this file instead of relying on prior scripts.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app import MEAApp
except ModuleNotFoundError as exc:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "MEA Analysis Workbench — missing dependency",
            f"Python could not import '{exc.name}'.\n\n"
            "Run install_and_run.bat, or install the packages from requirements.txt "
            "in the same Python environment used by Spyder.",
        )
        root.destroy()
    finally:
        raise


if __name__ == "__main__":
    MEAApp().mainloop()
