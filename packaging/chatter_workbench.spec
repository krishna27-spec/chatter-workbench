# PyInstaller recipe for Chatter Workbench, the same on Linux and Windows.
#   Linux:   packaging/build_linux.sh      Windows: packaging\build_windows.bat
# One folder (not one file): starts fast and the bundled models stay as plain files next to the program.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
PKG = ROOT / "models" / "chatter_model_v2_chatter3"          # the deployed top-3 package, bundled
WIN = sys.platform == "win32"
NAME = "ChatterWorkbench" if WIN else "chatter-workbench"

hidden = [
    # plug-ins found at run time by pkgutil, so PyInstaller can't see them being imported
    "workbench.core.formats.lvm", "workbench.core.formats.csv_fmt", "workbench.core.formats.session",
    "workbench.core.ops.clean", "workbench.selftest",
    # the model package's own code (the SVM pickle needs chatter_models by that name)
    "chatter_models", "features", "causal", "denoise",
    # what the pickled models are made of
    "sklearn.ensemble._forest", "sklearn.tree._classes", "sklearn.pipeline", "sklearn.preprocessing._data",
    "sklearn.svm", "joblib", "pywt", "pycatch22", "scipy.special", "scipy.ndimage", "scipy.signal",
    "nidaqmx", "nidaqmx.system", "nidaqmx.stream_readers",
] + collect_submodules("sklearn.tree") + collect_submodules("sklearn.ensemble")
ni_datas, ni_binaries, ni_hidden = collect_all("nidaqmx")      # its data files and every submodule
hidden += ni_hidden

datas = [(str(PKG / f), "models/chatter_model_v2_chatter3")
         for f in ("meta.json", "extra_trees.joblib", "random_forest.joblib", "svm_y.joblib")]
datas += [(str(ROOT / "workbench" / "app" / "fonts"), "workbench/app/fonts")]
datas += ni_datas

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT), str(PKG)],
    binaries=ni_binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "torch", "IPython", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtQml",
              "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=NAME,
    console=not WIN,                       # Windows: no black console window behind the app
    icon=str(ROOT / "packaging" / ("icon.ico" if WIN else "icon.png")),
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
