#!/usr/bin/env bash
# Pack only what the Windows build needs into build/chatter-workbench-src-<version>.zip:
# the app code, the packaging files and the top-3 model package. No recordings, no lab data, no notebooks.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(env -u PYTHONPATH .venv-wb/bin/python -c "import workbench; print(workbench.__version__)")
OUT="build/chatter-workbench-src-${VERSION}.zip"
mkdir -p build && rm -f "$OUT"
env -u PYTHONPATH .venv-wb/bin/python - "$OUT" <<'PY'
import sys, zipfile
from pathlib import Path
out = sys.argv[1]
keep = [p for p in Path("workbench").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
keep += [p for p in Path("packaging").iterdir() if p.is_file()]
pkg = Path("models/chatter_model_v2_chatter3")
keep += [pkg / f for f in ("meta.json", "extra_trees.joblib", "random_forest.joblib", "svm_y.joblib",
                           "chatter_models.py", "features.py", "causal.py", "denoise.py")]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(keep):
        z.write(p, f"chatter-workbench/{p.as_posix()}")
print(f"{out}: {len(keep)} files")
PY
ls -lh "$OUT"
