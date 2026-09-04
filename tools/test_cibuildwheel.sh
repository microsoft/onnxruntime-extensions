#!/bin/bash
set -euo pipefail

python - <<'PY'
import onnxruntime_extensions as _ortx  # noqa: F401
import onnxruntime_extensions._extensions_pydll as _ext
print(_ext.__file__)
PY

if python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"; then
  python -m pip install -q abi3audit
  ext_path="$(python -c "import onnxruntime_extensions._extensions_pydll as m; print(m.__file__)")"
  abi3audit --assume-minimum-abi3 3.12 "$ext_path"
fi
