#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-$HOME/.conda/envs/synpred-standalone}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

[ -n "$CONDA_BIN" ] || {
  echo "ERROR: conda was not found"
  exit 1
}

[ ! -e "$ENV_PREFIX" ] || {
  echo "ERROR: environment already exists: $ENV_PREFIX"
  exit 1
}

"$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.8.2 pip=23.0.1
PY="$ENV_PREFIX/bin/python"

"$PY" -m pip install setuptools==65.7.0 wheel==0.38.4
"$PY" -m pip install tensorflow==2.3.1 h5py==2.10.0 protobuf==3.20.3
"$PY" -m pip install --only-binary=:all:           numpy==1.20.3 scipy==1.6.3 pandas==1.2.5           scikit-learn==0.24.2 joblib==1.0.1 threadpoolctl==2.2.0           xgboost==1.4.2 rdkit-pypi==2021.9.3
"$PY" -m pip install networkx==2.5.1 mordred==1.2.0
"$PY" -m pip install --force-reinstall --no-deps           numpy==1.20.3 h5py==2.10.0 protobuf==3.20.3

PYTHONNOUSERSITE=1 "$PY" - <<'CHECK'
import h5py, mordred, numpy, pandas, rdkit, sklearn, tensorflow, xgboost
expected = {
    "numpy": "1.20.3",
    "pandas": "1.2.5",
    "sklearn": "0.24.2",
    "tensorflow": "2.3.1",
    "h5py": "2.10.0",
    "xgboost": "1.4.2",
    "rdkit": "2021.09.3",
    "mordred": "1.2.0",
}
actual = {
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "sklearn": sklearn.__version__,
    "tensorflow": tensorflow.__version__,
    "h5py": h5py.__version__,
    "xgboost": xgboost.__version__,
    "rdkit": rdkit.__version__,
    "mordred": mordred.__version__,
}
if actual != expected:
    raise RuntimeError((expected, actual))
print("SYNPRED_STANDALONE_ENV=PASS")
CHECK
