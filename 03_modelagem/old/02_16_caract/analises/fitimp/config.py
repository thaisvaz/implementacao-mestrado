from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(".")
PKLS_DIR = BASE_DIR / "pkls"
NOMES_COLUNAS_PATH = BASE_DIR / "nomescolunas.txt"

OUT_DIR = BASE_DIR / "out_mod"
OUT_DIR.mkdir(exist_ok=True)

# If not None, only these feature sets are processed (e.g. {"rtr","yty"})
ALLOWED_FEATURE_SETS: set[str] | None = None

LGBM_IMPORTANCE_TYPE = "gain"      # fallback to split
CUM_CUTOFF_PCT_TO_PLOT = 1.0       # cumulative cutoff (0..100)
MAX_FEATURES_IN_PLOT = 200         # readability guard
