from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
OUT_MOD_DIR = Path("out_mod")
CSV_PATTERN = "*__importance.csv"

BOTTOM_CUM_PCT = 90.0  # bottom 10% => cum_pct_importance >= 90
OUTPUT_SUFFIX = "__intersection_less_10_percent.csv"


# -----------------------------
# Data structs
# -----------------------------
@dataclass(frozen=True)
class ImportanceCSV:
    characteristic: str
    model_key: str
    path: Path


# -----------------------------
# Helpers
# -----------------------------
def discover_importance_csvs(base_dir: Path) -> List[ImportanceCSV]:
    """
    Expected layout:
      out_mod/<characteristic>/<model_key>/*__importance.csv
    """
    out: List[ImportanceCSV] = []
    for csv_path in base_dir.glob(f"*/*/{CSV_PATTERN}"):
        # csv_path.parts: (..., out_mod, characteristic, model_key, filename)
        try:
            characteristic = csv_path.parents[1].name
            model_key = csv_path.parents[0].name
        except Exception:
            continue
        out.append(ImportanceCSV(characteristic=characteristic, model_key=model_key, path=csv_path))
    return out


def pick_latest_per_model(items: List[ImportanceCSV]) -> Dict[Tuple[str, str], ImportanceCSV]:
    """
    If multiple csvs exist for same (characteristic, model_key), pick the newest by mtime.
    """
    best: Dict[Tuple[str, str], ImportanceCSV] = {}
    for it in items:
        k = (it.characteristic, it.model_key)
        if k not in best:
            best[k] = it
        else:
            if it.path.stat().st_mtime > best[k].path.stat().st_mtime:
                best[k] = it
    return best


def is_problematic_all_zero(df: pd.DataFrame) -> bool:
    if "abs_importance" not in df.columns:
        return True
    s = pd.to_numeric(df["abs_importance"], errors="coerce").fillna(0.0)
    return bool((s == 0.0).all())


def bottom10_set_from_csv(path: Path, bottom_cum_pct: float = 90.0) -> Optional[Set[str]]:
    """
    Returns set of variables in bottom 10% by cumulative percent importance:
      cum_pct_importance >= 90.0

    Returns None if csv is invalid/problematic/all-zeros.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read: {path} | {e}")
        return None

    required = {"variable", "abs_importance", "cum_pct_importance"}
    if not required.issubset(df.columns):
        print(f"[WARN] Missing columns in {path.name}. Required: {sorted(required)}. Found: {list(df.columns)}")
        return None

    # ignore all-zero files (elasticnet case)
    if is_problematic_all_zero(df):
        print(f"[INFO] Ignoring ALL-ZERO importance file: {path}")
        return None

    cum = pd.to_numeric(df["cum_pct_importance"], errors="coerce")
    vars_ = df["variable"].astype(str)

    mask = cum >= float(bottom_cum_pct)
    s = set(vars_[mask].tolist())

    # If for some weird reason it's empty, still return empty set (valid but rare)
    return s


def compute_intersection_for_characteristic(
    latest_map: Dict[Tuple[str, str], ImportanceCSV],
    characteristic: str,
    bottom_cum_pct: float,
) -> Tuple[Optional[Set[str]], List[str], List[str]]:
    """
    Returns:
      - intersection set (or None if no valid models)
      - list of models used (valid)
      - list of models ignored
    """
    # collect models for this characteristic
    models = sorted({mk for (ch, mk) in latest_map.keys() if ch == characteristic})

    sets_by_model: Dict[str, Set[str]] = {}
    ignored: List[str] = []

    for mk in models:
        item = latest_map[(characteristic, mk)]
        s = bottom10_set_from_csv(item.path, bottom_cum_pct=bottom_cum_pct)
        if s is None:
            ignored.append(mk)
            continue
        sets_by_model[mk] = s

    used_models = sorted(sets_by_model.keys())
    if not used_models:
        return None, [], ignored

    # intersection across used models
    inter = None
    for mk in used_models:
        if inter is None:
            inter = set(sets_by_model[mk])
        else:
            inter &= sets_by_model[mk]

    return inter if inter is not None else set(), used_models, ignored


def write_intersection_csv(
    out_dir: Path,
    characteristic: str,
    inter: Set[str],
    used_models: List[str],
    ignored_models: List[str],
) -> Path:
    out_path = out_dir / characteristic / f"{characteristic}{OUTPUT_SUFFIX}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"variable": sorted(inter)})
    df.insert(0, "characteristic", characteristic)
    df.insert(1, "n_models_used", len(used_models))
    df.insert(2, "models_used", " | ".join(used_models))
    df.insert(3, "models_ignored", " | ".join(ignored_models) if ignored_models else "")

    df.to_csv(out_path, index=False)
    return out_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not OUT_MOD_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {OUT_MOD_DIR.resolve()}")

    items = discover_importance_csvs(OUT_MOD_DIR)
    if not items:
        raise RuntimeError(f"No importance CSVs found under: {OUT_MOD_DIR.resolve()}")

    latest_map = pick_latest_per_model(items)

    characteristics = sorted({ch for (ch, _mk) in latest_map.keys()})

    print(f"[INFO] Found {len(characteristics)} characteristics.")
    print(f"[INFO] Using bottom-10% cutoff: cum_pct_importance >= {BOTTOM_CUM_PCT}")

    for ch in characteristics:
        inter, used, ignored = compute_intersection_for_characteristic(
            latest_map=latest_map,
            characteristic=ch,
            bottom_cum_pct=BOTTOM_CUM_PCT,
        )

        if inter is None:
            print(f"[WARN] {ch}: no valid model CSVs (all ignored or invalid). Skipping output.")
            continue

        out_path = write_intersection_csv(
            out_dir=OUT_MOD_DIR,
            characteristic=ch,
            inter=inter,
            used_models=used,
            ignored_models=ignored,
        )

        print(
            f"[OK] {ch}: intersection size={len(inter)} | models_used={used} | ignored={ignored} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
