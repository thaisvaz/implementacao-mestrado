from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional


def load_feature_names_from_nomescolunas(path: Path) -> List[str]:
    """
    nomescolunas.txt: single huge comma-separated line.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    cols = [c.strip() for c in text.split(",") if c.strip()]
    if not cols:
        raise ValueError(f"No columns parsed from {path}")
    return cols


def try_get_model_feature_names(model: Any) -> Optional[List[str]]:
    # sklearn: feature_names_in_
    if hasattr(model, "feature_names_in_"):
        v = getattr(model, "feature_names_in_")
        try:
            return list(v)
        except Exception:
            pass

    # LightGBM wrapper
    for attr in ["feature_name_", "feature_names_"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (list, tuple)):
                return list(val)

    # Booster has feature_name()
    if hasattr(model, "feature_name") and callable(getattr(model, "feature_name")):
        try:
            names = model.feature_name()
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass

    # Wrapper exposes booster_.feature_name()
    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_")
        if hasattr(booster, "feature_name") and callable(getattr(booster, "feature_name")):
            try:
                names = booster.feature_name()
                if isinstance(names, list) and names:
                    return names
            except Exception:
                pass

    return None


def align_feature_names(feature_names: List[str], n: int) -> List[str]:
    if len(feature_names) == n:
        return feature_names
    m = min(len(feature_names), n)
    print(f"[WARN] feature_names ({len(feature_names)}) != values ({n}). Trimming to {m}.")
    return feature_names[:m]
