from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------
# Config (edit these)
# -----------------------------
BASE_DIR = Path(".")  # folder that contains "pkls/" and "nomescolunas.txt"
PKL_PATH = BASE_DIR / "pkls" / "rtr__lightgbm_21012026_model.pkl"
NOMES_COLUNAS_PATH = BASE_DIR / "nomescolunas.txt"

OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

IMPORTANCE_TYPE_PRIMARY = "gain"  # "gain" is usually best; fallback to "split"
TOP_PERCENT_TO_PLOT = 1.0         # plot top 5% most important features


# -----------------------------
# Utilities
# -----------------------------
def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_feature_names_from_nomescolunas(path: Path) -> List[str]:
    """
    nomescolunas.txt is a single huge line:
      col1,col2,col3,...
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    cols = [c.strip() for c in text.split(",") if c.strip()]
    if not cols:
        raise ValueError(f"No columns parsed from {path}")
    return cols


def try_get_model_feature_names(model: Any) -> Optional[List[str]]:
    """
    Best-effort feature names from LightGBM objects.
    """
    for attr in ["feature_name_", "feature_names_"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (list, tuple)):
                return list(val)

    if hasattr(model, "feature_name") and callable(getattr(model, "feature_name")):
        try:
            names = model.feature_name()
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass

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


def get_lgbm_booster(model: Any) -> Any:
    """
    Return a LightGBM Booster-like object that supports:
      - feature_importance(importance_type=...)
      - feature_name()
    """
    if hasattr(model, "feature_importance") and hasattr(model, "feature_name"):
        return model

    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_")
        if hasattr(booster, "feature_importance") and hasattr(booster, "feature_name"):
            return booster

    raise TypeError(
        "Loaded pickle is not a LightGBM Booster or an sklearn LGBM model with booster_. "
        f"Got type={type(model)}"
    )


def align_feature_names(feature_names: List[str], importances_len: int) -> List[str]:
    """
    Ensure feature_names length matches importance vector length.
    If mismatch, we trim to min length and warn via print.
    """
    if len(feature_names) == importances_len:
        return feature_names

    m = min(len(feature_names), importances_len)
    print(
        f"[WARN] Feature name count ({len(feature_names)}) != importance length ({importances_len}). "
        f"Trimming both to {m}."
    )
    return feature_names[:m]


def build_importance_df(
    booster: Any,
    feature_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Returns a DF sorted by abs importance desc with:
      - variable
      - abs_importance
      - pct_importance
      - cum_pct_importance
    """
    # Try primary importance type; fallback to split if needed.
    try:
        imp = booster.feature_importance(importance_type=importance_type)
        used_type = importance_type
    except Exception:
        imp = booster.feature_importance(importance_type="split")
        used_type = "split"
        print(f"[WARN] importance_type='{importance_type}' failed. Using 'split' instead.")

    # LightGBM returns numpy array; keep it that way and handle safely
    importances = list(imp)
    feature_names = align_feature_names(feature_names, len(importances))
    importances = importances[: len(feature_names)]

    # ✅ FIX: ensure we have a pandas Series before fillna()
    imp_s = pd.Series(pd.to_numeric(importances, errors="coerce")).fillna(0.0)

    df = pd.DataFrame(
        {
            "variable": feature_names[: len(imp_s)],
            "abs_importance": imp_s.to_numpy(),
        }
    )

    df = df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    total = float(df["abs_importance"].sum())
    if total <= 0.0:
        df["pct_importance"] = 0.0
        df["cum_pct_importance"] = 0.0
    else:
        df["pct_importance"] = (df["abs_importance"] / total) * 100.0
        df["cum_pct_importance"] = df["pct_importance"].cumsum()

    df.attrs["importance_type_used"] = used_type
    df.attrs["total_importance"] = total
    return df


def save_debug_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """
    Minimal YAML writer without extra deps.
    """
    lines: List[str] = []
    for k, v in payload.items():
        lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_top_percent(df: pd.DataFrame, out_png: Path, top_percent: float = 1.0) -> None:
    """
    Plot only top X% features by abs_importance.
    Uses matplotlib.
    """
    import matplotlib.pyplot as plt

    n = len(df)
    k = max(1, int(math.ceil(n * (top_percent / 100.0))))

    top = df.head(k).copy()
    top = top.iloc[::-1]  # biggest on top in barh

    ax = top.plot(
        kind="barh",
        x="variable",
        y="abs_importance",
        legend=False,
        figsize=(10, max(4, int(0.25 * k))),
    )
    ax.set_title(f"Top {top_percent:.1f}% feature importance (abs) — LightGBM")
    ax.set_xlabel("abs_importance")
    ax.set_ylabel("variable")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Plot saved to: {out_png.resolve()}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print(f"[INFO] Loading model pickle: {PKL_PATH}")
    model = load_pickle(PKL_PATH)

    booster = get_lgbm_booster(model)

    # Feature names priority:
    # 1) model/booster built-in names
    # 2) nomescolunas.txt
    model_names = try_get_model_feature_names(model)
    if model_names:
        feature_names = model_names
        source = "model"
    else:
        feature_names = load_feature_names_from_nomescolunas(NOMES_COLUNAS_PATH)
        source = "nomescolunas.txt"

    df = build_importance_df(
        booster=booster,
        feature_names=feature_names,
        importance_type=IMPORTANCE_TYPE_PRIMARY,
    )

    used_type = df.attrs.get("importance_type_used", "unknown")
    total = df.attrs.get("total_importance", "unknown")

    out_csv = OUT_DIR / "rtr_lightgbm_feature_importance.csv"
    out_png = OUT_DIR / "rtr_lightgbm_feature_importance_top5pct.png"
    out_yaml = OUT_DIR / "rtr_lightgbm_debug.yaml"

    df_out = df[["variable", "abs_importance", "pct_importance", "cum_pct_importance"]].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved to: {out_csv.resolve()}")
    print(f"[INFO] importance_type used: {used_type} | total_importance: {total} | names_source: {source}")

    plot_top_percent(df, out_png, top_percent=TOP_PERCENT_TO_PLOT)

    save_debug_yaml(
        out_yaml,
        {
            "pickle": PKL_PATH.name,
            "model_type": type(model).__name__,
            "names_source": source,
            "n_features_in_names": len(feature_names),
            "n_rows_in_df": len(df),
            "importance_type_requested": IMPORTANCE_TYPE_PRIMARY,
            "importance_type_used": used_type,
            "top_percent_plotted": TOP_PERCENT_TO_PLOT,
        },
    )
    print(f"[OK] Debug YAML saved to: {out_yaml.resolve()}")


if __name__ == "__main__":
    main()
