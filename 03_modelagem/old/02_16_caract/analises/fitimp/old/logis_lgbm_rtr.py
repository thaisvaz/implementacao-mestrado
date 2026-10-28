from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(".")
PKLS_DIR = BASE_DIR / "pkls"
NOMES_COLUNAS_PATH = BASE_DIR / "nomescolunas.txt"

OUT_DIR = BASE_DIR / "out2"
OUT_DIR.mkdir(exist_ok=True)

ONLY_PREFIX = "rtr__"                 # for now: only rtr__
CUM_CUTOFF_PCT_TO_PLOT = 1.0          # plot until cumulative reaches 1.0%
LGBM_IMPORTANCE_TYPE_PRIMARY = "gain" # fallback to split


# -----------------------------
# IO
# -----------------------------
def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_feature_names_from_nomescolunas(path: Path) -> List[str]:
    """
    nomescolunas.txt is a single huge comma-separated line.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    cols = [c.strip() for c in text.split(",") if c.strip()]
    if not cols:
        raise ValueError(f"No columns parsed from {path}")
    return cols


def find_model_pkls(pkls_dir: Path, prefix: str) -> List[Path]:
    if not pkls_dir.exists():
        raise FileNotFoundError(f"pkls dir not found: {pkls_dir}")
    return sorted(pkls_dir.glob(f"{prefix}*_model.pkl"))


def model_basename(path: Path) -> str:
    # e.g. rtr__lightgbm_21012026_model.pkl -> rtr__lightgbm_21012026
    return re.sub(r"_model\.pkl$", "", path.name)


# -----------------------------
# Feature name extraction (best effort)
# -----------------------------
def try_get_model_feature_names(model: Any) -> Optional[List[str]]:
    # sklearn: sometimes has feature_names_in_
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


# -----------------------------
# Model detection
# -----------------------------
def is_lightgbm_model(model: Any) -> bool:
    if hasattr(model, "feature_importance") and hasattr(model, "feature_name"):
        return True
    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_")
        if hasattr(booster, "feature_importance") and hasattr(booster, "feature_name"):
            return True
    return False


def is_logreg_model(model: Any) -> bool:
    return hasattr(model, "coef_")


def get_lgbm_booster(model: Any) -> Any:
    if hasattr(model, "feature_importance") and hasattr(model, "feature_name"):
        return model
    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_")
        if hasattr(booster, "feature_importance") and hasattr(booster, "feature_name"):
            return booster
    raise TypeError(f"Not a LightGBM model/booster: {type(model)}")


# -----------------------------
# Importance builders
# -----------------------------
def importance_table_from_values(feature_names: List[str], values: List[float]) -> pd.DataFrame:
    """
    Build standardized output:
      variable, abs_importance, pct_importance, cum_pct_importance
    """
    feature_names = align_feature_names(feature_names, len(values))
    values = values[: len(feature_names)]

    s = pd.Series(pd.to_numeric(values, errors="coerce")).fillna(0.0).astype(float)

    df = pd.DataFrame({"variable": feature_names[: len(s)], "abs_importance": s.to_numpy()})
    df = df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    total = float(df["abs_importance"].sum())
    if total <= 0:
        df["pct_importance"] = 0.0
        df["cum_pct_importance"] = 0.0
    else:
        df["pct_importance"] = (df["abs_importance"] / total) * 100.0
        df["cum_pct_importance"] = df["pct_importance"].cumsum()

    df.attrs["total_importance"] = total
    return df


def build_importance_lightgbm(booster: Any, feature_names: List[str], primary_type: str) -> Tuple[pd.DataFrame, str]:
    try:
        imp = booster.feature_importance(importance_type=primary_type)
        used_type = primary_type
    except Exception:
        imp = booster.feature_importance(importance_type="split")
        used_type = "split"
        print(f"[WARN] LGBM importance_type '{primary_type}' failed, using 'split'.")

    df = importance_table_from_values(feature_names, list(imp))
    df.attrs["importance_type_used"] = used_type
    return df, used_type


def build_importance_logreg(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Logistic Regression importance = abs(coef).
    For multiclass: mean abs across classes.
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise TypeError("Model missing coef_")

    coef_df = pd.DataFrame(coef).abs()
    if coef_df.shape[0] == 1:
        values = coef_df.iloc[0].to_list()
        n_classes = 1
    else:
        values = coef_df.mean(axis=0).to_list()
        n_classes = coef_df.shape[0]

    df = importance_table_from_values(feature_names, values)
    df.attrs["importance_type_used"] = "abs(coef)"
    df.attrs["n_classes_in_coef"] = n_classes
    return df


# -----------------------------
# Debug YAML
# -----------------------------
def save_debug_yaml(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    for k, v in payload.items():
        lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Plot (NEW: cutoff by cumulative percent)
# -----------------------------
def plot_top_cum_percent(
    df: pd.DataFrame,
    out_png: Path,
    cutoff_pct: float,
    title: str,
    max_features: int | None = 200,  # safety to keep plots readable
) -> None:
    """
    Plot features until cum_pct_importance reaches cutoff_pct.
    Expects df sorted by abs_importance desc and containing cum_pct_importance (0..100).
    """
    import matplotlib.pyplot as plt

    if "cum_pct_importance" not in df.columns:
        raise ValueError("df must contain 'cum_pct_importance'")
    if df.empty:
        raise ValueError("df is empty")

    mask = df["cum_pct_importance"] <= cutoff_pct
    top = df.loc[mask].copy()

    # Ensure at least 1 feature, and include the crossing feature
    if top.empty:
        top = df.head(1).copy()
    else:
        if len(top) < len(df):
            top = pd.concat([top, df.iloc[[len(top)]]], ignore_index=True)

    if max_features is not None and len(top) > max_features:
        top = top.head(max_features).copy()

    top = top.iloc[::-1]  # biggest on top in barh

    ax = top.plot(
        kind="barh",
        x="variable",
        y="abs_importance",
        legend=False,
        figsize=(10, max(4, int(0.25 * len(top)))),
    )
    ax.set_title(title)
    ax.set_xlabel("abs_importance")
    ax.set_ylabel("variable")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(
        f"[OK] Plot saved: {out_png.resolve()} "
        f"| n_features_plotted={len(top)} | cum_cutoff={cutoff_pct}%"
    )


# -----------------------------
# Per-file driver
# -----------------------------
def process_one_model(pkl_path: Path, all_feature_names: List[str]) -> None:
    name = model_basename(pkl_path)
    print(f"\n[INFO] Loading: {pkl_path}")

    model = load_pickle(pkl_path)

    model_names = try_get_model_feature_names(model)
    if model_names:
        feature_names = model_names
        names_source = "model"
    else:
        feature_names = all_feature_names
        names_source = "nomescolunas.txt"

    if is_lightgbm_model(model):
        booster = get_lgbm_booster(model)
        df, used_type = build_importance_lightgbm(booster, feature_names, LGBM_IMPORTANCE_TYPE_PRIMARY)
        model_kind = "lightgbm"
        importance_used = used_type
        plot_title = f"{name} — LightGBM ({importance_used}) — cum {CUM_CUTOFF_PCT_TO_PLOT:.1f}%"
    elif is_logreg_model(model):
        df = build_importance_logreg(model, feature_names)
        model_kind = "logistic_regression"
        importance_used = df.attrs.get("importance_type_used", "abs(coef)")
        plot_title = f"{name} — LogisticRegression ({importance_used}) — cum {CUM_CUTOFF_PCT_TO_PLOT:.1f}%"
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    out_csv = OUT_DIR / f"{name}__importance.csv"
    out_png = OUT_DIR / f"{name}__cum{CUM_CUTOFF_PCT_TO_PLOT:.1f}pct.png"
    out_yaml = OUT_DIR / f"{name}__debug.yaml"

    df_out = df[["variable", "abs_importance", "pct_importance", "cum_pct_importance"]].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved: {out_csv.resolve()}")

    plot_top_cum_percent(
        df=df,
        out_png=out_png,
        cutoff_pct=CUM_CUTOFF_PCT_TO_PLOT,
        title=plot_title,
        max_features=200,
    )

    save_debug_yaml(
        out_yaml,
        {
            "pickle": pkl_path.name,
            "basename": name,
            "model_type": type(model).__name__,
            "model_kind": model_kind,
            "names_source": names_source,
            "n_features_names": len(feature_names),
            "n_rows_df": len(df),
            "importance_used": importance_used,
            "cum_cutoff_pct_plotted": CUM_CUTOFF_PCT_TO_PLOT,
        },
    )
    print(f"[OK] Debug YAML saved: {out_yaml.resolve()}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print(f"[INFO] Reading feature names: {NOMES_COLUNAS_PATH}")
    all_feature_names = load_feature_names_from_nomescolunas(NOMES_COLUNAS_PATH)

    print(f"[INFO] Scanning pkls: {PKLS_DIR} (prefix={ONLY_PREFIX})")
    files = find_model_pkls(PKLS_DIR, ONLY_PREFIX)
    if not files:
        raise RuntimeError(f"No pkls found: {PKLS_DIR}/{ONLY_PREFIX}*_model.pkl")

    print(f"[INFO] Found {len(files)} files:")
    for f in files:
        print("  -", f.name)

    for f in files:
        process_one_model(f, all_feature_names)


if __name__ == "__main__":
    main()
