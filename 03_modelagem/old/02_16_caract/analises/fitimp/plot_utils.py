from __future__ import annotations

from pathlib import Path
import pandas as pd


def plot_top_cum_percent(
    df: pd.DataFrame,
    out_png: Path,
    cutoff_pct: float,
    title: str,
    max_features: int | None = 200,
) -> int:
    """
    Plot features until cum_pct_importance reaches cutoff_pct (0..100).
    Returns number of features plotted.
    """
    import matplotlib.pyplot as plt

    if "cum_pct_importance" not in df.columns:
        raise ValueError("df must contain 'cum_pct_importance'")
    if df.empty:
        raise ValueError("df is empty")

    top = df[df["cum_pct_importance"] <= cutoff_pct].copy()

    # ensure at least 1 feature, include crossing feature
    if top.empty:
        top = df.head(1).copy()
    else:
        if len(top) < len(df):
            top = pd.concat([top, df.iloc[[len(top)]]], ignore_index=True)

    if max_features is not None and len(top) > max_features:
        top = top.head(max_features).copy()

    top = top.iloc[::-1]  # biggest at top in barh

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

    print(f"[OK] Plot saved: {out_png.resolve()} | n_features_plotted={len(top)} | cum_cutoff={cutoff_pct}%")
    return len(top)
