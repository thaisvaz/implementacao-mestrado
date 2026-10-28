from __future__ import annotations

from typing import List
import pandas as pd

from feature_names import align_feature_names


def importance_table_from_values(feature_names: List[str], values: List[float]) -> pd.DataFrame:
    """
    Standard output:
      variable, abs_importance, pct_importance, cum_pct_importance
    Expects values already represent absolute importance (>=0).
    """
    feature_names = align_feature_names(feature_names, len(values))
    values = values[: len(feature_names)]

    s = pd.Series(pd.to_numeric(values, errors="coerce")).fillna(0.0).astype(float)

    df = (
        pd.DataFrame({"variable": feature_names[: len(s)], "abs_importance": s.to_numpy()})
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )

    total = float(df["abs_importance"].sum())
    if total <= 0:
        df["pct_importance"] = 0.0
        df["cum_pct_importance"] = 0.0
    else:
        df["pct_importance"] = (df["abs_importance"] / total) * 100.0
        df["cum_pct_importance"] = df["pct_importance"].cumsum()

    df.attrs["total_importance"] = total
    return df
