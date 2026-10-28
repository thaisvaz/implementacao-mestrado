from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import pandas as pd

from importance_common import importance_table_from_values


def build_importance_sgd(model: Any, feature_names: list[str]) -> Tuple[pd.DataFrame, str]:
    """
    SGDClassifier (linear):
      - coef_ shape: (1, n_features) or (n_classes, n_features)
    Importance proxy:
      - abs(coef) or mean(abs(coef)) across classes
    """
    if not hasattr(model, "coef_"):
        raise TypeError(f"Model has no coef_: {type(model)}")

    coef = np.asarray(model.coef_)
    abs_coef = np.abs(coef)

    if abs_coef.ndim == 2 and abs_coef.shape[0] == 1:
        values = abs_coef[0].tolist()
        meta = "abs(coef) [binary]"
    elif abs_coef.ndim == 2:
        values = abs_coef.mean(axis=0).tolist()
        meta = f"mean(abs(coef)) across {abs_coef.shape[0]} classes"
    else:
        values = abs_coef.reshape(-1).tolist()
        meta = "abs(coef) [flattened]"

    df = importance_table_from_values(feature_names, values)
    df.attrs["importance_type_used"] = meta
    return df, meta
