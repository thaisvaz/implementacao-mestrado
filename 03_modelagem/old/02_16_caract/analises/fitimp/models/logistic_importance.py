from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import pandas as pd

from importance_common import importance_table_from_values


def build_importance_logistic(model: Any, feature_names: list[str]) -> Tuple[pd.DataFrame, str]:
    """
    LogisticRegression:
      - coef_ shape: (1, n_features) for binary (liblinear may be (1, n_features))
      - coef_ shape: (n_classes, n_features) for multinomial/ovr
    Importance proxy:
      - abs(coef) for binary
      - mean(abs(coef)) across classes for multiclass
    (This is NOT p-value significance; sklearn LR doesn't provide p-values.)
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
        # very unusual, but safe
        values = abs_coef.reshape(-1).tolist()
        meta = "abs(coef) [flattened]"

    df = importance_table_from_values(feature_names, values)
    df.attrs["importance_type_used"] = meta
    return df, meta
