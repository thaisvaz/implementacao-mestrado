from __future__ import annotations

from typing import Any, Tuple
import pandas as pd

import config
from importance_common import importance_table_from_values


def _get_lgbm_booster(model: Any) -> Any:
    if hasattr(model, "feature_importance") and hasattr(model, "feature_name"):
        return model
    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_")
        if hasattr(booster, "feature_importance") and hasattr(booster, "feature_name"):
            return booster
    raise TypeError(f"Not a LightGBM model/booster: {type(model)}")


def build_importance_lightgbm(model: Any, feature_names: list[str]) -> Tuple[pd.DataFrame, str]:
    booster = _get_lgbm_booster(model)

    primary_type = config.LGBM_IMPORTANCE_TYPE
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
