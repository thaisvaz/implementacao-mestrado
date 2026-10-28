from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import re
import pandas as pd

from models.lightgbm_importance import build_importance_lightgbm
from models.logistic_importance import build_importance_logistic
from models.ridge_importance import build_importance_ridge
from models.linearsvc_importance import build_importance_linearsvc
from models.sgd_importance import build_importance_sgd


@dataclass(frozen=True)
class ImportanceHandler:
    name: str
    can_handle: Callable[[str, Any], bool]
    compute_importance: Callable[[Any, list[str]], Tuple[pd.DataFrame, str]]


def normalize_model_key_for_folder(model_key: str) -> str:
    """
    filesystem-safe folder name derived from filename model_key
    """
    s = model_key.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_model"


def resolve_importance_handler(model_key: str, model: Any) -> Optional[ImportanceHandler]:
    """
    Decide handler based primarily on model_key (from filename),
    and secondarily on model attributes.
    """

    key = model_key.lower()

    handlers = [
        ImportanceHandler(
            name="lightgbm",
            can_handle=lambda k, m: "lightgbm" in k,
            compute_importance=lambda model, feature_names: build_importance_lightgbm(model, feature_names),
        ),
        ImportanceHandler(
            name="logistic_regression",
            can_handle=lambda k, m: "logistic_regression" in k,
            compute_importance=lambda model, feature_names: build_importance_logistic(model, feature_names),
        ),
        ImportanceHandler(
            name="ridge_classifier",
            can_handle=lambda k, m: "ridge_classifier" in k,
            compute_importance=lambda model, feature_names: build_importance_ridge(model, feature_names),
        ),
        ImportanceHandler(
            name="linearsvc",
            can_handle=lambda k, m: "linearsvc" in k,
            compute_importance=lambda model, feature_names: build_importance_linearsvc(model, feature_names),
        ),
        ImportanceHandler(
            name="sgd_classifier",
            can_handle=lambda k, m: "sgd_classifier" in k,
            compute_importance=lambda model, feature_names: build_importance_sgd(model, feature_names),
        ),
    ]

    for h in handlers:
        if h.can_handle(key, model):
            return h

    return None
