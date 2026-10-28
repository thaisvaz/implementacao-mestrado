from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def find_all_prefixed_model_pkls(pkls_dir: Path) -> List[Path]:
    """
    Only files like: <feat>__<model>_<date>_model.pkl
    Example: rtr__lightgbm_21012026_model.pkl
    Ignores ones without '__' (like lightgbm_1768..._model.pkl).
    """
    if not pkls_dir.exists():
        raise FileNotFoundError(f"pkls dir not found: {pkls_dir}")

    out: List[Path] = []
    for p in sorted(pkls_dir.glob("*_model.pkl")):
        if "__" in p.name:
            out.append(p)
    return out


def parse_feature_and_model_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    filename like: rtr__lightgbm_21012026_model.pkl
      -> ("rtr", "lightgbm")

    filename like: yty__logistic_regression_elasticnet_21012026_model.pkl
      -> ("yty", "logistic_regression_elasticnet")

    Returns None if doesn't match.
    """
    if not filename.endswith("_model.pkl"):
        return None

    base = filename[: -len("_model.pkl")]  # rtr__lightgbm_21012026
    if "__" not in base:
        return None

    feat, rest = base.split("__", 1)

    # remove trailing _<8digits> (date)
    rest = re.sub(r"_\d{8}$", "", rest)

    model_key = rest
    if not feat or not model_key:
        return None

    return feat, model_key


def model_basename(path: Path) -> str:
    # rtr__lightgbm_21012026_model.pkl -> rtr__lightgbm_21012026
    return re.sub(r"_model\.pkl$", "", path.name)
