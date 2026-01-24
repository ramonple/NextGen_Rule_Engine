from typing import Iterable, Optional, Sequence, Tuple, Set, List, Dict, Union
import numpy as np
import pandas as pd
import warnings
# -------------------------
# Helpers / small utilities
# -------------------------

_EPS = 1e-6


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _is_categorical(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)


def _safe_log(x: Union[pd.Series, np.ndarray, float]) -> Union[pd.Series, np.ndarray, float]:
    """Log with clipping to avoid -inf/inf."""
    return np.log(np.clip(x, _EPS, None))


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")