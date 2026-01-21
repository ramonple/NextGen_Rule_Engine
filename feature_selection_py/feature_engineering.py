import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_divide(a: pd.Series, b: pd.Series, *, epsilon: float = 0.0, fill_value: float = 0.0,
                 clip: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Safe elementwise division a / b.
    - epsilon: treats |b| <= epsilon as "zero"
    - fill_value: value used where division is not allowed
    - clip: optional (low, high) clipping of the output to tame extreme ratios
    """
    denom = b.to_numpy(dtype=float)
    num = a.to_numpy(dtype=float)

    mask = np.isfinite(denom) & (np.abs(denom) > float(epsilon))
    out = np.full_like(num, float(fill_value), dtype=float)
    out[mask] = num[mask] / denom[mask]

    if clip is not None:
        out = np.clip(out, clip[0], clip[1])

    return out


def generate_interactions(
    df: pd.DataFrame,
    feature_groups: Dict[str, Sequence[str]],
    operations: Sequence[str],
    *,
    max_order: int = 2,
    triplet_operations: Optional[Sequence[str]] = None,
    numeric_only: bool = True,
    epsilon: float = 0.0,
    div_fill_value: float = 0.0,
    div_clip: Optional[Tuple[float, float]] = None,
    max_new_features: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate interaction features within each feature group.

    Supported operations:
      - add, sub, mul, div, max, min, mean

    Parameters
    ----------
    df : DataFrame
    feature_groups : dict[group_name -> list of columns]
    operations : list of ops to apply for order=2
    max_order : {2,3}
    triplet_operations : ops for order=3 (defaults to intersection of operations and {'add','mul'})
    numeric_only : if True, only numeric columns are used
    epsilon : denom threshold for division
    div_fill_value : output used when denom is zero/invalid
    div_clip : optional clipping for division output
    max_new_features : optional guardrail (stop after creating this many features)
    verbose : if True prints skipped feature info

    Returns
    -------
    DataFrame of new features
    """
    allowed_ops = {"add", "sub", "mul", "div", "max", "min", "mean"}
    ops = [op for op in operations if op in allowed_ops]
    if not ops:
        raise ValueError(f"No valid operations provided. Allowed: {sorted(allowed_ops)}")

    if max_order not in (2, 3):
        raise ValueError("max_order must be 2 or 3")

    if triplet_operations is None:
        triplet_operations = [op for op in ops if op in {"add", "mul"}]
    else:
        triplet_operations = [op for op in triplet_operations if op in allowed_ops]

    new_features = pd.DataFrame(index=df.index)
    created = 0

    for group_name, features in feature_groups.items():
        # keep only columns that exist
        cols = [c for c in features if c in df.columns]

        if numeric_only:
            cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        # skip groups that don't have enough features
        if len(cols) < 2:
            continue

        for order in range(2, max_order + 1):
            for combo in itertools.combinations(cols, order):
                if max_new_features is not None and created >= max_new_features:
                    return new_features

                try:
                    if order == 2:
                        f1, f2 = combo
                        s1, s2 = df[f1], df[f2]

                        for op in ops:
                            if max_new_features is not None and created >= max_new_features:
                                return new_features

                            if op == "add":
                                name = f"{f1}_add_{f2}"
                                new_features[name] = s1 + s2
                            elif op == "sub":
                                name = f"{f1}_sub_{f2}"
                                new_features[name] = s1 - s2
                            elif op == "mul":
                                name = f"{f1}_mul_{f2}"
                                new_features[name] = s1 * s2
                            elif op == "div":
                                name = f"{f1}_div_{f2}"
                                new_features[name] = _safe_divide(
                                    s1, s2,
                                    epsilon=epsilon,
                                    fill_value=div_fill_value,
                                    clip=div_clip,
                                )
                            elif op == "max":
                                name = f"max_{f1}_{f2}"
                                new_features[name] = df[[f1, f2]].max(axis=1)
                            elif op == "min":
                                name = f"min_{f1}_{f2}"
                                new_features[name] = df[[f1, f2]].min(axis=1)
                            elif op == "mean":
                                name = f"mean_{f1}_{f2}"
                                new_features[name] = df[[f1, f2]].mean(axis=1)

                            created += 1

                    elif order == 3:
                        if not triplet_operations:
                            continue

                        f1, f2, f3 = combo
                        s1, s2, s3 = df[f1], df[f2], df[f3]

                        for op in triplet_operations:
                            if max_new_features is not None and created >= max_new_features:
                                return new_features

                            if op == "add":
                                name = f"{f1}_add_{f2}_add_{f3}"
                                new_features[name] = s1 + s2 + s3
                            elif op == "mul":
                                name = f"{f1}_mul_{f2}_mul_{f3}"
                                new_features[name] = s1 * s2 * s3
                            # (you can extend to mean/max/min for triplets if you want)

                            created += 1

                except Exception as e:
                    if verbose:
                        print(f"[{group_name}] Skipping combo={combo} due to error: {e}")

    return new_features
