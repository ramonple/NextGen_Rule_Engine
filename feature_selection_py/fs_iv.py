from typing import Iterable, Optional, Sequence, Tuple, Set, List, Dict, Union
import warnings

import numpy as np
import pandas as pd

# =============================================
#     categorical & numerical features split
# =============================================

def num_cat_list(
    data: pd.DataFrame,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
) -> Tuple[List[str], List[str]]:
    """
    Split the dataset's columns into numeric and categorical lists.

    Notes:
    - Excludes `bad_flag`, `bal_variable`, and `bad_bal_variable` from numeric list.
    - Categorical list is object/categorical dtypes.
    """
    _require_columns(data, [bad_flag, bal_variable, bad_bal_variable])

    num_list = [col for col in data.columns if _is_numeric(data[col])]
    cat_list = [col for col in data.columns if _is_categorical(data[col])]

    items_to_remove = {bad_flag, bal_variable, bad_bal_variable}
    num_list = [col for col in num_list if col not in items_to_remove]

    return num_list, cat_list


# =============================================
#       information value calculation
# =============================================

def information_value_calculation(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: Sequence[str],
    cat_list: Sequence[str],
    bins: int = 10,
    eps: float = _EPS,
) -> pd.DataFrame:
    """
    Calculate IV (Information Value) for numeric and categorical variables.

    Assumptions:
    - `bad_flag` is binary where 1=bad and 0=good (or any non-zero treated as bad).
    - For numeric variables, values are binned into quantiles (qcut) and IV is computed on the bins.
    - For categorical variables, IV is computed on each category.

    Returns:
    DataFrame with columns: Variable, Information Value
    """
    _require_columns(data, [bad_flag])
    # normalize target to {0,1}
    target = (data[bad_flag].fillna(0) > 0).astype(int)

    names: List[str] = []
    scores: List[float] = []

    # ---- numeric ----
    for col in num_list:
        if col not in data.columns:
            continue

        x = data[col]
        df = pd.DataFrame({bad_flag: target, col: x}).dropna(subset=[col]).copy()
        if df.empty or df[col].nunique(dropna=True) < 2:
            continue

        df["account"] = 1

        # qcut can fail with too many duplicates; duplicates='drop' helps but may still yield 0 bins
        try:
            cuts = pd.qcut(df[col], q=bins, duplicates="drop")
        except Exception:
            # fallback to equal-width bins
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cuts = pd.cut(df[col], bins=min(bins, max(2, df[col].nunique())), duplicates="drop")

        grouped = df.groupby(cuts, observed=True).sum(numeric_only=True)

        bad = grouped[bad_flag].astype(float)
        total = grouped["account"].astype(float)
        good = total - bad

        bad_total = bad.sum()
        good_total = good.sum()

        if bad_total <= 0 or good_total <= 0:
            # IV is undefined / uninformative if only one class exists
            score = 0.0
        else:
            db = (bad / bad_total).clip(lower=eps)
            dg = (good / good_total).clip(lower=eps)
            woe = _safe_log(dg / db)
            iv = (dg - db) * woe
            score = float(np.nansum(iv.values))

        names.append(col)
        scores.append(score)

    # ---- categorical ----
    for col in cat_list:
        if col not in data.columns:
            continue

        s = data[col].astype("object")
        df = pd.DataFrame({bad_flag: target, col: s}).copy()
        df[col] = df[col].fillna("__MISSING__")

        # Ensure we have both classes
        if df[bad_flag].nunique() < 2:
            names.append(col)
            scores.append(0.0)
            continue

        ct = pd.crosstab(df[col], df[bad_flag])
        # handle missing columns (e.g., all good or all bad)
        if 0 not in ct.columns:
            ct[0] = 0
        if 1 not in ct.columns:
            ct[1] = 0
        ct = ct[[0, 1]]
        ct.columns = ["good", "bad"]

        good = ct["good"].astype(float)
        bad = ct["bad"].astype(float)

        good_dist = (good / max(good.sum(), eps)).clip(lower=eps)
        bad_dist = (bad / max(bad.sum(), eps)).clip(lower=eps)

        woe = _safe_log(bad_dist / good_dist)
        iv = (good_dist - bad_dist) * woe
        score = float(np.nansum(iv.values))

        names.append(col)
        scores.append(score)

    result_df = pd.DataFrame({"Variable": names, "Information Value": scores})
    result_df = result_df.sort_values(by="Information Value", ascending=False).reset_index(drop=True)
    return result_df


def information_value_calculation_dictionary(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: Sequence[str],
    cat_list: Sequence[str],
    data_dictionary: Optional[pd.DataFrame] = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Same as `information_value_calculation`, with optional enrichment using a data dictionary.

    Expected data_dictionary columns (if provided):
    - Variable, Definition, Val    enriched["cleaned_var"] = enriched["Variable"].astype(str).apply(
        lambda x: x.split("_")[-1] if "_" in x else x
    )d Min, Valid Max, Direction
    """
    iv_df = information_value_calculation(data, bad_flag, num_list, cat_list, bins=bins)

    if data_dictionary is None:
        return iv_df

    if "Variable" not in data_dictionary.columns:
        warnings.warn("data_dictionary does not contain a 'Variable' column; returning IV table only.")
        return iv_df

    enriched = iv_df.copy()
    enriched["cleaned_var"] = enriched["Variable"].astype(str).apply(
        lambda x: x.split("_")[-1] if "_" in x else x
    )
    enriched = enriched.merge(
        data_dictionary,
        left_on="cleaned_var",
        right_on="Variable",
        how="left",
        suffixes=("", "_dict"),
    )

    # Keep a tidy column set if possible
    preferred_cols = [
        "Variable",
        "Information Value",
        "Definition",
        "Valid Min",
        "Valid Max",
        "Direction",
    ]
    existing = [c for c in preferred_cols if c in enriched.columns]
    return enriched[existing + [c for c in enriched.columns if c not in existing]]


    def iv_survived(
    iv_table: pd.DataFrame,
    iv_threshold: float,
    num_list: Sequence[str],
    cat_list: Sequence[str],
) -> Tuple[Set[str], int, int, int]:
    """
    Return features that pass IV filter and counts.

    Returns:
      (survived_set, num_count, cat_count, total_count)
    """
    if "Variable" not in iv_table.columns or "Information Value" not in iv_table.columns:
        raise KeyError("iv_table must contain 'Variable' and 'Information Value' columns.")

    filter_iv = iv_table.loc[iv_table["Information Value"] >= iv_threshold, "Variable"]
    survived_iv_set = set(filter_iv.astype(str).tolist())

    num_iv_survived = sum(var in survived_iv_set for var in num_list)
    cat_iv_survived = sum(var in survived_iv_set for var in cat_list)
    total_iv_survived = num_iv_survived + cat_iv_survived

    return survived_iv_set, num_iv_survived, cat_iv_survived, total_iv_survived
