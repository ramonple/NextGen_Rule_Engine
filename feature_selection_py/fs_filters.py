from typing import Iterable, Optional, Sequence, Tuple, Set, List, Dict, Union

import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings

# =============================================
#       Correlation Analysis
# =============================================

def correlation_with_target(
    data: pd.DataFrame,
    variable_list: Sequence[str],
    bad_flag: str,
) -> pd.DataFrame:
    """
    Calculate correlation between numeric variables and the target (bad_flag).

    Notes:
    - Uses Pearson correlation (pandas corrwith).
    - Non-numeric variables are ignored.
    """
    _require_columns(data, [bad_flag])
    numeric_vars = [v for v in variable_list if v in data.columns and _is_numeric(data[v])]
    if not numeric_vars:
        return pd.DataFrame(columns=["Variable", "Correlation", "Abs Corr"])

    corr = data[numeric_vars].corrwith(data[bad_flag])
    out = corr.reset_index()
    out.columns = ["Variable", "Correlation"]
    out["Abs Corr"] = out["Correlation"].abs()
    return out.sort_values(by="Abs Corr", ascending=False).reset_index(drop=True)

    def correlation_between_features(data: pd.DataFrame, variable_list: Sequence[str]) -> pd.DataFrame:
    """Calculate pairwise correlation between features (numeric)."""
    valid_vars = [v for v in variable_list if v in data.columns and _is_numeric(data[v])]
    if not valid_vars:
        return pd.DataFrame()
    return data[valid_vars].corr()

    def get_highly_corr(corr_matrix: pd.DataFrame, corr_threshold: float) -> pd.DataFrame:
    """
    Get pairs of features with absolute correlation above the threshold.

    Returns columns: Variable 1, Variable 2, Correlation
    """
    if corr_matrix.empty:
        return pd.DataFrame(columns=["Variable 1", "Variable 2", "Correlation"])

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = upper.stack(dropna=True).reset_index()
    pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
    pairs = pairs.loc[pairs["Correlation"].abs() > corr_threshold].reset_index(drop=True)
    return pairs






# =================================================
#      Multicollinearity Analysis - VIF
# =================================================

def calculate_vif(
    data: pd.DataFrame,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    vif_threshold: float = 10.0,
    drop_high_vif: bool = True,
    max_iter: int = 100,
    fillna: str = "median",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate Variance Inflation Factor (VIF) for numeric predictors.

    Parameters
    ----------
    fillna:
        - "drop": drop rows with any NA in predictors
        - "median": fill NA with median (default)
        - "zero": fill NA with zero

    Returns
    -------
    vif_data: DataFrame with Feature and VIF
    remaining_features: list of remaining feature names (after iterative dropping)
    """
    _require_columns(data, [bad_flag, bal_variable, bad_bal_variable])

    exclude = {bad_flag, bal_variable, bad_bal_variable}
    numerical_cols = [c for c in data.select_dtypes(include=[np.number]).columns.tolist() if c not in exclude]
    if not numerical_cols:
        raise ValueError("No numerical predictor columns available.")

    X = data[numerical_cols].copy()

    if fillna == "drop":
        X = X.dropna(axis=0)
    elif fillna == "median":
        X = X.apply(lambda s: s.fillna(s.median()), axis=0)
    elif fillna == "zero":
        X = X.fillna(0)
    else:
        raise ValueError("fillna must be one of: 'drop', 'median', 'zero'.")

    # Remove constant columns
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns after removing constants / exclusions.")

    remaining = X.columns.tolist()

    it = 0
    while True:
        it += 1
        if it > max_iter:
            warnings.warn("Reached max_iter while dropping high-VIF features; returning current result.")
            break

        # statsmodels VIF requires float ndarray
        X_values = X.astype(float).values

        vif_data = pd.DataFrame(
            {"Feature": X.columns, "VIF": [variance_inflation_factor(X_values, i) for i in range(X.shape[1])]}
        )

        max_vif = float(vif_data["VIF"].max())
        if drop_high_vif and max_vif > vif_threshold and X.shape[1] > 1:
            drop_feature = vif_data.sort_values(by="VIF", ascending=False)["Feature"].iloc[0]
            X = X.drop(columns=[drop_feature])
            if drop_feature in remaining:
                remaining.remove(drop_feature)
        else:
            break

    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True), remaining


# =============================================
#      Worst-tail rankers
# =============================================
def rank_features_by_worst_tail_bad_vol(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    worst_pct: float = 0.05,          # e.g. 0.05 for worst 5%
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,       # 0 => infer if missing
) -> pd.DataFrame:
    """
    Rank numerical features by BAD VOLUME captured in the worst X% tail.
    """

    if not (0 < worst_pct < 1):
        raise ValueError("worst_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = data.copy()

    # --- target cleaning ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    overall_bad_rate = df[bad_flag].mean()
    if pd.isna(overall_bad_rate):
        raise ValueError("Overall bad rate is NaN (bad_flag might be empty after cleaning).")

    # --- normalise data_dictionary columns (case-insensitive) ---
    dd = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]

        # accept both "variable"/"definition"/"direction" in any case
        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(f"data_dictionary must include columns: {required} (case-insensitive). Got: {set(dd.columns)}")

        dd["variable"] = dd["variable"].astype(str).str.strip()

        # quick lookup table by variable
        dd_lookup = dd.set_index("variable", drop=False)

    def clean_variable_name(target_variable: str) -> str:
        s = str(target_variable)
        cleaned = s.split("_")[-1] if "_" in s else s
        cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
        return cleaned

    results = []

    for col in num_list:
        if col not in df.columns:
            continue

        s_raw = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            s_raw = s_raw.mask(s_raw == ms)

        mask = s_raw.notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x = s_raw.loc[mask]
        y_sub = df.loc[mask, bad_flag]

        cleaned_name = clean_variable_name(col)

        # --- get direction & definition from dictionary if possible ---
        definition = None
        direction = direction_default

        if dd is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        # direction meaning:
        #  1 => high values are worse (worst tail = top X%)
        # -1 => low values are worse (worst tail = bottom X%)
        #  0/unknown => infer via Spearman correlation with target
        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1  # fallback
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- compute worst-tail cutoff + bad rate ---
        if direction_used == 1:
            cutoff = float(x.quantile(1 - worst_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(worst_pct*100)}%"
        else:
            cutoff = float(x.quantile(worst_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(worst_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n

        results.append({
            "feature": col,                         # raw dataset column name
            "cleaned_name": cleaned_name,          # cleaned name used for dictionary matching
            "definition": definition,              # from data_dictionary['definition']
            "direction_used": direction_used,      # +1 high_worse / -1 low_worse
            "tail_side": tail_side,
            "cutoff": cutoff,
            "tail_n": tail_n,                      # volume
            "tail_bad_n": tail_bad_n,              # bad volume
            "tail_bad_rate": tail_bad_rate,        # bad rate in worst tail
            "overall_bad_rate": overall_bad_rate,
            "spearman_corr_if_inferred": inferred_corr
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["tail_bad_rate", "tail_n"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out

def rank_features_by_worst_tail_bad_bal(
    data: pd.DataFrame,
    bad_flag: str,
    bad_bal: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    worst_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,
):
    """
    Rank numerical features by BAD BALANCE captured in the worst X% tail.
    """

    df = data.copy()
    total_bad = df.loc[df[bad_flag] == 1, bad_bal].sum()

    # --- clean target ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    # ---  balance ---

    bal = pd.to_numeric(df[bad_bal], errors="coerce")
    df[bad_bal] = bal

    # --- prepare data dictionary ---
    dd = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]

        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(f"data_dictionary must include columns: {required}")

        dd["variable"] = dd["variable"].astype(str).str.strip()
        dd_lookup = dd.set_index("variable", drop=False)

    # --- cleaning rule you defined ---
    def clean_variable_name(target_variable: str) -> str:
        s = str(target_variable)
        cleaned = s.split("_")[-1] if "_" in s else s
        cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
        return cleaned

    results = []

    for col in num_list:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        mask = x.notna() & df[bad_bal].notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x = x.loc[mask]
        y_sub = df.loc[mask, bad_flag]
        bal_sub = df.loc[mask, bad_bal]

        cleaned_name = clean_variable_name(col)

        # --- get direction & definition from dictionary ---
        definition = None
        direction = direction_default

        if dd is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        # infer direction if missing
        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- determine worst tail ---
        if direction_used == 1:
            cutoff = float(x.quantile(1 - worst_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(worst_pct*100)}%"
        else:
            cutoff = float(x.quantile(worst_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(worst_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        # --- bad / volume metrics ---
        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n

        # --- BALANCE metrics  ---
        tail_total_balance = bal_sub.loc[tail_mask].sum()
        tail_bad_balance = bal_sub.loc[tail_mask & (y_sub == 1)].sum()

        bad_balance_rate_in_tail = (
            tail_bad_balance / tail_total_balance if tail_total_balance > 0 else np.nan
        )

        bad_balance_share_of_total = (
            tail_bad_balance / total_bad if total_bad > 0 else np.nan
        )

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            # volume & quality
            "tail_n": tail_n,
            "tail_bad_n": tail_bad_n,
            "tail_bad_rate": tail_bad_rate,

            # balance focus
            "tail_total_balance": tail_total_balance,
            "tail_bad_balance": tail_bad_balance,
            "bad_balance_rate_in_tail": bad_balance_rate_in_tail,
            "bad_balance_share_of_total": bad_balance_share_of_total,

            # diagnostics
            "spearman_corr_if_inferred": inferred_corr,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    #  Rank by BAD BALANCE captured
    out = out.sort_values(
        ["tail_bad_balance", "bad_balance_share_of_total"],
        ascending=[False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out