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
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd


def clean_variable_name(name: str) -> str:
    """
    Simple, deterministic "clean" to match dictionary keys.
    Replace with your own if you already have one.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    # Keep alnum + underscore
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", ".", ":", "/", "\\"):
            out.append("_")
        else:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2.strip("_")


def _safe_get_dd_row(dd_lookup: Optional[pd.DataFrame], cleaned_name: str) -> Optional[pd.Series]:
    """
    Robustly fetch a single dictionary row for `cleaned_name`, even if:
      - dd_lookup has duplicate index values (returns DataFrame)
      - row exists but fields are weird types
    Expect dd_lookup indexed by cleaned_name.
    """
    if dd_lookup is None or cleaned_name not in dd_lookup.index:
        return None

    row = dd_lookup.loc[cleaned_name]
    if isinstance(row, pd.DataFrame):
        # Prefer first row with non-null direction; else first row
        if "direction" in row.columns:
            row2 = row[row["direction"].notna()]
            row = row2.iloc[0] if len(row2) else row.iloc[0]
        else:
            row = row.iloc[0]

    # Now row should be a Series
    if not isinstance(row, pd.Series):
        try:
            row = pd.Series(row)
        except Exception:
            return None
    return row


def _coerce_direction(value: Any, direction_default: int) -> int:
    """
    Convert direction to int safely.
    Accepts scalar, Series/list/ndarray (takes first non-null).
    """
    d = value

    if isinstance(d, pd.Series):
        d = d.dropna()
        d = d.iloc[0] if len(d) else np.nan
    elif isinstance(d, (list, tuple)):
        d2 = [x for x in d if pd.notnull(x)]
        d = d2[0] if len(d2) else np.nan
    elif isinstance(d, np.ndarray):
        d2 = d[~pd.isnull(d)]
        d = d2[0] if len(d2) else np.nan

    if pd.isnull(d):
        return direction_default

    try:
        return int(float(d))
    except Exception:
        return direction_default


def _infer_direction_from_corr(x: pd.Series, y: pd.Series) -> int:
    """
    Infer direction from Spearman correlation between x and y.
    Returns +1 if corr >= 0, else -1 (never returns 0).
    """
    try:
        corr = x.corr(y, method="spearman")
    except Exception:
        corr = np.nan

    if pd.isna(corr):
        return 1
    return 1 if corr >= 0 else -1


def _tail_mask_by_direction(x: pd.Series, worst_pct: float, direction: int) -> pd.Series:
    """
    Build a boolean mask for the "worst tail" of x.
    direction = +1 => higher x is "worse" => take upper tail (>= quantile(1-worst_pct))
    direction = -1 => lower x is "worse" => take lower tail (<= quantile(worst_pct))
    """
    if not (0 < worst_pct < 1):
        raise ValueError("worst_pct must be in (0, 1)")

    if direction >= 0:
        q = x.quantile(1 - worst_pct)
        return x >= q
    else:
        q = x.quantile(worst_pct)
        return x <= q


def _prepare_dd_lookup(data_dictionary: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Prepare dd_lookup indexed by cleaned_name.
    Expected (if present) columns: ['name' or 'variable', 'definition', 'direction'].
    If your dictionary already comes with cleaned_name index, you can pass it directly.
    """
    if data_dictionary is None:
        return None

    dd = data_dictionary.copy()

    # Choose a "name" column if not already indexed
    if dd.index.name is None or dd.index.dtype == "int64":
        # try common name columns
        name_col = None
        for cand in ("cleaned_name", "name", "variable", "var_name", "field", "feature"):
            if cand in dd.columns:
                name_col = cand
                break

        if name_col is None:
            # cannot build lookup
            return None

        if name_col == "cleaned_name":
            dd["__cleaned_name__"] = dd["cleaned_name"].astype(str)
        else:
            dd["__cleaned_name__"] = dd[name_col].astype(str).map(clean_variable_name)

        dd = dd.set_index("__cleaned_name__")
    else:
        # index exists; assume it's already cleaned_name-like
        pass

    return dd


def rank_features_by_worst_tail_bad_vol(
    df: pd.DataFrame,
    bad_flag: str,
    feature_cols: Sequence[str],
    data_dictionary: Optional[pd.DataFrame] = None,
    *,
    worst_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: Sequence[Union[int, float]] = (-9999,),
    direction_default: int = 0,
) -> pd.DataFrame:
    """
    Rank numeric features by BAD VOLUME RATE in the worst tail.

    Bad volume rate in tail = tail_bad_n / tail_total_n
    Also returns:
      - tail_bad_n, tail_total_n
      - tail_bad_rate
      - tail_share_of_total_n (how much volume sits in tail)
      - tail_bad_share_of_total_bad_n (how much bad volume captured in tail)
      - direction_used (+1 upper tail, -1 lower tail)
      - dictionary definition/direction if provided
    """
    dd_lookup = _prepare_dd_lookup(data_dictionary)

    if bad_flag not in df.columns:
        raise KeyError(f"bad_flag column '{bad_flag}' not found in df")

    y_all = df[bad_flag]
    # total bad volume in whole dataset (non-missing bad_flag only)
    valid_y_mask = y_all.notna()
    total_n_all = int(valid_y_mask.sum())
    total_bad_n_all = float(y_all.loc[valid_y_mask].sum())

    results: list[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        mask = x.notna() & df[bad_flag].notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x_sub = x.loc[mask]
        y_sub = df.loc[mask, bad_flag].astype(float)

        cleaned_name = clean_variable_name(col)
        definition = None

        # direction: dictionary overrides if non-zero; else infer from corr; else default to +1
        direction = direction_default
        row = _safe_get_dd_row(dd_lookup, cleaned_name)
        if row is not None:
            definition = row.get("definition", None)
            direction = _coerce_direction(row.get("direction", np.nan), direction_default)

        if direction == 0:
            direction = _infer_direction_from_corr(x_sub, y_sub)

        tail_mask = _tail_mask_by_direction(x_sub, worst_pct=worst_pct, direction=direction)

        tail_total_n = int(tail_mask.sum())
        if tail_total_n == 0:
            continue

        tail_bad_n = float(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_total_n

        tail_share_of_total_n = tail_total_n / max(total_n_all, 1)
        tail_bad_share_of_total_bad_n = tail_bad_n / max(total_bad_n_all, 1e-12)

        results.append(
            {
                "feature": col,
                "cleaned_name": cleaned_name,
                "definition": definition,
                "direction_used": int(direction),
                "n_non_missing": n_non_missing,
                "tail_total_n": tail_total_n,
                "tail_bad_n": tail_bad_n,
                "tail_bad_rate": tail_bad_rate,
                "tail_share_of_total_n": tail_share_of_total_n,
                "tail_bad_share_of_total_bad_n": tail_bad_share_of_total_bad_n,
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Sort by the *rate* (worst tails), break ties by tail_total_n (stability), then by captured bad
    out = out.sort_values(
        ["tail_bad_rate", "tail_total_n", "tail_bad_n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return out


def rank_features_by_worst_tail_bad_bal(
    df: pd.DataFrame,
    bad_flag: str,
    feature_cols: Sequence[str],
    total_bal: str,
    bad_bal: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    *,
    worst_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: Sequence[Union[int, float]] = (-9999,),
    direction_default: int = 0,
) -> pd.DataFrame:
    """
    Rank numeric features by BAD BALANCE RATE in the worst tail.

    Bad balance rate in tail = tail_bad_balance / tail_total_balance

    IMPORTANT:
      - We do NOT require bad_bal to be non-missing; missing bad_bal is treated as 0.
      - total_bal must be non-missing in the mask.
    """
    dd_lookup = _prepare_dd_lookup(data_dictionary)

    for req in (bad_flag, total_bal, bad_bal):
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' not found in df")

    # overall totals (treat missing bad_bal as 0; require total_bal not missing)
    overall_mask = df[bad_flag].notna() & df[total_bal].notna()
    total_balance_all = float(df.loc[overall_mask, total_bal].sum(skipna=True))
    total_bad_balance_all = float(df.loc[overall_mask, bad_bal].fillna(0).sum(skipna=True))

    results: list[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")

        # treat sentinels as missing
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        # need x + total balance + bad_flag; bad_bal can be missing -> fill 0
        mask = x.notna() & df[total_bal].notna() & df[bad_flag].notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x_sub = x.loc[mask]
        y_sub = df.loc[mask, bad_flag].astype(float)
        total_bal_sub = pd.to_numeric(df.loc[mask, total_bal], errors="coerce")
        bad_bal_sub = pd.to_numeric(df.loc[mask, bad_bal], errors="coerce").fillna(0)

        # drop any rows where total_bal becomes NaN after coercion
        good_bal_mask = total_bal_sub.notna()
        x_sub = x_sub.loc[good_bal_mask]
        y_sub = y_sub.loc[good_bal_mask]
        total_bal_sub = total_bal_sub.loc[good_bal_mask]
        bad_bal_sub = bad_bal_sub.loc[good_bal_mask]

        if len(x_sub) < min_non_missing:
            continue

        cleaned_name = clean_variable_name(col)
        definition = None

        direction = direction_default
        row = _safe_get_dd_row(dd_lookup, cleaned_name)
        if row is not None:
            definition = row.get("definition", None)
            direction = _coerce_direction(row.get("direction", np.nan), direction_default)

        if direction == 0:
            direction = _infer_direction_from_corr(x_sub, y_sub)

        tail_mask = _tail_mask_by_direction(x_sub, worst_pct=worst_pct, direction=direction)

        tail_total_balance = float(total_bal_sub.loc[tail_mask].sum(skipna=True))
        if tail_total_balance <= 0:
            continue

        tail_bad_balance = float(bad_bal_sub.loc[tail_mask].sum(skipna=True))
        bad_balance_rate_in_tail = tail_bad_balance / tail_total_balance

        tail_balance_share_of_total = tail_total_balance / max(total_balance_all, 1e-12)
        tail_bad_balance_share_of_total_bad = tail_bad_balance / max(total_bad_balance_all, 1e-12)

        results.append(
            {
                "feature": col,
                "cleaned_name": cleaned_name,
                "definition": definition,
                "direction_used": int(direction),
                "n_non_missing": int(len(x_sub)),
                "tail_total_balance": tail_total_balance,
                "tail_bad_balance": tail_bad_balance,
                "bad_balance_rate_in_tail": bad_balance_rate_in_tail,
                "tail_balance_share_of_total": tail_balance_share_of_total,
                "tail_bad_balance_share_of_total_bad": tail_bad_balance_share_of_total_bad,
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Sort by the *rate* (what you want), break ties by tail_total_balance, then captured bad balance
    out = out.sort_values(
        ["bad_balance_rate_in_tail", "tail_total_balance", "tail_bad_balance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return out


#============================================
#  best tail analysis
#=============================================
def rank_features_by_best_tail_good_vol(
    data: pd.DataFrame,
    bad_flag: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    best_pct: float = 0.05,           # e.g. 0.05 for best 5%
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,       # 0 => infer if missing
) -> pd.DataFrame:
    """
    Rank numerical features by GOOD VOLUME captured in the best X% tail
    (i.e., the tail with the lowest bad rate, based on direction).

    Best-tail definition (mirrors worst-tail logic):
      - if direction_used == 1 (high values are worse), best tail = bottom X%
      - if direction_used == -1 (low values are worse), best tail = top X%
      - if direction_used == 0, infer via Spearman corr with bad_flag
    """

    if not (0 < best_pct < 1):
        raise ValueError("best_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = data.copy()

    # --- target cleaning ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    overall_bad_rate = df[bad_flag].mean()
    if pd.isna(overall_bad_rate):
        raise ValueError("Overall bad rate is NaN (bad_flag might be empty after cleaning).")

    overall_good_rate = 1 - overall_bad_rate

    # --- normalise data_dictionary columns (case-insensitive) ---
    dd = None
    dd_lookup = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]

        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(
                f"data_dictionary must include columns: {required} (case-insensitive). Got: {set(dd.columns)}"
            )

        dd["variable"] = dd["variable"].astype(str).str.strip()
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

        # --- dictionary info ---
        definition = None
        direction = direction_default

        if dd_lookup is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        # direction meaning:
        #  1 => high values are worse
        # -1 => low values are worse
        #  0 => infer
        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1  # fallback
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- compute best-tail cutoff + stats (FLIPPED from worst-tail) ---
        if direction_used == 1:
            # high worse => best tail is bottom X%
            cutoff = float(x.quantile(best_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(best_pct*100)}%"
        else:
            # low worse => best tail is top X%
            cutoff = float(x.quantile(1 - best_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(best_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_good_n = int(tail_n - tail_bad_n)

        tail_bad_rate = tail_bad_n / tail_n
        tail_good_rate = 1 - tail_bad_rate

        # simple “lift” for goodness
        good_rate_lift = tail_good_rate / overall_good_rate if overall_good_rate > 0 else np.nan

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            "tail_n": tail_n,
            "tail_bad_n": tail_bad_n,
            "tail_good_n": tail_good_n,
            "tail_bad_rate": tail_bad_rate,
            "tail_good_rate": tail_good_rate,

            "overall_bad_rate": overall_bad_rate,
            "overall_good_rate": overall_good_rate,
            "good_rate_lift": good_rate_lift,

            "spearman_corr_if_inferred": inferred_corr
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Approval feature screening: low bad rate first, then more good volume
    out = out.sort_values(
        ["tail_bad_rate", "tail_good_n", "tail_n"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rank_features_by_best_tail_good_bal(
    data: pd.DataFrame,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    num_list: list,
    data_dictionary: pd.DataFrame = None,
    best_pct: float = 0.05,
    min_non_missing: int = 200,
    missing_sentinels: tuple = (-9999,),
    direction_default: int = 0,
) -> pd.DataFrame:
    """
    Rank numerical features by GOOD BALANCE captured in the best X% tail.

    Uses:
      - total_bal: total exposure/balance
      - bad_bal: bad exposure/balance
      - good_bal is derived as (total_bal - bad_bal)
    """

    if not (0 < best_pct < 1):
        raise ValueError("best_pct must be between 0 and 1 (e.g., 0.05 for 5%).")

    df = data.copy()

    # --- clean target ---
    y = pd.to_numeric(df[bad_flag], errors="coerce")
    df = df.loc[y.isin([0, 1])].copy()
    df[bad_flag] = y.loc[df.index].astype(int)

    # --- balances ---
    df[total_bal] = pd.to_numeric(df[total_bal], errors="coerce")
    df[bad_bal] = pd.to_numeric(df[bad_bal], errors="coerce")

    # derived good balance
    good_bal_all = (df[total_bal] - df[bad_bal])
    total_good_balance_all = float(good_bal_all.sum(skipna=True))

    # --- prepare data dictionary ---
    dd_lookup = None
    if data_dictionary is not None:
        dd = data_dictionary.copy()
        dd.columns = [c.strip().lower() for c in dd.columns]
        required = {"variable", "definition", "direction"}
        if not required.issubset(set(dd.columns)):
            raise ValueError(f"data_dictionary must include columns: {required}")
        dd["variable"] = dd["variable"].astype(str).str.strip()
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

        x = pd.to_numeric(df[col], errors="coerce")
        for ms in missing_sentinels:
            x = x.mask(x == ms)

        # need x + balances
        mask = x.notna() & df[total_bal].notna() & df[bad_bal].notna()
        n_non_missing = int(mask.sum())
        if n_non_missing < min_non_missing:
            continue

        x = x.loc[mask]
        y_sub = df.loc[mask, bad_flag]
        total_bal_sub = df.loc[mask, total_bal]
        bad_bal_sub = df.loc[mask, bad_bal]
        good_bal_sub = (total_bal_sub - bad_bal_sub)

        cleaned_name = clean_variable_name(col)

        definition = None
        direction = direction_default

        if dd_lookup is not None and cleaned_name in dd_lookup.index:
            row = dd_lookup.loc[cleaned_name]
            definition = row.get("definition", None)
            d = row.get("direction", np.nan)
            if pd.notnull(d):
                try:
                    direction = int(float(d))
                except Exception:
                    direction = direction_default

        inferred_corr = np.nan
        direction_used = direction

        if direction_used == 0:
            inferred_corr = x.corr(y_sub, method="spearman")
            if pd.isna(inferred_corr) or inferred_corr == 0:
                direction_used = 1
            else:
                direction_used = 1 if inferred_corr > 0 else -1

        # --- determine best tail (FLIPPED from worst tail) ---
        if direction_used == 1:
            cutoff = float(x.quantile(best_pct))
            tail_mask = x <= cutoff
            tail_side = f"bottom_{int(best_pct*100)}%"
        else:
            cutoff = float(x.quantile(1 - best_pct))
            tail_mask = x >= cutoff
            tail_side = f"top_{int(best_pct*100)}%"

        tail_n = int(tail_mask.sum())
        if tail_n == 0:
            continue

        # --- volume metrics ---
        tail_bad_n = int(y_sub.loc[tail_mask].sum())
        tail_bad_rate = tail_bad_n / tail_n
        tail_good_n = int(tail_n - tail_bad_n)

        # --- balance metrics ---
        tail_total_balance = float(total_bal_sub.loc[tail_mask].sum())
        tail_bad_balance = float(bad_bal_sub.loc[tail_mask].sum())
        tail_good_balance = float(good_bal_sub.loc[tail_mask].sum())

        good_balance_rate_in_tail = (
            tail_good_balance / tail_total_balance if tail_total_balance > 0 else np.nan
        )

        good_balance_share_of_total = (
            tail_good_balance / total_good_balance_all if total_good_balance_all > 0 else np.nan
        )

        results.append({
            "feature": col,
            "cleaned_name": cleaned_name,
            "definition": definition,
            "direction_used": direction_used,
            "tail_side": tail_side,
            "cutoff": cutoff,

            "tail_n": tail_n,
            "tail_good_n": tail_good_n,
            "tail_bad_n": tail_bad_n,
            "tail_bad_rate": tail_bad_rate,

            "tail_total_balance": tail_total_balance,
            "tail_good_balance": tail_good_balance,
            "tail_bad_balance": tail_bad_balance,

            "good_balance_rate_in_tail": good_balance_rate_in_tail,
            "good_balance_share_of_total": good_balance_share_of_total,

            "spearman_corr_if_inferred": inferred_corr,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Approval feature screening: capture more good balance, and keep tail bad rate low
    out = out.sort_values(
        ["tail_bad_rate", "tail_good_balance", "good_balance_share_of_total"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
