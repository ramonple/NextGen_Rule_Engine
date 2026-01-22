from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Set, List, Dict, Union

import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


try:
    import seaborn as sns  # optional, used in good_bad_distribution
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:
    import plotly.graph_objects as go  # optional, used in plot_information_value
    import plotly.express as px        # optional, used in funnel + heatmap + worst-tail plots
except Exception:  # pragma: no cover
    go = None  # type: ignore
    px = None  # type: ignore

from statsmodels.stats.outliers_influence import variance_inflation_factor

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


def plot_information_value(information_value_table: pd.DataFrame, top_x: int = 30):
    """
    Plot the top_x variables by Information Value using Plotly.
    Returns the Plotly Figure.

    Requires: plotly
    """
    if go is None:
        raise ImportError("plotly is required for plot_information_value (plotly.graph_objects not found).")

    if top_x <= 0:
        raise ValueError("top_x must be > 0")

    iv_table = (
        information_value_table[["Variable", "Information Value"]]
        .sort_values("Information Value", ascending=False)
        .head(top_x)
        .copy()
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=iv_table["Variable"],
                y=iv_table["Information Value"],
                text=iv_table["Information Value"].round(3),
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        height=500,
        width=max(450, int(top_x * 40)),
        title="Information Value",
        xaxis_title="Variable",
        yaxis_title="Information Value",
        template="plotly_white",
        xaxis=dict(tickangle=90),
    )
    fig.show()
    return fig


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


# =============================================
#       worst tail analysis
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



def plot_top_worst_tail_bad_rates(
    rank_table,
    top_n=20,
    bad_metric = "BR Bal" # "BR Vol" or "BR Bal"
    use_cleaned_name=False,
    show_pct=True
):
"""
    Plot the top N features by worst-tail bad rate using Plotly.
"""

    if rank_table is None or rank_table.empty:
        raise ValueError("rank_table is empty.")

    label_col = "cleaned_name" if use_cleaned_name else "feature"

    dfp = rank_table.head(top_n).copy()
    dfp = dfp.sort_values("tail_bad_rate", ascending=True)

    # Format display values
    if show_pct:
        dfp["bad_rate_display"] = (dfp["tail_bad_rate"] * 100).round(2).astype(str) + "%"
        x_vals = dfp["tail_bad_rate"] * 100
        x_label = "Bad rate in worst tail (%)"
    else:
        dfp["bad_rate_display"] = dfp["tail_bad_rate"].round(4).astype(str)
        x_vals = dfp["tail_bad_rate"]
        x_label = "Bad rate in worst tail"

    fig = px.bar(
        dfp,
        x=x_vals,
        y=dfp[label_col].astype(str),
        orientation="h",
        text="bad_rate_display",
        hover_data={
            "tail_n": True,              # volume
            "tail_bad_n": True,          # bad volume
            "cutoff": True,
            "definition": True,
            "direction_used": True,
            "tail_side": True,
        },
        labels={
            "x": x_label,
            "y": "Feature"
        },
        title=f"Top {min(top_n, len(rank_table))} features by worst-tail {bad_metric}"
        
    )

    # Make labels look nice
    fig.update_traces(textposition="outside")

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_tickformat=".2f" if show_pct else ".3f",
        margin=dict(l=200, r=50, t=60, b=50),
        height=max(500, 35 * len(dfp))
    )

    fig.show()

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


def top_iv_correlation(
    iv_table: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_x: int,
    figsize: Tuple[int, int] = (8, 6),
    print_num: bool = True,
):
    """
    Plot the correlation heatmap for the top_x features by IV.

    Requires: plotly
    Returns the Plotly Figure.
    """
    if px is None:
        raise ImportError("plotly is required for top_iv_correlation (plotly.express not found).")

    if top_x <= 0:
        raise ValueError("top_x must be > 0")

    top_features = (
        iv_table.sort_values(by="Information Value", ascending=False)
        .head(top_x)["Variable"]
        .astype(str)
        .tolist()
    )
    top_features = [f for f in top_features if f in corr_matrix.columns]

    if len(top_features) == 0:
        raise ValueError("None of top features by IV found in correlation matrix.")

    filt = corr_matrix.loc[top_features, top_features]

    fig = px.imshow(
        filt,
        text_auto=".2f" if print_num else False,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Correlation Heatmap (top {len(top_features)} by IV)",
    )
    fig.update_layout(
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        xaxis=dict(tickangle=90),
    )
    fig.show()
    return fig


def filter_iv_corr(
    iv_table: pd.DataFrame,
    highly_correlated_pairs: pd.DataFrame,
    iv_threshold: float,
) -> List[str]:
    """
    Filter features based on IV and correlation.
    For each highly correlated pair, drop the one with lower IV.
    """
    iv_dict: Dict[str, float] = iv_table.set_index("Variable")["Information Value"].to_dict()

    drop_variables: Set[str] = set()

    for _, row in highly_correlated_pairs.iterrows():
        var1, var2 = str(row["Variable 1"]), str(row["Variable 2"])
        if var1 not in iv_dict or var2 not in iv_dict:
            continue

        if iv_dict[var1] >= iv_dict[var2]:
            drop_variables.add(var2)
        else:
            drop_variables.add(var1)

    final_variables = sorted(
        [v for v, iv in iv_dict.items() if v not in drop_variables and iv > iv_threshold],
        key=lambda v: iv_dict[v],
        reverse=True,
    )
    return final_variables


def corr_survived(
    iv_table: pd.DataFrame,
    highly_correlated_pairs: pd.DataFrame,
    num_list: Sequence[str],
    cat_list: Sequence[str],
    iv_threshold: float,
) -> Tuple[int, int, int]:
    """
    Count the number of features survived after IV & Correlation filters.

    Returns: (num_survived, cat_survived, total_survived)
    """
    final_variables = filter_iv_corr(iv_table, highly_correlated_pairs, iv_threshold)

    num_set, cat_set = set(num_list), set(cat_list)
    num_sur = sum(v in num_set for v in final_variables)
    cat_sur = sum(v in cat_set for v in final_variables)
    return num_sur, cat_sur, num_sur + cat_sur


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
#      Logic checks / distributions
# =============================================

def distribution_by_decile_logic(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    bins: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Logic checks for numeric features + plots bad-rate by deciles.

    Returns the grouped summary table (or None if skipped).
    """
    _require_columns(data, [target_variable, bad_flag, bal_variable, bad_bal_variable])

    df = data[[target_variable, bad_flag, bal_variable, bad_bal_variable]].copy()
    df = df.dropna(subset=[target_variable])

    if df[target_variable].nunique() < 3:
        return None

    baseline_br_vol = (df[bad_flag].fillna(0) > 0).mean()
    denom_bal = df[bal_variable].sum()
    baseline_br_bal = (df[bad_bal_variable].sum() / denom_bal) if denom_bal != 0 else 0.0

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    min_value, max_value, direction, definition = -np.inf, np.inf, 0.0, "N/A"

    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_\
            ar].iloc[0]
            min_value = float(row.get("Valid Min", -np.inf)) if pd.notnull(row.get("Valid Min", np.nan)) else -np.inf
            max_value = float(row.get("Valid Max", np.inf)) if pd.notnull(row.get("Valid Max", np.nan)) else np.inf
            direction = float(row.get("Direction", 0.0)) if pd.notnull(row.get("Direction", np.nan)) else 0.0
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"

    # out-of-range as "missing group"
    oob = df[(df[target_variable] < min_value) | (df[target_variable] > max_value)]
    if not oob.empty:
        br_oob = (oob[bad_flag].fillna(0) > 0).mean() * 100
        pct_oob = (len(oob) / len(df)) * 100
        over_baseline = (br_oob / (baseline_br_vol * 100) * 100) if baseline_br_vol > 0 else np.nan
        print(
            f"{target_variable}: Out-of-range Vol%={pct_oob:.2f}%, "
            f"BR(out-of-range)={br_oob:.2f}%, "
            f"over baseline={over_baseline:.2f}%"
        )

    df = df[(df[target_variable] >= min_value) & (df[target_variable] <= max_value)].copy()
    if len(df) < 5:
        return None

    try:
        df["decile"] = pd.qcut(df[target_variable], q=bins, duplicates="drop")
    except Exception:
        df["decile"] = pd.cut(df[target_variable], bins=min(bins, max(2, df[target_variable].nunique())), duplicates="drop")

    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    results = df.groupby("decile", observed=True).agg(
        Total_Volume=("is_bad", "size"),
        Bad_Volume=("is_bad", "sum"),
        Bad_Vol_Rate=("is_bad", "mean"),
        Total_Balance=(bal_variable, "sum"),
        Bad_Balance=(bad_bal_variable, "sum"),
    )
    results["Bad Vol%"] = results["Bad_Vol_Rate"] * 100
    results["Bad Bal%"] = np.where(
        results["Total_Balance"] != 0,
        (results["Bad_Balance"] / results["Total_Balance"]) * 100,
        np.nan,
    )

    # direction sanity check (optional)
    smallest_group_br = results["Bad Vol%"].iloc[0]
    largest_group_br = results["Bad Vol%"].iloc[-1]
    if direction == -1 and smallest_group_br > largest_group_br:
        print(f"{target_variable}: Counterintuitive - lower value has higher bad rate (direction=-1).")
    elif direction == 1 and smallest_group_br < largest_group_br:
        print(f"{target_variable}: Counterintuitive - larger value has higher bad rate (direction=1).")

    # --- plots ---
    results[["Bad Vol%"]].plot.bar()
    plt.axhline(y=baseline_br_vol * 100, linestyle="dotted", color="red", label="Baseline BR Vol")
    plt.ylabel("Bad Vol%")
    plt.title(f"Bad Vol% by {target_variable}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    results[["Bad Bal%"]].plot.bar()
    plt.axhline(y=baseline_br_bal * 100, linestyle="dotted", color="red", label="Baseline BR Bal")
    plt.ylabel("Bad Bal%")
    plt.title(f"Bad Bal% by {target_variable}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if definition != "N/A":
        print(f"{target_variable}: {definition}")

    return results.reset_index()


def distribution_by_group(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    max_categories: int = 30,
) -> Optional[pd.DataFrame]:
    """
    Plot volume + bad-rate and balance + bad-balance-rate for a categorical variable.

    Returns the grouped summary table (or None if skipped).
    """
    _require_columns(data, [target_variable, bad_flag, bal_variable, bad_bal_variable])

    df = data[[target_variable, bad_flag, bal_variable, bad_bal_variable]].copy()
    df[target_variable] = df[target_variable].astype("object").fillna("__MISSING__")
    df["volume"] = 1
    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    group = df.groupby(target_variable, dropna=False).agg(
        Total_Volume=("volume", "sum"),
        Bad_Volume=("is_bad", "sum"),
        Total_Balance=(bal_variable, "sum"),
        Bad_Balance=(bad_bal_variable, "sum"),
    )
    group["Bad Vol%"] = np.where(group["Total_Volume"] != 0, (group["Bad_Volume"] / group["Total_Volume"]) * 100, np.nan)
    group["Bad Bal%"] = np.where(group["Total_Balance"] != 0, (group["Bad_Balance"] / group["Total_Balance"]) * 100, np.nan)

    # Keep plots readable
    if group.shape[0] > max_categories:
        group = group.sort_values("Total_Volume", ascending=False).head(max_categories)

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_var].iloc[0]
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"
            if definition != "N/A":
                print(f"{target_variable}: {definition}")

    baseline_br_vol = df["is_bad"].mean() * 100
    denom_bal = df[bal_variable].sum()
    baseline_br_bal = (df[bad_bal_variable].sum() / denom_bal * 100) if denom_bal != 0 else 0.0

    # --- Bad Volume plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(group.index.astype(str), group["Total_Volume"])
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel("Total Volume")
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(group.index.astype(str), group["Bad Vol%"], marker="o")
    ax2.set_ylabel("Bad Vol%")
    ax2.axhline(y=baseline_br_vol, linestyle="dotted", label="Baseline")
    fig.suptitle(f"{target_variable} - Bad Volume Distribution")
    fig.tight_layout()
    plt.show()

    # --- Bad Balance plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(group.index.astype(str), group["Total_Balance"])
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel("Total Balance")
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(group.index.astype(str), group["Bad Bal%"], marker="o")
    ax2.set_ylabel("Bad Bal%")
    ax2.axhline(y=baseline_br_bal, linestyle="dotted", label="Baseline")
    fig.suptitle(f"{target_variable} - Bad Balance Distribution")
    fig.tight_layout()
    plt.show()

    return group.reset_index()


def distribution_plot_num_cat(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
):
    """
    Route to numeric/categorical distribution plot.
    """
    if _is_categorical(data[target_variable]):
        return distribution_by_group(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)
    return distribution_by_decile_logic(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)


# =============================================
#    Good vs. Bad Distribution
# =============================================

def good_bad_distribution(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    narrow: bool = False,
    bins: int = 10,
):
    """
    Plot the Good vs. Bad distribution for a single variable.

    Notes:
    - For numeric variables uses seaborn if available; otherwise falls back to matplotlib hist.
    - For categorical variables uses countplot-style bar charts if seaborn is available; otherwise matplotlib.
    """
    _require_columns(data, [target_variable, bad_flag])

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    definition = "N/A"
    valid_min = -np.inf

    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_var].iloc[0]
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"
            vm = row.get("Valid Min", np.nan)
            valid_min = float(vm) if pd.notnull(vm) else -np.inf
            if definition != "N/A":
                print(f"{target_variable}: {definition}")

    df = data[[target_variable, bad_flag]].copy()
    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    # Filter extremes
    if _is_numeric(df[target_variable]):
        if narrow:
            df = df[df[target_variable] >= valid_min].copy()
        else:
            threshold = df[target_variable].quantile(0.999)
            df = df[(df[target_variable] >= valid_min) & (df[target_variable] < threshold)].copy()

    if df.empty:
        return

    # Numeric
    if _is_numeric(df[target_variable]):
        good = df.loc[df["is_bad"] == 0, target_variable].dropna()
        bad = df.loc[df["is_bad"] == 1, target_variable].dropna()

        fig, ax1 = plt.subplots(figsize=(8, 4))
        if sns is not None:
            sns.histplot(good, kde=True, bins=bins, ax=ax1, alpha=0.5, label="Good")
            ax2 = ax1.twinx()
            sns.histplot(bad, kde=True, bins=bins, ax=ax2, alpha=0.5, label="Bad")
            ax1.set_ylabel("Good Count")
            ax2.set_ylabel("Bad Count")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax1.hist(good, bins=bins, alpha=0.5, label="Good")
            ax1.hist(bad, bins=bins, alpha=0.5, label="Bad")
            ax1.set_ylabel("Count")
            ax1.legend()

        ax1.set_xlabel(target_variable)
        ax1.set_title(f"Distribution of {target_variable} by {bad_flag}")
        fig.tight_layout()
        plt.show()
        return

    # Categorical
    df[target_variable] = df[target_variable].astype("object").fillna("__MISSING__")
    flags = sorted(df["is_bad"].unique().tolist())
    n = len(flags)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, flag in enumerate(flags):
        sub = df[df["is_bad"] == flag]
        if sns is not None:
            sns.countplot(data=sub, x=target_variable, ax=axes[i])
        else:
            counts = sub[target_variable].value_counts()
            axes[i].bar(counts.index.astype(str), counts.values)
        axes[i].set_title(f"{bad_flag}={flag} - {target_variable}")
        axes[i].tick_params(axis="x", rotation=90)
        axes[i].set_xlabel(target_variable)
        axes[i].set_ylabel("Count" if i == 0 else "")

    fig.tight_layout()
    plt.show()


# =============================================
#    Funnel plot - feature selection process
# =============================================

def funnel_feature_selection(
    original_feature: int,
    iv_survived_n: int,
    corr_survived_n: int,
    iv_corr_survived_n: int,
    logic_survived_n: int,
    expert_judgement_n: Optional[int] = None,
):
    """
    Funnel plot for feature selection stages.

    Requires: plotly
    Returns Plotly Figure.
    """
    if px is None:
        raise ImportError("plotly is required for funnel_feature_selection (plotly.express not found).")

    funnel_dictionary: Dict[str, int] = {
        "Initial": int(original_feature),
        "Information Value Filter": int(iv_survived_n),
        "Correlation Filter": int(corr_survived_n),
        "IV + Corr Filter": int(iv_corr_survived_n),
        "Logic Checks": int(logic_survived_n),
    }
    if expert_judgement_n is not None:
        funnel_dictionary["Expert Judgement"] = int(expert_judgement_n)

    funnel_data = pd.DataFrame({"Stage": list(funnel_dictionary.keys()), "Feature Number": list(funnel_dictionary.values())})

    fig = px.funnel(
        funnel_data,
        x="Feature Number",
        y="Stage",
        title="Funnel plot for Feature Selection Process",
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(funnel_dictionary.keys()),
            tickfont=dict(size=14),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=600,
    )
    fig.show()
    return fig


