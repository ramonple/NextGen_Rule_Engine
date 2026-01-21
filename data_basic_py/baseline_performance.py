"""
baseline_performance_improved.py

Lightweight utilities for baseline portfolio / dataset performance summaries.

Key upgrades vs original:
- Returns structured outputs (dict / DataFrame) instead of only printing
- Safer handling of zero totals, missing columns, NaNs, and non-binary bad flags
- Optional sorting, totals row, share-of-total columns, and cumulative metrics
- Type hints + docstrings + minimal dependencies (numpy, pandas)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Literal

import numpy as np
import pandas as pd


RateMode = Literal["percent", "ratio"]


@dataclass(frozen=True)
class OverviewResult:
    """Container for base_overview results."""
    total_vol: int
    bad_vol: int
    bad_rate_vol: float  # percent or ratio depending on rate_mode
    total_bal: float
    bad_bal: float
    bad_rate_bal: float  # percent or ratio depending on rate_mode

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_vol": self.total_vol,
            "bad_vol": self.bad_vol,
            "bad_rate_vol": self.bad_rate_vol,
            "total_bal": self.total_bal,
            "bad_bal": self.bad_bal,
            "bad_rate_bal": self.bad_rate_bal,
        }


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _to_bad_indicator(
    s: pd.Series,
    *,
    bad_threshold: float = 0.0,
    assume_binary: bool = False,
) -> pd.Series:
    """
    Convert a 'bad' column to a 0/1 indicator.
    - If assume_binary=True, values are coerced to {0,1} by (s > 0).
    - Otherwise, we treat numeric/boolean as bad if > bad_threshold.
    """
    if assume_binary:
        return (pd.to_numeric(s, errors="coerce").fillna(0) > bad_threshold).astype(int)

    # If already boolean-like, preserve
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    # Numeric conversion and thresholding
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    return (x > bad_threshold).astype(int)


def _safe_rate(numer: Union[int, float], denom: Union[int, float], *, rate_mode: RateMode) -> float:
    if denom in (0, 0.0) or (isinstance(denom, float) and not np.isfinite(denom)):
        return 0.0
    val = float(numer) / float(denom)
    return val * 100.0 if rate_mode == "percent" else val


def base_overview(
    data: pd.DataFrame,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: Optional[str] = None,
    *,
    bad_threshold: float = 0.0,
    assume_bad_flag_binary: bool = False,
    rate_mode: RateMode = "percent",
    print_summary: bool = True,
) -> OverviewResult:
    """
    Get a baseline overview of dataset "bad" rates by volume and balance.

    Parameters
    ----------
    data : DataFrame
    bad_flag : str
        Column indicating "bad". Will be converted to 0/1 via > bad_threshold.
    bal_variable : str
        Column containing total balance / exposure.
    bad_bal_variable : str, optional
        If provided, this column is summed as bad balance. If not provided,
        bad balance is computed as bal_variable for bad rows.
    bad_threshold : float
        Threshold for determining bad rows (bad_flag > bad_threshold).
    assume_bad_flag_binary : bool
        If True, coercion uses (bad_flag > bad_threshold) regardless of dtype.
    rate_mode : {"percent","ratio"}
        Whether to return rates as percent (0-100) or ratio (0-1).
    print_summary : bool
        If True, prints a short text summary.

    Returns
    -------
    OverviewResult
    """
    if bad_bal_variable is None:
        _require_columns(data, [bad_flag, bal_variable])
    else:
        _require_columns(data, [bad_flag, bal_variable, bad_bal_variable])

    bad_ind = _to_bad_indicator(data[bad_flag], bad_threshold=bad_threshold, assume_binary=assume_bad_flag_binary)

    total_vol = int(len(data))
    bad_vol = int(bad_ind.sum())

    total_bal = float(pd.to_numeric(data[bal_variable], errors="coerce").fillna(0).sum())
    if bad_bal_variable is None:
        bad_bal = float(pd.to_numeric(data[bal_variable], errors="coerce").fillna(0).where(bad_ind.eq(1), 0).sum())
    else:
        bad_bal = float(pd.to_numeric(data[bad_bal_variable], errors="coerce").fillna(0).sum())

    bad_rate_vol = _safe_rate(bad_vol, total_vol, rate_mode=rate_mode)
    bad_rate_bal = _safe_rate(bad_bal, total_bal, rate_mode=rate_mode)

    res = OverviewResult(
        total_vol=total_vol,
        bad_vol=bad_vol,
        bad_rate_vol=bad_rate_vol,
        total_bal=total_bal,
        bad_bal=bad_bal,
        bad_rate_bal=bad_rate_bal,
    )

    if print_summary:
        if rate_mode == "percent":
            print(f"Total Volume: {total_vol:,} | Bad Volume: {bad_vol:,} | Bad Rate (Vol): {bad_rate_vol:.2f}%")
            print(f"Total Balance: {total_bal:,.2f} | Bad Balance: {bad_bal:,.2f} | Bad Rate (Bal): {bad_rate_bal:.2f}%")
        else:
            print(f"Total Volume: {total_vol:,} | Bad Volume: {bad_vol:,} | Bad Rate (Vol): {bad_rate_vol:.6f}")
            print(f"Total Balance: {total_bal:,.2f} | Bad Balance: {bad_bal:,.2f} | Bad Rate (Bal): {bad_rate_bal:.6f}")

    return res


def groupby_target_variable(
    data: pd.DataFrame,
    col: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: Optional[str] = None,
    *,
    bad_threshold: float = 0.0,
    assume_bad_flag_binary: bool = False,
    rate_mode: RateMode = "percent",
    sort_by: Optional[str] = None,
    ascending: bool = False,
    add_shares: bool = True,
    add_cumulative: bool = False,
    add_totals_row: bool = False,
    dropna_groups: bool = False,
) -> pd.DataFrame:
    """
    Group-by performance summary for a categorical (or discrete) variable.

    Output columns:
      - Total_vol, Bad_vol, Total_bal, Bad_bal
      - BR Vol, BR Bal (percent or ratio)
      - Optional: Vol_share, Bal_share, BadVol_share, BadBal_share
      - Optional cumulative columns if add_cumulative=True

    Notes
    -----
    - If bad_bal_variable is None, Bad_bal is computed as bal_variable for bad rows.
    - Uses safe divisions: if Total_vol or Total_bal is 0, rates become 0.

    Parameters
    ----------
    dropna_groups : bool
        If True, exclude NaN group values. If False, NaNs form their own group.

    Returns
    -------
    DataFrame
    """
    if bad_bal_variable is None:
        _require_columns(data, [col, bad_flag, bal_variable])
    else:
        _require_columns(data, [col, bad_flag, bal_variable, bad_bal_variable])

    df = data.copy()

    bad_ind = _to_bad_indicator(df[bad_flag], bad_threshold=bad_threshold, assume_binary=assume_bad_flag_binary)
    df["_bad_ind"] = bad_ind

    df["_bal"] = pd.to_numeric(df[bal_variable], errors="coerce").fillna(0)

    if bad_bal_variable is None:
        df["_bad_bal"] = df["_bal"].where(df["_bad_ind"].eq(1), 0.0)
    else:
        df["_bad_bal"] = pd.to_numeric(df[bad_bal_variable], errors="coerce").fillna(0)

    gb = df.groupby(col, dropna=dropna_groups).agg(
        Total_vol=("_bad_ind", "count"),
        Bad_vol=("_bad_ind", "sum"),
        Total_bal=("_bal", "sum"),
        Bad_bal=("_bad_bal", "sum"),
    )

    # Rates
    gb["BR Vol"] = [
        _safe_rate(b, t, rate_mode=rate_mode) for b, t in zip(gb["Bad_vol"].to_numpy(), gb["Total_vol"].to_numpy())
    ]
    gb["BR Bal"] = [
        _safe_rate(b, t, rate_mode=rate_mode) for b, t in zip(gb["Bad_bal"].to_numpy(), gb["Total_bal"].to_numpy())
    ]

    # Shares
    if add_shares:
        tot_v = float(gb["Total_vol"].sum())
        tot_bv = float(gb["Bad_vol"].sum())
        tot_bal = float(gb["Total_bal"].sum())
        tot_bbal = float(gb["Bad_bal"].sum())

        gb["Vol_share"] = [_safe_rate(v, tot_v, rate_mode="ratio") for v in gb["Total_vol"].to_numpy()]
        gb["BadVol_share"] = [_safe_rate(v, tot_bv, rate_mode="ratio") for v in gb["Bad_vol"].to_numpy()]
        gb["Bal_share"] = [_safe_rate(v, tot_bal, rate_mode="ratio") for v in gb["Total_bal"].to_numpy()]
        gb["BadBal_share"] = [_safe_rate(v, tot_bbal, rate_mode="ratio") for v in gb["Bad_bal"].to_numpy()]

    # Sorting
    if sort_by is not None:
        if sort_by not in gb.columns:
            raise KeyError(f"sort_by='{sort_by}' not in columns: {list(gb.columns)}")
        gb = gb.sort_values(sort_by, ascending=ascending)
    else:
        gb = gb.sort_index()

    # Cumulative metrics (typically after sorting by risk or volume)
    if add_cumulative:
        gb["Cum_Total_vol"] = gb["Total_vol"].cumsum()
        gb["Cum_Bad_vol"] = gb["Bad_vol"].cumsum()
        gb["Cum_Total_bal"] = gb["Total_bal"].cumsum()
        gb["Cum_Bad_bal"] = gb["Bad_bal"].cumsum()
        gb["Cum_BR_Vol"] = [
            _safe_rate(b, t, rate_mode=rate_mode)
            for b, t in zip(gb["Cum_Bad_vol"].to_numpy(), gb["Cum_Total_vol"].to_numpy())
        ]
        gb["Cum_BR_Bal"] = [
            _safe_rate(b, t, rate_mode=rate_mode)
            for b, t in zip(gb["Cum_Bad_bal"].to_numpy(), gb["Cum_Total_bal"].to_numpy())
        ]

    if add_totals_row:
        totals = pd.Series(
            {
                "Total_vol": gb["Total_vol"].sum(),
                "Bad_vol": gb["Bad_vol"].sum(),
                "Total_bal": gb["Total_bal"].sum(),
                "Bad_bal": gb["Bad_bal"].sum(),
            },
            name="__TOTAL__",
        )
        totals["BR Vol"] = _safe_rate(totals["Bad_vol"], totals["Total_vol"], rate_mode=rate_mode)
        totals["BR Bal"] = _safe_rate(totals["Bad_bal"], totals["Total_bal"], rate_mode=rate_mode)

        if add_shares:
            # totals shares are 1 by construction (or 0 if denominator 0)
            totals["Vol_share"] = 1.0 if gb["Total_vol"].sum() else 0.0
            totals["BadVol_share"] = 1.0 if gb["Bad_vol"].sum() else 0.0
            totals["Bal_share"] = 1.0 if gb["Total_bal"].sum() else 0.0
            totals["BadBal_share"] = 1.0 if gb["Bad_bal"].sum() else 0.0

        gb = pd.concat([gb, totals.to_frame().T], axis=0)

    # Cleanup temp cols in local df (gb is detached)
    return gb


def multi_groupby(
    data: pd.DataFrame,
    cols: list[str],
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: Optional[str] = None,
    *,
    bad_threshold: float = 0.0,
    assume_bad_flag_binary: bool = False,
    rate_mode: RateMode = "percent",
    **groupby_kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience: run groupby_target_variable for multiple columns and return a dict.

    Example
    -------
    summaries = multi_groupby(df, ["region","channel"], "bad", "balance")
    summaries["region"]
    """
    out: Dict[str, pd.DataFrame] = {}
    for c in cols:
        out[c] = groupby_target_variable(
            data=data,
            col=c,
            bad_flag=bad_flag,
            bal_variable=bal_variable,
            bad_bal_variable=bad_bal_variable,
            bad_threshold=bad_threshold,
            assume_bad_flag_binary=assume_bad_flag_binary,
            rate_mode=rate_mode,
            **groupby_kwargs,
        )
    return out
