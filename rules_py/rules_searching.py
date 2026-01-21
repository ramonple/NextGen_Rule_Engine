"""
rules_searching.py

Core utilities for *searching* and *scoring* rule candidates.

Design goals:
- Fast, side‑effect free scoring for large grid / heuristic searches
- Consistent definitions of "rule-triggered" population (rows removed/declined)
- Metrics returned as plain dicts or DataFrames for easy ranking / filtering
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# A rule can be:
# - a boolean mask aligned to data.index
# - a numpy boolean array of length len(data)
# - a callable that returns a mask when passed the dataframe
RuleLike = Union[
    pd.Series,
    np.ndarray,
    Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]],
]


# ---------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------
def final_feature_for_rule_construction() -> pd.DataFrame:
    """Return an empty template DataFrame used for defining/searching feature rules."""
    columns = [
        "variable",
        "Valid Min",
        "Valid Max",
        "Search Min",
        "Search Max",
        "Step",
        "Direction",
        "Type",
    ]
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: Sequence[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing columns: {missing}")


def _as_mask(data: pd.DataFrame, rule: RuleLike) -> pd.Series:
    """Coerce rule into a boolean Series aligned to data.index."""
    mask = rule(data) if callable(rule) else rule
    mask = pd.Series(mask, index=data.index)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return mask


def _safe_div(n: float, d: float) -> float:
    return np.nan if d == 0 else (n / d)


def _pct(n: float, d: float, ndigits: int = 2) -> float:
    if d == 0:
        return 0.0
    return float(np.round((n / d) * 100.0, ndigits))


def _ratio(n: float, d: float, ndigits: int = 4) -> float:
    """Return a ratio with sensible inf/nan handling."""
    if d == 0:
        return float(np.inf) if n > 0 else float(np.nan)
    return float(np.round(n / d, ndigits))


# ---------------------------------------------------------------------
# Core: New baseline performance after applying a rule
# ---------------------------------------------------------------------
def new_baseline_performance_after_rule(
    data: pd.DataFrame,
    rule: RuleLike,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
) -> Dict[str, Any]:
    """
    Compute *new baseline* performance after implementing a rule.

    Interpretation:
      - rule evaluates to a boolean mask of rows to REMOVE/DECLINE
        True  => removed by the rule
        False => kept in the new baseline

    Returns a dict with the following keys (stable contract):

      New baseline:
        - new_total_volume
        - new_total_bad
        - new_bad_vol_pct
        - new_total_balance
        - new_bad_balance
        - new_bad_bal_pct

      Removed by rule:
        - volume_decreased
        - volume_decreased_pct
        - bad_removed
        - bad_removed_pct_of_baseline_bad
        - balance_decreased
        - balance_decreased_pct
        - bad_balance_reduced
        - bad_balance_reduced_pct_of_baseline_bad_balance

      Marginal comparisons (ratios vs baseline rates):
        - marginal_bad_vol_pct_over_baseline_bad_vol_pct
        - marginal_bad_bal_pct_over_baseline_bad_bal_pct

      G:B:
        - G_to_B   (bad removed share / good removed share)

      Baseline reference (handy for ranking/debug):
        - baseline_bad_vol_pct
        - baseline_bad_bal_pct
    """
    _require_columns(data, [bad_flag, total_bal, bad_bal], "data")

    mask = _as_mask(data, rule)

    # Baseline totals
    base_vol = int(len(data))
    base_bad_vol = int((data[bad_flag] > 0).sum())
    base_good_vol = base_vol - base_bad_vol

    base_bal = float(data[total_bal].sum())
    base_bad_bal = float(data[bad_bal].sum())

    base_br_vol_pct = _pct(base_bad_vol, base_vol)
    base_br_bal_pct = _pct(base_bad_bal, base_bal)

    # Removed population (rule-triggered)
    removed = data.loc[mask]
    rem_vol = int(len(removed))
    rem_bad_vol = int((removed[bad_flag] > 0).sum())
    rem_good_vol = rem_vol - rem_bad_vol

    rem_bal = float(removed[total_bal].sum())
    rem_bad_bal = float(removed[bad_bal].sum())

    # New baseline after removing
    kept = data.loc[~mask]
    new_vol = int(len(kept))
    new_bad_vol = int((kept[bad_flag] > 0).sum())
    new_bal = float(kept[total_bal].sum())
    new_bad_bal = float(kept[bad_bal].sum())

    new_br_vol_pct = _pct(new_bad_vol, new_vol)
    new_br_bal_pct = _pct(new_bad_bal, new_bal)

    # Removed shares (%)
    vol_removed_pct = _pct(rem_vol, base_vol)
    bal_removed_pct = _pct(rem_bal, base_bal)

    bad_removed_pct = _pct(rem_bad_vol, base_bad_vol) if base_bad_vol else 0.0
    bad_bal_removed_pct = _pct(rem_bad_bal, base_bad_bal) if base_bad_bal else 0.0

    # Ratios of rates vs baseline (requested "over")
    mar_bad_vol_over_base = _ratio(new_br_vol_pct, base_br_vol_pct)
    mar_bad_bal_over_base = _ratio(new_br_bal_pct, base_br_bal_pct)

    # G:B = (bad removed share) / (good removed share)
    bad_removed_share = _safe_div(rem_bad_vol, base_bad_vol) if base_bad_vol else np.nan
    good_removed_share = _safe_div(rem_good_vol, base_good_vol) if base_good_vol else np.nan
    gb = _ratio(bad_removed_share, good_removed_share)

    return {
        # New baseline
        "new_total_volume": new_vol,
        "volume_decreased": rem_vol,
        "volume_decreased_pct": vol_removed_pct,
        "new_total_bad": new_bad_vol,
        "bad_removed": rem_bad_vol,
        "bad_removed_pct_of_baseline_bad": bad_removed_pct,
        "new_bad_vol_pct": new_br_vol_pct,
        "new_total_balance": new_bal,
        "balance_decreased": rem_bal,
        "balance_decreased_pct": bal_removed_pct,
        "new_bad_balance": new_bad_bal,
        "bad_balance_reduced": rem_bad_bal,
        "bad_balance_reduced_pct_of_baseline_bad_balance": bad_bal_removed_pct,
        "new_bad_bal_pct": new_br_bal_pct,
        # Marginal ratios
        "marginal_bad_vol_pct_over_baseline_bad_vol_pct": mar_bad_vol_over_base,
        "marginal_bad_bal_pct_over_baseline_bad_bal_pct": mar_bad_bal_over_base,
        # G:B
        "G_to_B": gb,
        # Baseline reference
        "baseline_bad_vol_pct": base_br_vol_pct,
        "baseline_bad_bal_pct": base_br_bal_pct,
    }


def build_new_baseline_table_from_rule_list(
    data: pd.DataFrame,
    rule_list: List[Dict[str, RuleLike]],
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    *,
    baseline_name: str = "BASELINE",
) -> pd.DataFrame:
    """
    Create a comparison table for multiple rules (first row = original baseline).

    Input format:
        rule_list = [
            {"Rule 1": rule_1_mask},
            {"Rule 2": rule_2_mask},
            {"Rule 1 or 2": rule_1_mask | rule_2_mask},
            {"Rule 1 and 2": rule_1_mask & rule_2_mask},
        ]

    Each mask must be aligned to data.index (or be a callable returning such a mask).

    Returns:
        DataFrame with a 'rule_name' column + the exact keys produced by
        new_baseline_performance_after_rule().
    """
    rows: List[Dict[str, Any]] = []

    # Baseline row = remove nothing
    base = new_baseline_performance_after_rule(
        data=data,
        rule=pd.Series(False, index=data.index),
        bad_flag=bad_flag,
        total_bal=total_bal,
        bad_bal=bad_bal,
    )
    rows.append({"rule_name": baseline_name, **base})

    for item in rule_list:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                "Each element of rule_list must be a dict with exactly one entry: {'Rule name': rule}"
            )
        rule_name, rule = next(iter(item.items()))
        metrics = new_baseline_performance_after_rule(
            data=data,
            rule=rule,
            bad_flag=bad_flag,
            total_bal=total_bal,
            bad_bal=bad_bal,
        )
        rows.append({"rule_name": str(rule_name), **metrics})

    df = pd.DataFrame(rows)
    cols = ["rule_name"] + [c for c in df.columns if c != "rule_name"]
    return df[cols]


# ---------------------------------------------------------------------
# Legacy / search utilities (kept for backwards compatibility)
# ---------------------------------------------------------------------
def combine_checking_gb_ratio(
    data: pd.DataFrame,
    min_bads: int,
    rule: pd.DataFrame,
    bad_flag: str,
) -> float:
    """
    GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods).

    Returns:
      - np.nan if rule removes fewer than `min_bads` bads, or if baseline has 0 bads.
      - np.inf if rule removes 0 goods but removes some bads.
    """
    _require_columns(data, [bad_flag], "data")
    _require_columns(rule, [bad_flag], "rule")

    total_volume = int(len(data))
    total_bad_volume = int((data[bad_flag] > 0).sum())
    total_good_volume = total_volume - total_bad_volume

    if total_volume == 0 or total_bad_volume == 0 or total_good_volume < 0:
        return np.nan

    rule_bad_volume = int((rule[bad_flag] > 0).sum())
    if rule_bad_volume < int(min_bads):
        return np.nan

    rule_volume = int(len(rule))
    rule_good_volume = rule_volume - rule_bad_volume

    bad_removed_share = rule_bad_volume / total_bad_volume
    good_removed_share = (rule_good_volume / total_good_volume) if total_good_volume > 0 else np.nan

    if good_removed_share == 0:
        return np.inf

    return float(np.round(bad_removed_share / good_removed_share, 4))


def combine_checking_bal_br_times(
    data: pd.DataFrame,
    min_bads: int,
    rule: pd.DataFrame,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
) -> float:
    """
    Ratio = (Rule bad balance rate) / (Baseline bad balance rate)
          = (rule_bad_balance / rule_balance) / (total_bad_balance / total_balance)
    """
    _require_columns(data, [bad_flag, bal_variable, bad_bal_variable], "data")
    _require_columns(rule, [bad_flag, bal_variable, bad_bal_variable], "rule")

    total_balance = float(data[bal_variable].sum())
    total_bad_balance = float(data[bad_bal_variable].sum())
    if total_balance == 0:
        return np.nan

    baseline_rate = total_bad_balance / total_balance
    if baseline_rate == 0:
        return np.nan

    rule_bad_volume = int((rule[bad_flag] > 0).sum())
    if rule_bad_volume < int(min_bads):
        return np.nan

    rule_balance = float(rule[bal_variable].sum())
    rule_bad_balance = float(rule[bad_bal_variable].sum())
    if rule_balance == 0:
        return np.nan

    rule_rate = rule_bad_balance / rule_balance
    return float(np.round(rule_rate / baseline_rate, 4))
