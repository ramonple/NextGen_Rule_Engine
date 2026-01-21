from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def final_feature_for_rule_construction() -> pd.DataFrame:
    """
    Create an empty DataFrame template used for defining feature rules.

    Columns:
        - variable: Name of the feature
        - Valid Min: Minimum acceptable value of the feature
        - Valid Max: Maximum acceptable value of the feature
        - Search Min: Minimum value to search in rule optimization
        - Search Max: Maximum value to search in rule optimization
        - Step: Step size for searching
        - Direction: Increase or decrease direction for search
        - Type: Data type or business indicator category

    Returns:
        Empty DataFrame with the predefined columns.
    """
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


@dataclass(frozen=True)
class RulePerformance:
    # Baseline
    total_volume: int
    total_balance: float
    total_bad_volume: int
    total_bad_balance: float
    baseline_br_vol_pct: float
    baseline_br_bal_pct: float

    # Rule / marginal
    rule_volume: int
    rule_balance: float
    rule_bad_volume: int
    rule_bad_balance: float
    rule_volume_pct: float
    rule_balance_pct: float
    rule_bad_volume_pct: float
    rule_bad_balance_pct: float
    rule_br_vol_pct: float
    rule_br_bal_pct: float

    # “GB ratio” (bad removed % / good removed %)
    gb_ratio: float  # may be nan/inf

    # New baseline after removing rule
    new_volume: int
    new_balance: float
    new_bad_volume: int
    new_bad_balance: float
    new_br_vol_pct: float
    new_br_bal_pct: float


def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing columns: {missing}")


def _safe_pct(numer: float, denom: float, *, ndigits: int = 2) -> float:
    if denom == 0:
        return 0.0
    return float(np.round((numer / denom) * 100.0, ndigits))


def _safe_ratio(numer: float, denom: float, *, ndigits: int = 4) -> float:
    if denom == 0:
        return np.inf if numer > 0 else np.nan
    return float(np.round(numer / denom, ndigits))


def rule_checking(
    data: pd.DataFrame,
    rule: pd.DataFrame,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    *,
    print_report: bool = True,
) -> RulePerformance:
    """
    Evaluate the performance impact of a single rule.

    Notes:
      - 'bad_flag' is treated as bad when > 0 (i.e., binary or count)
      - GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods)

    Returns:
      RulePerformance dataclass (also printable/serializable).
    """
    _require_columns(data, [bad_flag, bal_variable, bad_bal_variable], "data")
    _require_columns(rule, [bad_flag, bal_variable, bad_bal_variable], "rule")

    # Baseline
    total_volume = int(len(data))
    total_balance = float(data[bal_variable].sum())
    total_bad_balance = float(data[bad_bal_variable].sum())
    total_bad_volume = int((data[bad_flag] > 0).sum())
    total_good_volume = total_volume - total_bad_volume

    baseline_br_vol_pct = _safe_pct(total_bad_volume, total_volume)
    baseline_br_bal_pct = _safe_pct(total_bad_balance, total_balance)

    # Rule / marginal
    rule_volume = int(len(rule))
    rule_balance = float(rule[bal_variable].sum())
    rule_bad_balance = float(rule[bad_bal_variable].sum())
    rule_bad_volume = int((rule[bad_flag] > 0).sum())
    rule_good_volume = rule_volume - rule_bad_volume

    rule_br_vol_pct = _safe_pct(rule_bad_volume, rule_volume)
    rule_br_bal_pct = _safe_pct(rule_bad_balance, rule_balance)

    rule_volume_pct = _safe_pct(rule_volume, total_volume)
    rule_balance_pct = _safe_pct(rule_balance, total_balance)

    rule_bad_volume_pct = _safe_pct(rule_bad_volume, total_bad_volume) if total_bad_volume else 0.0
    rule_bad_balance_pct = _safe_pct(rule_bad_balance, total_bad_balance) if total_bad_balance else 0.0

    bad_removed_share = _safe_ratio(rule_bad_volume, total_bad_volume, ndigits=6) if total_bad_volume else 0.0
    good_removed_share = _safe_ratio(rule_good_volume, total_good_volume, ndigits=6) if total_good_volume else np.nan
    gb_ratio = _safe_ratio(bad_removed_share, good_removed_share, ndigits=4)

    # New baseline after removing rule population
    new_volume = total_volume - rule_volume
    new_balance = total_balance - rule_balance
    new_bad_volume = total_bad_volume - rule_bad_volume
    new_bad_balance = total_bad_balance - rule_bad_balance

    new_br_vol_pct = _safe_pct(new_bad_volume, new_volume)
    new_br_bal_pct = _safe_pct(new_bad_balance, new_balance)

    perf = RulePerformance(
        total_volume=total_volume,
        total_balance=total_balance,
        total_bad_volume=total_bad_volume,
        total_bad_balance=total_bad_balance,
        baseline_br_vol_pct=baseline_br_vol_pct,
        baseline_br_bal_pct=baseline_br_bal_pct,
        rule_volume=rule_volume,
        rule_balance=rule_balance,
        rule_bad_volume=rule_bad_volume,
        rule_bad_balance=rule_bad_balance,
        rule_volume_pct=rule_volume_pct,
        rule_balance_pct=rule_balance_pct,
        rule_bad_volume_pct=rule_bad_volume_pct,
        rule_bad_balance_pct=rule_bad_balance_pct,
        rule_br_vol_pct=rule_br_vol_pct,
        rule_br_bal_pct=rule_br_bal_pct,
        gb_ratio=gb_ratio,
        new_volume=new_volume,
        new_balance=new_balance,
        new_bad_volume=new_bad_volume,
        new_bad_balance=new_bad_balance,
        new_br_vol_pct=new_br_vol_pct,
        new_br_bal_pct=new_br_bal_pct,
    )

    if print_report:
        _print_rule_report(perf)

    return perf


def _print_rule_report(perf: RulePerformance) -> None:
    print("\n==================== BASELINE PERFORMANCE ====================")
    print(f"Total Volume      : {perf.total_volume}")
    print(f"Total Balance     : {perf.total_balance:,.2f}")
    print(f"Total Bad Volume  : {perf.total_bad_volume}")
    print(f"Total Bad Balance : {perf.total_bad_balance:,.2f}")
    print(f"BR (Vol)          : {perf.baseline_br_vol_pct}%")
    print(f"BR (Bal)          : {perf.baseline_br_bal_pct}%")

    print("\n==================== RULE IMPACT (MARGINAL) ====================")
    print(f"Rule Volume       : {perf.rule_volume} ({perf.rule_volume_pct}%)")
    print(f"Rule Balance      : {perf.rule_balance:,.2f} ({perf.rule_balance_pct}%)")
    print(f"Rule Bad Volume   : {perf.rule_bad_volume} ({perf.rule_bad_volume_pct}% of bads)")
    print(f"Rule Bad Balance  : {perf.rule_bad_balance:,.2f} ({perf.rule_bad_balance_pct}% of bad bal)")
    print(f"Rule BR (Vol)     : {perf.rule_br_vol_pct}% (Baseline {perf.baseline_br_vol_pct}%)")
    print(f"Rule BR (Bal)     : {perf.rule_br_bal_pct}% (Baseline {perf.baseline_br_bal_pct}%)")
    print(f"GB ratio          : {perf.gb_ratio}")

    print("\n==================== NEW BASELINE AFTER RULE ====================")
    print(f"New Volume        : {perf.new_volume}")
    print(f"New Balance       : {perf.new_balance:,.2f} ({perf.new_balance/1e6:,.2f}M)")
    print(f"New Bad Volume    : {perf.new_bad_volume}")
    print(f"New Bad Balance   : {perf.new_bad_balance:,.2f}")
    print(f"New BR (Vol)      : {perf.new_br_vol_pct}%")
    print(f"New BR (Bal)      : {perf.new_br_bal_pct}%")
    print("===============================================================\n")


def combine_checking_gb_ratio(
    data: pd.DataFrame,
    min_bads: int,
    rule: pd.DataFrame,
    bad_flag: str,
) -> float:
    """
    Compute GB ratio for a rule.
    GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods)

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

    Returns np.nan if:
      - rule removes fewer than `min_bads` bads
      - baseline balance is 0 or baseline rate is 0
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
