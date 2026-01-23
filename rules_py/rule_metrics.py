import pandas as pd
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# =============================================================================
# Baseline caching (speed)
# =============================================================================

def compute_baseline_stats(
    data: pd.DataFrame,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
) -> Dict[str, float]:
    """
    Compute baseline totals/rates once and reuse across many rule evaluations.
    """
    for c in (bad_flag, total_bal, bad_bal):
        if c not in data.columns:
            raise KeyError(f"data missing required column: {c}")

    total_vol = int(len(data))
    total_bad_vol = int((data[bad_flag] > 0).sum())
    total_good_vol = total_vol - total_bad_vol

    total_balance = float(data[total_bal].sum())
    total_bad_balance = float(data[bad_bal].sum())

    baseline_bad_vol_rate = (total_bad_vol / total_vol) if total_vol else 0.0
    baseline_bad_bal_rate = (total_bad_balance / total_balance) if total_balance else np.nan

    return {
        "total_vol": float(total_vol),
        "total_bad_vol": float(total_bad_vol),
        "total_good_vol": float(total_good_vol),
        "total_balance": total_balance,
        "total_bad_balance": total_bad_balance,
        "baseline_bad_vol_rate": baseline_bad_vol_rate,
        "baseline_bad_bal_rate": baseline_bad_bal_rate,
    }


# =============================================================================
# Rule evaluation metrics (fast, reusable)
# =============================================================================

RuleLike = Union[
    pd.DataFrame,                   # legacy: removed population df
    pd.Series, np.ndarray,           # boolean mask aligned to data.index
    Callable[[pd.DataFrame], Any],   # callable(df)->mask
]

def _resolve_mask(data: pd.DataFrame, rule: RuleLike) -> pd.Series:
    """
    Normalize rule to a boolean mask aligned to data.index.
    """
    if isinstance(rule, pd.DataFrame):
        return pd.Series(data.index.isin(rule.index), index=data.index, dtype=bool)

    m = rule(data) if callable(rule) else rule
    return pd.Series(m, index=data.index).astype(bool)


def combine_checking_gb_ratio(
    data: pd.DataFrame,
    min_bads: int,
    rule: RuleLike,
    bad_flag: str,
    *,
    baseline_stats: Optional[Dict[str, float]] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods)

    Speed:
      - pass baseline_stats from compute_baseline_stats() to avoid recomputation

    Returns:
      - scalar (default) or a dict (return_details=True) with marginal-set fields.
    """
    if bad_flag not in data.columns:
        raise KeyError(f"data missing column: {bad_flag}")

    if baseline_stats is None:
        # minimal baseline for GB
        total_vol = int(len(data))
        total_bad = int((data[bad_flag] > 0).sum())
        total_good = total_vol - total_bad
    else:
        total_bad = int(baseline_stats["total_bad_vol"])
        total_good = int(baseline_stats["total_good_vol"])

    mask = _resolve_mask(data, rule)

    rem_vol = int(mask.sum())
    rem_bad = int(((data[bad_flag] > 0) & mask).sum())
    rem_good = rem_vol - rem_bad

    if rem_bad < int(min_bads) or total_bad <= 0 or total_good <= 0:
        metric = np.nan
    else:
        bad_removed_share = rem_bad / total_bad
        good_removed_share = rem_good / total_good
        metric = np.inf if good_removed_share == 0 else float(np.round(bad_removed_share / good_removed_share, 4))

    if not return_details:
        return metric

    return {
        "volume_removed": rem_vol,
        "bad_vol_detected": rem_bad,
        "total_balance_removed": np.nan,
        "bad_balance_removed": np.nan,
        "metric": metric,
        "metric_name": "G_to_B",
    }


def combine_checking_bal_br_times(
    data: pd.DataFrame,
    min_bads: int,
    rule: RuleLike,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    *,
    baseline_stats: Optional[Dict[str, float]] = None,
    mode: str = "removed",  # "removed" (marginal-set rate) OR "new_baseline" (remaining portfolio rate)
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    BR balance times metric.

    mode="removed":
        (removed_bad_bal / removed_bal) / (baseline_bad_bal / baseline_bal)
        -> >1 is better (removed set is worse than baseline)

    mode="new_baseline":
        (new_bad_bal / new_bal) / (baseline_bad_bal / baseline_bal)
        -> <1 is better (new baseline improves)

    Speed:
      - pass baseline_stats to avoid recomputing baseline rates and totals
      - avoids slicing full DF except for masked sums
    """
    for c in (bad_flag, bal_variable, bad_bal_variable):
        if c not in data.columns:
            raise KeyError(f"data missing column: {c}")

    if baseline_stats is None:
        baseline_stats = compute_baseline_stats(data, bad_flag, bal_variable, bad_bal_variable)

    baseline_rate = float(baseline_stats["baseline_bad_bal_rate"])
    if not np.isfinite(baseline_rate) or baseline_rate == 0:
        metric = np.nan
        if not return_details:
            return metric
        return {
            "volume_removed": 0,
            "bad_vol_detected": 0,
            "total_balance_removed": 0.0,
            "bad_balance_removed": 0.0,
            "metric": metric,
            "metric_name": "BR_Bal_Times",
            "mode": mode,
        }

    mask = _resolve_mask(data, rule)

    rem_vol = int(mask.sum())
    rem_bad = int(((data[bad_flag] > 0) & mask).sum())
    rem_bal = float(data.loc[mask, bal_variable].sum())
    rem_bad_bal = float(data.loc[mask, bad_bal_variable].sum())

    if rem_bad < int(min_bads):
        metric = np.nan
    else:
        if mode == "removed":
            rem_rate = (rem_bad_bal / rem_bal) if rem_bal else np.nan
            metric = float(np.round(rem_rate / baseline_rate, 4)) if np.isfinite(rem_rate) else np.nan
        elif mode == "new_baseline":
            new_bal = float(baseline_stats["total_balance"] - rem_bal)
            new_bad_bal = float(baseline_stats["total_bad_balance"] - rem_bad_bal)
            new_rate = (new_bad_bal / new_bal) if new_bal else np.nan
            metric = float(np.round(new_rate / baseline_rate, 4)) if np.isfinite(new_rate) else np.nan
        else:
            raise ValueError("mode must be 'removed' or 'new_baseline'")

    if not return_details:
        return metric

    return {
        "volume_removed": rem_vol,
        "bad_vol_detected": rem_bad,
        "total_balance_removed": rem_bal,
        "bad_balance_removed": rem_bad_bal,
        "metric": metric,
        "metric_name": "BR_Bal_Times",
        "mode": mode,
    }


def rule_metric_summary(
    data: pd.DataFrame,
    rule: RuleLike,
    *,
    metric_name: str,
    min_bads: int,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    baseline_stats: Optional[Dict[str, float]] = None,
    bal_times_mode: str = "removed",
) -> Dict[str, Any]:
    """
    Unified interface returning marginal-set fields + the chosen metric.
    """
    m = metric_name.strip().lower()
    if m in ("g_to_b", "gb", "gb_ratio", "g:b"):
        return combine_checking_gb_ratio(
            data=data,
            min_bads=min_bads,
            rule=rule,
            bad_flag=bad_flag,
            baseline_stats=baseline_stats,
            return_details=True,
        )

    if m in ("br_bal_times", "bal_br_times", "br balance times", "br_baltime", "br_bal_time"):
        return combine_checking_bal_br_times(
            data=data,
            min_bads=min_bads,
            rule=rule,
            bad_flag=bad_flag,
            bal_variable=total_bal,
            bad_bal_variable=bad_bal,
            baseline_stats=baseline_stats,
            mode=bal_times_mode,
            return_details=True,
        )

    raise ValueError(f"Unknown metric_name: {metric_name}")