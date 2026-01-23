from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd


RuleLike = Union[
    pd.Series,
    np.ndarray,
    Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]],
]


def _require_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing columns: {missing}")


def _as_mask(data: pd.DataFrame, rule: RuleLike) -> pd.Series:
    mask = rule(data) if callable(rule) else rule
    mask = pd.Series(mask, index=data.index)
    return mask.astype(bool)


def _safe_div(n: float, d: float) -> float:
    return np.nan if d == 0 else n / d


def _pct(n: float, d: float, ndigits: int = 2) -> float:
    return 0.0 if d == 0 else float(np.round((n / d) * 100, ndigits))


def _ratio(n: float, d: float, ndigits: int = 4) -> float:
    if d == 0:
        return float(np.inf) if n > 0 else np.nan
    return float(np.round(n / d, ndigits))


def new_baseline_performance_after_rule(
    data: pd.DataFrame,
    rule: RuleLike,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
) -> Dict[str, Any]:

    _require_columns(data, [bad_flag, total_bal, bad_bal], "data")
    mask = _as_mask(data, rule)

    base_vol = len(data)
    base_bad_vol = (data[bad_flag] > 0).sum()
    base_good_vol = base_vol - base_bad_vol

    base_bal = data[total_bal].sum()
    base_bad_bal = data[bad_bal].sum()

    base_br_vol_pct = _pct(base_bad_vol, base_vol)
    base_br_bal_pct = _pct(base_bad_bal, base_bal)

    removed = data.loc[mask]
    rem_vol = len(removed)
    rem_bad_vol = (removed[bad_flag] > 0).sum()
    rem_good_vol = rem_vol - rem_bad_vol

    rem_bal = removed[total_bal].sum()
    rem_bad_bal = removed[bad_bal].sum()

    kept = data.loc[~mask]
    new_vol = len(kept)
    new_bad_vol = (kept[bad_flag] > 0).sum()
    new_bal = kept[total_bal].sum()
    new_bad_bal = kept[bad_bal].sum()

    return {
        "new_total_volume": new_vol,
        "volume_decreased": rem_vol,
        "volume_decreased_pct": _pct(rem_vol, base_vol),
        "new_total_bad": new_bad_vol,
        "bad_removed": rem_bad_vol,
        "bad_removed_pct_of_baseline_bad": _pct(rem_bad_vol, base_bad_vol),
        "new_bad_vol_pct": _pct(new_bad_vol, new_vol),
        "new_total_balance": new_bal,
        "balance_decreased": rem_bal,
        "balance_decreased_pct": _pct(rem_bal, base_bal),
        "new_bad_balance": new_bad_bal,
        "bad_balance_reduced": rem_bad_bal,
        "bad_balance_reduced_pct_of_baseline_bad_balance": _pct(rem_bad_bal, base_bad_bal),
        "new_bad_bal_pct": _pct(new_bad_bal, new_bal),
        "marginal_bad_vol_pct_over_baseline_bad_vol_pct": _ratio(
            _pct(new_bad_vol, new_vol), base_br_vol_pct
        ),
        "marginal_bad_bal_pct_over_baseline_bad_bal_pct": _ratio(
            _pct(new_bad_bal, new_bal), base_br_bal_pct
        ),
        "G_to_B": _ratio(
            _safe_div(rem_bad_vol, base_bad_vol),
            _safe_div(rem_good_vol, base_good_vol)
        ),
        "baseline_bad_vol_pct": base_br_vol_pct,
        "baseline_bad_bal_pct": base_br_bal_pct,
    }


def build_new_baseline_table_from_rule_list(
    data: pd.DataFrame,
    rule_list: List[Dict[str, RuleLike]],
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    baseline_name: str = "BASELINE",
) -> pd.DataFrame:

    rows = []

    base = new_baseline_performance_after_rule(
        data,
        pd.Series(False, index=data.index),
        bad_flag,
        total_bal,
        bad_bal,
    )
    rows.append({"rule_name": baseline_name, **base})

    for item in rule_list:
        rule_name, rule = next(iter(item.items()))
        metrics = new_baseline_performance_after_rule(
            data, rule, bad_flag, total_bal, bad_bal
        )
        rows.append({"rule_name": rule_name, **metrics})

    return pd.DataFrame(rows)
