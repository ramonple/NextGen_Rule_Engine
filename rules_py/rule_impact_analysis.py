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

def _resolve_mask(data: pd.DataFrame, rule: RuleLike) -> pd.Series:
    """
    Normalize `rule` into a boolean mask aligned to `data.index`.

    Supported rule formats:
      1) pd.DataFrame: treated as marginal population (rows to remove)
      2) pd.Series / np.ndarray: boolean mask aligned to data.index
      3) callable: function(df) -> boolean mask
    """
    if isinstance(rule, pd.DataFrame):
        return pd.Series(data.index.isin(rule.index), index=data.index, dtype=bool)

    m = rule(data) if callable(rule) else rule
    return pd.Series(m, index=data.index).astype(bool)

# ==============================================================
def rule_checking(
    data: pd.DataFrame,
    rule: RuleLike,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    *,
    print_details: bool = True,
    return_details: bool = True,
    baseline_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the performance impact of a single rule.

    IMPORTANT: `rule` is NOT required to be `data[data['x'] > 1]`.
    Recommended input is a boolean mask or callable returning a mask.

    Examples
    --------
    mask = data["x"] > 1
    rule_checking(data, mask, bad_flag="bad", bal_variable="bal", bad_bal_variable="bad_bal")

    rule_fn = lambda df: (df["x"] > 10) & (df["y"] < 20)
    rule_checking(data, rule_fn, ...)

    legacy:
    rule_df = data.loc[data["x"] > 1]
    rule_checking(data, rule_df, ...)
    """
    # ---- validate columns ----
    for c in (bad_flag, bal_variable, bad_bal_variable):
        if c not in data.columns:
            raise KeyError(f"data missing required column: {c}")

    # ---- baseline stats (optional cache) ----
    if baseline_stats is None:
        total_volume = int(len(data))
        total_balance = float(data[bal_variable].sum())
        total_bad_balance = float(data[bad_bal_variable].sum())
        total_bad_volume = int((data[bad_flag] > 0).sum())
    else:
        total_volume = int(baseline_stats.get("total_vol", len(data)))
        total_balance = float(baseline_stats.get("total_balance", data[bal_variable].sum()))
        total_bad_balance = float(baseline_stats.get("total_bad_balance", data[bad_bal_variable].sum()))
        total_bad_volume = int(baseline_stats.get("total_bad_vol", (data[bad_flag] > 0).sum()))

    total_good_volume = total_volume - total_bad_volume

    original_br_vol = np.round((total_bad_volume / total_volume) * 100, 2) if total_volume > 0 else 0.0
    original_br_bal = np.round((total_bad_balance / total_balance) * 100, 2) if total_balance > 0 else 0.0

    # ---- marginal (rule-selected) population ----
    mask = _resolve_mask(data, rule)
    marginal = data.loc[mask]

    marginal_volume = int(mask.sum())
    marginal_balance = float(marginal[bal_variable].sum())
    marginal_bad_balance = float(marginal[bad_bal_variable].sum())
    marginal_bad_volume = int((marginal[bad_flag] > 0).sum())
    marginal_good_volume = marginal_volume - marginal_bad_volume

    marginal_br_vol = np.round((marginal_bad_volume / marginal_volume) * 100, 2) if marginal_volume > 0 else 0.0
    marginal_br_bal = np.round((marginal_bad_balance / marginal_balance) * 100, 2) if marginal_balance > 0 else 0.0

    marginal_volume_pct = np.round((marginal_volume / total_volume) * 100, 2) if total_volume > 0 else 0.0
    marginal_balance_pct = np.round((marginal_balance / total_balance) * 100, 2) if total_balance > 0 else 0.0
    marginal_bad_volume_pct = np.round((marginal_bad_volume / total_bad_volume) * 100, 2) if total_bad_volume > 0 else 0.0
    marginal_bad_balance_pct = np.round((marginal_bad_balance / total_bad_balance) * 100, 2) if total_bad_balance > 0 else 0.0

    bad_removed_share = (marginal_bad_volume / total_bad_volume) if total_bad_volume > 0 else np.nan
    good_removed_share = (marginal_good_volume / total_good_volume) if total_good_volume > 0 else np.nan
    gb_ratio = np.round(bad_removed_share / good_removed_share, 4) if (good_removed_share not in (0, np.nan) and good_removed_share > 0) else np.nan

    # ---- new baseline after removing marginal ----
    new_volume = total_volume - marginal_volume
    new_balance = total_balance - marginal_balance
    new_bad_volume = total_bad_volume - marginal_bad_volume
    new_bad_balance = total_bad_balance - marginal_bad_balance

    new_br_vol = np.round((new_bad_volume / new_volume) * 100, 2) if new_volume > 0 else 0.0
    new_br_bal = np.round((new_bad_balance / new_balance) * 100, 2) if new_balance > 0 else 0.0

    out: Dict[str, Any] = {
        # baseline
        "total_volume": total_volume,
        "total_balance": total_balance,
        "total_bad_volume": total_bad_volume,
        "total_bad_balance": total_bad_balance,
        "baseline_br_vol_pct": original_br_vol,
        "baseline_br_bal_pct": original_br_bal,
        # marginal / removed
        "marginal_volume": marginal_volume,
        "marginal_volume_pct": marginal_volume_pct,
        "marginal_balance": marginal_balance,
        "marginal_balance_pct": marginal_balance_pct,
        "marginal_bad_volume": marginal_bad_volume,
        "marginal_bad_volume_pct_of_baseline_bad": marginal_bad_volume_pct,
        "marginal_bad_balance": marginal_bad_balance,
        "marginal_bad_balance_pct_of_baseline_bad_balance": marginal_bad_balance_pct,
        "marginal_br_vol_pct": marginal_br_vol,
        "marginal_br_bal_pct": marginal_br_bal,
        "G_to_B": gb_ratio,
        # new baseline
        "new_volume": new_volume,
        "new_balance": new_balance,
        "new_bad_volume": new_bad_volume,
        "new_bad_balance": new_bad_balance,
        "new_br_vol_pct": new_br_vol,
        "new_br_bal_pct": new_br_bal,
    }

    if print_details:
        print("\n==================== BASELINE PERFORMANCE ====================")
        print(f"Total Volume      : {out['total_volume']}")
        print(f"Total Balance     : {out['total_balance']:,.2f}")
        print(f"Total Bad Volume  : {out['total_bad_volume']}")
        print(f"Total Bad Balance : {out['total_bad_balance']:,.2f}")
        print(f"BR (Vol)          : {out['baseline_br_vol_pct']}%")
        print(f"BR (Bal)          : {out['baseline_br_bal_pct']}%")

        print("\n==================== RULE IMPACT (MARGINAL) ====================")
        print(f"Rule Volume       : {out['marginal_volume']} ({out['marginal_volume_pct']}% of total)")
        print(f"Rule Balance      : {out['marginal_balance']:,.2f} ({out['marginal_balance_pct']}% of total)")
        print(f"Rule Bad Volume   : {out['marginal_bad_volume']} ({out['marginal_bad_volume_pct_of_baseline_bad']}% of all bads)")
        print(f"Rule Bad Balance  : {out['marginal_bad_balance']:,.2f} ({out['marginal_bad_balance_pct_of_baseline_bad_balance']}% of bad bal)")
        print(f"Rule BR (Vol)     : {out['marginal_br_vol_pct']}% (Baseline={out['baseline_br_vol_pct']}%)")
        print(f"Rule BR (Bal)     : {out['marginal_br_bal_pct']}% (Baseline={out['baseline_br_bal_pct']}%)")
        print(f"G:B (G_to_B)      : {out['G_to_B']}")

        print("\n==================== NEW BASELINE AFTER RULE ====================")
        print(f"New Volume        : {out['new_volume']}")
        print(f"New Balance       : {out['new_balance']:,.2f}")
        print(f"New Bad Volume    : {out['new_bad_volume']}")
        print(f"New Bad Balance   : {out['new_bad_balance']:,.2f}")
        print(f"New BR (Vol)      : {out['new_br_vol_pct']}%")
        print(f"New BR (Bal)      : {out['new_br_bal_pct']}%")
        print("\n===============================================================\n")

    return out if return_details else {}


# ==============================================================
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



#=======================================
#         Summary table creation 
#=======================================
def rule_to_string(rule, fallback: str = "<unnamed rule>") -> str:
    """
    Return a human-readable rule identifier.

    Priority:
    1. rule.name (if set)
    2. fallback
    """
    try:
        name = getattr(rule, "name", None)
        if isinstance(name, str) and name.strip():
            return name
    except Exception:
        pass

    return fallback




def build_rule_impact_table_from_masks(
    data: pd.DataFrame,
    rules: Union[List[pd.Series], Dict[str, pd.Series]],
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
) -> pd.DataFrame:
    # ---- baseline ----
    base_vol = len(data)
    base_bad_vol = int((data[bad_flag] > 0).sum())
    base_bal = float(data[total_bal].sum())
    base_bad_bal = float(data[bad_bal].sum())
    base_good_vol = base_vol - base_bad_vol

    # normalize input
    if isinstance(rules, dict):
        items = list(rules.items())  # (name, rule)
    else:
        items = [(None, r) for r in rules]

    rows = []

    for i, (name, rule) in enumerate(items, start=1):
        mask = pd.Series(rule, index=data.index).astype(bool)
        removed = data.loc[mask]

        # ---- naming priority: dict key > rule.name > fallback ----
        rule_logic = (
            name
            or (rule.name if isinstance(rule, pd.Series) else None)
            or f"rule_{i}"
        )

        # ---- marginal removed stats ----
        rem_vol = int(mask.sum())
        rem_bad_vol = int((removed[bad_flag] > 0).sum())
        rem_good_vol = rem_vol - rem_bad_vol
        rem_bal = float(removed[total_bal].sum())
        rem_bad_bal = float(removed[bad_bal].sum())

        # ---- new baseline after removing ----
        new_vol = base_vol - rem_vol
        new_bad_vol = base_bad_vol - rem_bad_vol
        new_bal = base_bal - rem_bal
        new_bad_bal = base_bad_bal - rem_bad_bal

        marginal_br_vol = 0.0 if rem_vol == 0 else round(rem_bad_vol / rem_vol * 100, 2)
        marginal_br_bal = 0.0 if rem_bal == 0 else round(rem_bad_bal / rem_bal * 100, 2)

        # ---- G:B ratio ----
        bad_removed_share = rem_bad_vol / base_bad_vol if base_bad_vol > 0 else np.nan
        good_removed_share = rem_good_vol / base_good_vol if base_good_vol > 0 else np.nan
        if good_removed_share == 0:
            gb_ratio = np.inf if bad_removed_share > 0 else np.nan
        else:
            gb_ratio = round(bad_removed_share / good_removed_share, 4)

        rows.append({
            "rule_logic": rule_logic,
            "total_volume": new_vol,
            "total_reduced": rem_vol,
            "total_bad_volume": new_bad_vol,
            "bad_volume_reduced": rem_bad_vol,
            "total_balance": new_bal,
            "balance_reduced_pct": round(rem_bal / base_bal * 100, 2) if base_bal > 0 else 0.0,
            "total_bad_balance": new_bad_bal,
            "bad_balance_reduced_pct": round(rem_bad_bal / base_bad_bal * 100, 2) if base_bad_bal > 0 else 0.0,
            "marginal_bad_vol_rate_pct": marginal_br_vol,
            "marginal_bad_bal_rate_pct": marginal_br_bal,
            "G_to_B": gb_ratio,
        })

    return pd.DataFrame(rows)
