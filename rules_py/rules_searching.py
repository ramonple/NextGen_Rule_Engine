
"""
rules_searching.py (prepared)

Core utilities for:
- defining rule-search feature specs
- generating 1D atoms from spec (continuous + categorical)
- combining atoms into 2D/3D rules (AND/OR + mixed 3D parentheses)
- evaluating rules quickly with cached baseline stats
- producing simple result tables for optimisation (marginal removed population focus)

Conventions
-----------
- A "rule" (mask) identifies rows to REMOVE (flag/reject).
- bad_flag: bad is defined as (data[bad_flag] > 0)
- total_bal: exposure / total balance column
- bad_bal: bad balance column
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ast

import numpy as np
import pandas as pd


# =============================================================================
# Feature spec template
# =============================================================================

def final_feature_for_rule_construction() -> pd.DataFrame:
    """
    Create an empty DataFrame template used for defining feature rules.

    Columns:
        - variable: feature name in `data`
        - Valid Min / Valid Max: meaningful value bounds (use these to exclude imputation extremes, e.g. -9999)
        - Search Min / Search Max / Step: threshold grid to scan
        - Direction: relationship to bad rate
            -1 = higher is better (bad tail on low side)
            +1 = higher is worse (bad tail on high side)
        - Type: "continuous" or "categorical"
        - Search Values: (categorical only) candidate sets; examples:
            ['xx','yy'] OR "xx,yy" OR "[['xx','yy'], ['aa']]"
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
        "Search Values",  # categorical candidate sets
    ]
    return pd.DataFrame(columns=columns)


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


# =============================================================================
# Rule generation
# =============================================================================

@dataclass(frozen=True)
class AtomRule:
    feature: str
    expr: str
    mask: pd.Series


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Step must be > 0")
    if stop < start:
        return []
    n = int(np.floor((stop - start) / step)) + 1
    vals = [start + i * step for i in range(n)]
    return [float(np.round(v, 10)) for v in vals if v <= stop + 1e-12]


def _parse_category_sets(val: Any) -> List[List[Any]]:
    """
    Categorical candidate sets can be provided as:
      - ['xx','yy']
      - [['xx','yy'], ['aa']]
      - "xx,yy"
      - "['xx','yy']"
      - "[['xx','yy'], ['aa']]"
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        vv = list(val)
        if vv and all(isinstance(x, (list, tuple, set)) for x in vv):
            return [list(x) for x in vv]
        return [vv]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            return _parse_category_sets(obj)
        except Exception:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [parts]
    return [[val]]


def _build_1d_atoms_from_spec(
    data: pd.DataFrame,
    variable_dataframe: pd.DataFrame,
    *,
    var_col: str = "variable",
    type_col: str = "Type",
    direction_col: str = "Direction",
    valid_min_col: str = "Valid Min",
    valid_max_col: str = "Valid Max",
    search_min_col: str = "Search Min",
    search_max_col: str = "Search Max",
    step_col: str = "Step",
    cat_values_col: str = "Search Values",
) -> List[AtomRule]:
    """
    Build all 1D rule atoms from the variable spec.

    Continuous:
      - Computes effective bounds to exclude imputation extremes:
          LB = max(ValidMin, SearchMin, observed_min)
          UB = min(ValidMax, SearchMax, observed_max)
      - Direction = -1 (higher better): select LOW tail within bounds:  LB <= x < t
      - Direction = +1 (higher worse): select HIGH tail within bounds: t < x <= UB
      - Expr includes LB/UB actually used.

    Categorical:
      - Uses Search Values -> data['x'].isin([...])
    """
    if var_col not in variable_dataframe.columns:
        raise KeyError(f"variable_dataframe missing column: {var_col}")

    atoms: List[AtomRule] = []

    for _, row in variable_dataframe.iterrows():
        feature = str(row.get(var_col))
        if feature not in data.columns:
            continue

        ftype = str(row.get(type_col, "continuous")).lower().strip()

        # ----- Continuous -----
        if ftype in ("continuous", "numeric", "float", "int"):
            try:
                direction = int(row.get(direction_col))
            except Exception:
                continue
            if direction not in (-1, 1):
                continue

            vmin = _to_float(row.get(valid_min_col))
            vmax = _to_float(row.get(valid_max_col))
            smin = _to_float(row.get(search_min_col))
            smax = _to_float(row.get(search_max_col))
            step = _to_float(row.get(step_col))

            if step is None or step <= 0:
                continue

            x = data[feature]
            obs_min = float(np.nanmin(x.values))
            obs_max = float(np.nanmax(x.values))

            LB = float(max([c for c in [vmin, smin, obs_min] if c is not None]))
            UB = float(min([c for c in [vmax, smax, obs_max] if c is not None]))
            if UB <= LB:
                continue

            t_start = float(max(smin if smin is not None else LB, LB))
            t_stop = float(min(smax if smax is not None else UB, UB))
            thresholds = _frange(t_start, t_stop, step)
            if not thresholds:
                continue

            for t in thresholds:
                if direction == -1:
                    mask = (x >= LB) & (x < t)
                    expr = f"({LB} <= data['{feature}'] < {t}) [UB={UB}]"
                else:
                    mask = (x > t) & (x <= UB)
                    expr = f"({t} < data['{feature}'] <= {UB}) [LB={LB}]"

                atoms.append(AtomRule(feature=feature, expr=expr, mask=pd.Series(mask, index=data.index)))

        # ----- Categorical -----
        elif ftype in ("categorical", "category", "cat", "string", "str", "object"):
            cat_sets = _parse_category_sets(row.get(cat_values_col))
            if not cat_sets:
                continue
            x = data[feature]
            for cats in cat_sets:
                mask = x.isin(cats)
                expr = f"data['{feature}'].isin({cats})"
                atoms.append(AtomRule(feature=feature, expr=expr, mask=pd.Series(mask, index=data.index)))

        else:
            continue

    return atoms


def _combine_atoms(
    atoms: List[AtomRule],
    dim: int,
    *,
    include_and: bool = True,
    include_or: bool = True,
    include_mixed_3d: bool = True,
    forbid_same_feature: bool = True,
) -> List[Tuple[str, pd.Series]]:
    """
    Combine 1D atoms into dim-D rules.

    dim=2:
      - (A & B), (A | B)

    dim=3:
      - (A & B & C), (A | B | C)
      - mixed parentheses forms:
          (A | B) & C
          (A | C) & B
          A & (B | C)   (printed that way)
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    out: List[Tuple[str, pd.Series]] = []

    for combo in combinations(atoms, dim):
        if forbid_same_feature:
            feats = [a.feature for a in combo]
            if len(set(feats)) != len(feats):
                continue

        exprs = [a.expr for a in combo]

        if include_and:
            m = combo[0].mask.copy()
            for a in combo[1:]:
                m = m & a.mask
            out.append((f"{dim}D AND: " + " & ".join([f"({e})" for e in exprs]), pd.Series(m, index=combo[0].mask.index)))

        if include_or:
            m = combo[0].mask.copy()
            for a in combo[1:]:
                m = m | a.mask
            out.append((f"{dim}D OR: " + " | ".join([f"({e})" for e in exprs]), pd.Series(m, index=combo[0].mask.index)))

        if dim == 3 and include_mixed_3d:
            a, b, c = combo
            m1 = (a.mask | b.mask) & c.mask
            l1 = f"3D MIX: (({a.expr}) | ({b.expr})) & ({c.expr})"
            out.append((l1, pd.Series(m1, index=a.mask.index)))

            m2 = (a.mask | c.mask) & b.mask
            l2 = f"3D MIX: (({a.expr}) | ({c.expr})) & ({b.expr})"
            out.append((l2, pd.Series(m2, index=a.mask.index)))

            m3 = (b.mask | c.mask) & a.mask
            l3 = f"3D MIX: ({a.expr}) & (({b.expr}) | ({c.expr}))"
            out.append((l3, pd.Series(m3, index=a.mask.index)))

    return out


# =============================================================================
# Search algorithms (simple output)
# =============================================================================

def _evaluate_rules_to_simple_table(
    data: pd.DataFrame,
    rules: List[Tuple[str, pd.Series]],
    *,
    baseline_stats: Dict[str, float],
    metric_name: str,
    min_bads: int,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    bal_times_mode: str = "removed",
) -> pd.DataFrame:
    """
    Output:
      Rule | Volume Removed | Bad Vol Detected | Total Balance Removed | Bad Balance Removed | Metric
    """
    rows: List[Dict[str, Any]] = []

    for logic, mask in rules:
        summary = rule_metric_summary(
            data=data,
            rule=mask,
            metric_name=metric_name,
            min_bads=min_bads,
            bad_flag=bad_flag,
            total_bal=total_bal,
            bad_bal=bad_bal,
            baseline_stats=baseline_stats,
            bal_times_mode=bal_times_mode,
        )

        metric = summary["metric"]
        if metric is None or (isinstance(metric, float) and not np.isfinite(metric)):
            continue

        rows.append(
            {
                "Rule": logic,
                "Volume Removed": int(summary["volume_removed"]),
                "Bad Vol Detected": int(summary["bad_vol_detected"]),
                "Total Balance Removed": float(summary["total_balance_removed"]) if np.isfinite(summary["total_balance_removed"]) else float("nan"),
                "Bad Balance Removed": float(summary["bad_balance_removed"]) if np.isfinite(summary["bad_balance_removed"]) else float("nan"),
                "Metric": float(metric) if np.isfinite(metric) else metric,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Metric", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def Rules_Optimisation_Search_Algorithm_1D(
    data: pd.DataFrame,
    variable_dateframe: pd.DataFrame,
    bad_flag: str,
    bad_bal: str,
    total_bal: str,
    selected_function: Optional[Callable[[Dict[str, Any]], float]],  # kept for signature compatibility; not used
    Metric_name: str,
    min_bads: int,
) -> pd.DataFrame:
    """
    Generate and evaluate all 1D atoms.
    Note: selected_function is accepted for compatibility but not used here; use Metric_name instead.
    """
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)
    atoms = _build_1d_atoms_from_spec(data, variable_dateframe)
    rules_1d = [(f"1D: {a.expr}", a.mask) for a in atoms]
    return _evaluate_rules_to_simple_table(
        data,
        rules_1d,
        baseline_stats=baseline_stats,
        metric_name=Metric_name,
        min_bads=min_bads,
        bad_flag=bad_flag,
        total_bal=total_bal,
        bad_bal=bad_bal,
    )


def Rules_Optimisation_Search_Algorithm_2D(
    data: pd.DataFrame,
    variable_dateframe: pd.DataFrame,
    bad_flag: str,
    bad_bal: str,
    total_bal: str,
    selected_function: Optional[Callable[[Dict[str, Any]], float]],  # kept for signature compatibility; not used
    Metric_name: str,
    min_bads: int,
) -> pd.DataFrame:
    """
    Generate and evaluate 2D rules: AND/OR over distinct features.
    """
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)
    atoms = _build_1d_atoms_from_spec(data, variable_dateframe)
    rules_2d = _combine_atoms(atoms, 2, include_and=True, include_or=True, forbid_same_feature=True)
    return _evaluate_rules_to_simple_table(
        data,
        rules_2d,
        baseline_stats=baseline_stats,
        metric_name=Metric_name,
        min_bads=min_bads,
        bad_flag=bad_flag,
        total_bal=total_bal,
        bad_bal=bad_bal,
    )


def Rules_Optimisation_Search_Algorithm_3D(
    data: pd.DataFrame,
    variable_dateframe: pd.DataFrame,
    bad_flag: str,
    bad_bal: str,
    total_bal: str,
    selected_function: Optional[Callable[[Dict[str, Any]], float]],  # kept for signature compatibility; not used
    Metric_name: str,
    min_bads: int,
    *,
    include_mixed_3d: bool = True,
) -> pd.DataFrame:
    """
    Generate and evaluate 3D rules:
      - A & B & C
      - A | B | C
      - (A | B) & C, (A | C) & B, A & (B | C)
    """
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)
    atoms = _build_1d_atoms_from_spec(data, variable_dateframe)
    rules_3d = _combine_atoms(
        atoms,
        3,
        include_and=True,
        include_or=True,
        include_mixed_3d=include_mixed_3d,
        forbid_same_feature=True,
    )
    return _evaluate_rules_to_simple_table(
        data,
        rules_3d,
        baseline_stats=baseline_stats,
        metric_name=Metric_name,
        min_bads=min_bads,
        bad_flag=bad_flag,
        total_bal=total_bal,
        bad_bal=bad_bal,
    )


import re, hashlib

def make_rule(expr: str):
    norm = re.sub(r"\s+", " ", expr.strip())
    rule_id = hashlib.md5(norm.encode()).hexdigest()[:8]   # stable short id
    rule_name = re.sub(r"[^a-zA-Z0-9]+", "_", norm).strip("_").lower()
    rule_name = (rule_name[:40] + "_" + rule_id)  # readable + unique
    return {"rule_name": rule_name, "rule_logic": norm}

rules = [
    make_rule("data['a'] > 5"),
    make_rule("(data['b'] <= 10) & (data['c'].isna())"),
]

rules
