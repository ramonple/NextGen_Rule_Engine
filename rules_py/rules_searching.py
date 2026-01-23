from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ast

import numpy as np
import pandas as pd


from rule_metrics import (
    RuleLike,
    rule_metric_summary
    compute_baseline_stats,
    combine_checking_gb_ratio,
    combine_checking_bal_br_times,
)

# =============================================================================
# Solver for selected metric
# =============================================================================
def _resolve_selected_function(metric_name: str):
    m = (metric_name or "").strip().lower()
    if m in ("g_to_b", "gb", "gb_ratio", "g:b"):
        return combine_checking_gb_ratio
    if m in ("br_bal_times", "bal_br_times", "bad_bal", "bad_balance", "br"):
        return combine_checking_bal_br_times
    raise ValueError(f"Unknown Metric_name: {metric_name}")

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
    selected_function: Optional[Callable] = None,
    metric_name: Optional[str] = None,
    min_bads: int,
    bad_flag: str,
    total_bal: str,
    bad_bal: str,
    bal_times_mode: str = "removed",
) -> pd.DataFrame:
    """
    Output:
      Rule | Volume Removed | Bad Vol Detected | Total Balance Removed | Bad Balance Removed | Metric

    If selected_function is provided, it will be used to compute Metric for each rule.
    Otherwise, metric_name + rule_metric_summary will be used (fallback).
    """
    rows: List[Dict[str, Any]] = []

    for logic, mask in rules:
        # ----------------------------
        # 1) compute marginal stats (always)
        # ----------------------------
        volume_removed = int(mask.sum())
        bad_vol_detected = int(((data[bad_flag] > 0) & mask).sum())
        total_balance_removed = float(data.loc[mask, total_bal].sum())
        bad_balance_removed = float(data.loc[mask, bad_bal].sum())

        # ----------------------------
        # 2) metric (one scalar per rule)
        # ----------------------------
        if selected_function is not None:
            # Support both “new” metrics (mask-based) and “old” ones (df-based)
            try:
                # Preferred: selected_function(data, min_bads, mask, bad_flag, total_bal, bad_bal, ...)
                metric = selected_function(
                    data,
                    min_bads,
                    mask,
                    bad_flag,
                    total_bal,
                    bad_bal,
                    baseline_stats=baseline_stats,  # if supported
                    mode=bal_times_mode,            # if supported
                )
            except TypeError:
                # Legacy: selected_function(data, min_bads, rule_df, bad_flag, bad_bal, total_bal)
                rule_df = data.loc[mask]
                metric = selected_function(
                    data,
                    min_bads,
                    rule_df,
                    bad_flag,
                    bad_bal,
                    total_bal,
                )
        else:
            if metric_name is None:
                raise ValueError("Either selected_function or metric_name must be provided.")
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

        # Filter invalid metrics
        if metric is None or (isinstance(metric, float) and not np.isfinite(metric)):
            continue

        rows.append(
            {
                "Rule": logic,
                "Volume Removed": volume_removed,
                "Bad Vol Detected": bad_vol_detected,   # this is your “# bads removed”
                "Total Balance Removed": total_balance_removed,
                "Bad Balance Removed": bad_balance_removed,
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
    Metric_name: str,
    min_bads: int,
) -> pd.DataFrame:
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)
    selected_function = _resolve_selected_function(Metric_name)

    atoms = _build_1d_atoms_from_spec(data, variable_dateframe)
    rules_1d = [(f"1D: {a.expr}", a.mask) for a in atoms]

    return _evaluate_rules_to_simple_table(
        data,
        rules_1d,
        baseline_stats=baseline_stats,
        selected_function=selected_function,
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
    selected_function,  # kept for backward compatibility; can be None
    Metric_name: str,
    min_bads: int,
) -> pd.DataFrame:
    """
    Generate and evaluate 2D rules: AND/OR over distinct features.

    Metric_name determines the scoring function if selected_function is None:
      - GB ratio -> combine_checking_gb_ratio
      - Bad balance BR-times -> combine_checking_bal_br_times
    """
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)

    if selected_function is None:
        selected_function = _resolve_selected_function(Metric_name)

    atoms = _build_1d_atoms_from_spec(data, variable_dateframe)
    rules_2d = _combine_atoms(atoms, 2, include_and=True, include_or=True, forbid_same_feature=True)

    return _evaluate_rules_to_simple_table(
        data,
        rules_2d,
        baseline_stats=baseline_stats,
        selected_function=selected_function,   # <-- now USED
        metric_name=None,                      # <-- not needed when using selected_function
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
    selected_function,  # kept for backward compatibility; can be None
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

    Metric_name determines the scoring function if selected_function is None:
      - GB ratio -> combine_checking_gb_ratio
      - Bad balance BR-times -> combine_checking_bal_br_times
    """
    baseline_stats = compute_baseline_stats(data, bad_flag, total_bal, bad_bal)

    if selected_function is None:
        selected_function = _resolve_selected_function(Metric_name)

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
        selected_function=selected_function,   # <-- now USED
        metric_name=None,                      # <-- not needed when using selected_function
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

