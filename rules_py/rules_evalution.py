from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd

# Plotting (optional utilities in this module)
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# =================================
#       Redundancy Analysis
# =================================
from matplotlib_venn import venn2, venn2_circles

def two_rules_redundancy(
        rule1_total, rule1_bad,
        rule2_total, rule2_bad,
        rule12_both_total, rule12_both_bad,
        rule1_name, rule2_name
):

    only_r1_total = rule1_total - rule12_both_total
    only_r2_total = rule2_total - rule12_both_total

    venn2(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('All Declined Accounts')
    plt.show()

    only_r1_bad = rule1_bad - rule12_both_bad
    only_r2_bad = rule2_bad - rule12_both_bad

    venn2(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('Detected Bad Accounts')
    plt.show()


def three_rules_redundancy(
        r1_total, r1_bad,
        r2_total, r2_bad,
        r3_total, r3_bad,
        r12_total, r12_bad,
        r13_total, r13_bad,
        r23_total, r23_bad,
        r123_total, r123_bad,
        rule1_name, rule2_name, rule3_name
):
    """
    Visualize redundancy across 3 rules using Venn diagrams.
    
    All inputs represent counts:
      - r1_total: customers hit by Rule 1
      - r12_total: customers hit by both Rule 1 and Rule 2
      - r123_total: customers hit by Rules 1,2,3
      etc.
    Same structure for xxx_bad counts.
    """

    only_r1 = r1_total - r12_total - r13_total + r123_total
    only_r2 = r2_total - r12_total - r23_total + r123_total
    only_r3 = r3_total - r13_total - r23_total + r123_total

    only_r12 = r12_total - r123_total
    only_r13 = r13_total - r123_total
    only_r23 = r23_total - r123_total
    only_r123 = r123_total

    venn3(
        subsets = (
            only_r1,
            only_r2,
            only_r12,
            only_r3,
            only_r13,
            only_r23,
            only_r123
        ),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange','blue','green'),
        alpha=0.6
    )
    venn3_circles(
        subsets = (
            only_r1, only_r2, only_r12,
            only_r3, only_r13, only_r23, only_r123
        ),
        linestyle='dashed',
        linewidth=1
    )

    plt.title("All Declined Accounts")
    plt.show()


    only_r1_bad = r1_bad - r12_bad - r13_bad + r123_bad
    only_r2_bad = r2_bad - r12_bad - r23_bad + r123_bad
    only_r3_bad = r3_bad - r13_bad - r23_bad + r123_bad

    only_r12_bad = r12_bad - r123_bad
    only_r13_bad = r13_bad - r123_bad
    only_r23_bad = r23_bad - r123_bad
    only_r123_bad = r123_bad

    venn3(
        subsets = (
            only_r1_bad,
            only_r2_bad,
            only_r12_bad,
            only_r3_bad,
            only_r13_bad,
            only_r23_bad,
            only_r123_bad
        ),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange','blue','green'),
        alpha=0.6
    )
    venn3_circles(
        subsets = (
            only_r1_bad, only_r2_bad, only_r12_bad,
            only_r3_bad, only_r13_bad, only_r23_bad, only_r123_bad
        ),
        linestyle='dashed',
        linewidth=1
    )

    plt.title("Detected Bad Accounts")
    plt.show()


# =================================
#       New Baseline Analysis

# =================================
#       New Baseline Analysis
# =================================
#
# Canonical implementations live in rules_searching_v2.py (or rules_searching.py).
# This file re-exports them for convenience so notebooks that already import
# from rules_evaluation keep working.


# ---------------------------------------------------------------------
# Core scoring utilities (defined here so this module is standalone)
# ---------------------------------------------------------------------
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
    """Coerce a rule into a boolean Series aligned to data.index."""
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
    if d == 0:
        return float(np.inf) if n > 0 else float(np.nan)
    return float(np.round(n / d, ndigits))


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
      - `rule` evaluates to a boolean mask of rows to REMOVE/DECLINE.
        True  => removed by the rule
        False => kept in the new baseline

    Returns a dict with stable keys:

      New baseline:
        - new_total_volume, new_total_bad, new_bad_vol_pct
        - new_total_balance, new_bad_balance, new_bad_bal_pct

      Removed by rule:
        - volume_decreased, volume_decreased_pct
        - bad_removed, bad_removed_pct_of_baseline_bad
        - balance_decreased, balance_decreased_pct
        - bad_balance_reduced, bad_balance_reduced_pct_of_baseline_bad_balance

      Marginal comparisons (ratios vs baseline rates):
        - marginal_bad_vol_pct_over_baseline_bad_vol_pct
        - marginal_bad_bal_pct_over_baseline_bad_bal_pct

      G:B:
        - G_to_B  (bad removed share / good removed share)

      Baseline reference:
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

    # Ratios of rates vs baseline ("over")
    mar_bad_vol_over_base = _ratio(new_br_vol_pct, base_br_vol_pct)
    mar_bad_bal_over_base = _ratio(new_br_bal_pct, base_br_bal_pct)

    # G:B = (bad removed share) / (good removed share)
    bad_removed_share = _safe_div(rem_bad_vol, base_bad_vol) if base_bad_vol else np.nan
    good_removed_share = _safe_div(rem_good_vol, base_good_vol) if base_good_vol else np.nan
    gb = _ratio(bad_removed_share, good_removed_share)

    return {
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
        "marginal_bad_vol_pct_over_baseline_bad_vol_pct": mar_bad_vol_over_base,
        "marginal_bad_bal_pct_over_baseline_bad_bal_pct": mar_bad_bal_over_base,
        "G_to_B": gb,
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

from typing import Callable, Union, Any, Dict, List

import importlib.util
from pathlib import Path as _Path

def _load_local_module(module_filename: str, module_name: str):
    here = _Path(__file__).resolve().parent
    target = here / module_filename
    if not target.exists():
        raise ModuleNotFoundError(f"Cannot find {module_filename} next to {__file__}")
    spec = importlib.util.spec_from_file_location(module_name, str(target))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

try:
    from rules_searching_v2 import RuleLike, new_baseline_performance_after_rule, build_new_baseline_table_from_rule_list
except Exception:  # pragma: no cover
    try:
        from rules_searching import RuleLike, new_baseline_performance_after_rule, build_new_baseline_table_from_rule_list
    except Exception:  # pragma: no cover
        _m = None
        for fn in ("rules_searching_v2.py", "rules_searching.py"):
            try:
                _m = _load_local_module(fn, fn.replace(".py", ""))
                break
            except Exception:
                continue
        if _m is None:
            raise
        RuleLike = getattr(_m, "RuleLike")
        new_baseline_performance_after_rule = getattr(_m, "new_baseline_performance_after_rule")
        build_new_baseline_table_from_rule_list = getattr(_m, "build_new_baseline_table_from_rule_list")


# ==============================================
#     Visualise the Group analysis results 
# ==============================================
def group_performance_one_rule(
    group_data,
    top_x,
    original_bal,
    new_bal,
    original_bad_bal,
    new_bad_bal,
    original_bal_br,
    new_bal_br
):
    """
    Produce two plots:
    
    Plot 1: Bad Balance + Balance BR (line)
    Plot 2: Total Balance + Balance BR (line)

    Parameters
    ----------
    group_data : pd.DataFrame
        Data indexed by group name.
    top_x : int
        Number of rows (after filtering Missing) to display.
    Column name parameters : str
        Names of relevant columns in the dataframe.
    """

    # Filter out Missing and take top N
    selected = group_data[group_data.index != 'Missing'].copy().head(top_x)

    # ==========================================
    #   PLOT 1: BAD BALANCE + BR
    # ==========================================
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])

    fig1.add_trace(
        go.Bar(x=selected.index, y=selected[original_bad_bal],name='Original Bad Balance'),
        secondary_y=False
    )

    fig1.add_trace(
        go.Bar(x=selected.index,y=selected[new_bad_bal],name='New Bad Balance'),
        secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(x=selected.index,y=selected[original_bal_br],mode='lines+markers',name='Original Bal BR'),
        secondary_y=True
    )

    fig1.add_trace(
        go.Scatter(x=selected.index,y=selected[new_bal_br],mode='lines+markers',name='New Bal BR'),
        secondary_y=True
    )

    fig1.update_layout(
        title='Bad Balance & Bal BR Changes',
        width=900,
        height=600,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis_title='Group',
        yaxis_title='Bad Balance',
        template='plotly_white'
    )

    fig1.update_yaxes(
        title_text='Balance BR (%)',
        secondary_y=True
    )

    fig1.show()

    # ==========================================
    #   PLOT 2: TOTAL BALANCE + BR
    # ==========================================
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    fig2.add_trace(
        go.Bar(x=selected.index,y=selected[original_bal],name='Original Balance'),
        secondary_y=False
    )

    fig2.add_trace(
        go.Bar(x=selected.index,y=selected[new_bal],name='New Balance'),
        secondary_y=False
    )

    fig2.add_trace(
        go.Scatter(x=selected.index,y=selected[original_bal_br],mode='lines+markers',name='Original Bal BR'),
        secondary_y=True
    )

    fig2.add_trace(
        go.Scatter(x=selected.index,y=selected[new_bal_br],mode='lines+markers',name='New Bal BR'),
        secondary_y=True
    )

    fig2.update_layout(
        title='Balance & Bal BR Changes',
        width=900,
        height=600,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis_title='Group',
        yaxis_title='Balance',
        template='plotly_white'
    )

    fig2.update_yaxes(
        title_text='Balance BR (%)',
        secondary_y=True
    )

    fig2.show()



def group_performance_two_rules(
    group_data,
    top_x,
    original_bal,
    new_bal1,
    new_bal2,
    original_bad_bal,
    new_bad_bal1,
    new_bad_bal2,
    original_bal_br,
    new_bal_br1,
    new_bal_br2
):
    """
    Produce two plots:

    Plot 1: Bad Balance vs BR
    Plot 2: Total Balance vs BR

    Each plot contains:
        - 3 bars  (Original, New1, New2)
        - 3 lines (Original BR, New1 BR, New2 BR)
    """

    # -----------------------------------------
    # Filter top groups excluding 'Missing'
    # -----------------------------------------
    selected = group_data[group_data.index != 'Missing'].copy().head(top_x)

    # =====================================================
    #   PLOT 1 — BAD BALANCE
    # =====================================================
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])

    # ---------- Bars ----------
    fig1.add_trace(go.Bar(x=selected.index, y=selected[original_bad_bal], name='Original Bad Bal'), secondary_y=False)
    fig1.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal1],   name='New Rule 1 Bad Bal'), secondary_y=False)
    fig1.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal2],   name='New Rule 2 Bad Bal'), secondary_y=False)

    # ---------- BR lines ----------
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br], mode='lines+markers', name='Original BR'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br1],   mode='lines+markers', name='New Rule 1 BR'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br2],   mode='lines+markers', name='New Rule 2 BR'), secondary_y=True)

    fig1.update_layout(
        title='Bad Balance & BR Comparison (Original vs Rule1 vs Rule2)',
        width=900,
        height=600,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Group',
        yaxis_title='Bad Balance',
        template='plotly_white'
    )
    fig1.update_yaxes(title_text='BR (%)', secondary_y=True)
    fig1.show()

    # =====================================================
    #   PLOT 2 — TOTAL BALANCE
    # =====================================================
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    # ---------- Bars ----------
    fig2.add_trace(go.Bar(x=selected.index, y=selected[original_bal], name='Original Balance'), secondary_y=False)
    fig2.add_trace(go.Bar(x=selected.index, y=selected[new_bal1],     name='New Rule 1 Balance'), secondary_y=False)
    fig2.add_trace(go.Bar(x=selected.index, y=selected[new_bal2],     name='New Rule 2 Balance'), secondary_y=False)

    # ---------- BR lines ----------
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br], mode='lines+markers', name='Original BR'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br1],     mode='lines+markers', name='New Rule 1 BR'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br2],     mode='lines+markers', name='New Rule 2 BR'), secondary_y=True)

    fig2.update_layout(
        title='Balance & BR Comparison (Original vs Rule1 vs Rule2)',
        width=900,
        height=600,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Group',
        yaxis_title='Balance',
        template='plotly_white'
    )
    fig2.update_yaxes(title_text='BR (%)', secondary_y=True)
    fig2.show()
