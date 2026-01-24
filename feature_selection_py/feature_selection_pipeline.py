# feature_selection_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---- your split modules (update names if needed) ----
from .fs_utils import _require_columns  # keep this small helper centralized
from .fs_iv import (
    num_cat_list,
    information_value_calculation_dictionary,
    iv_survived,
)
from .fs_filters import (
    correlation_between_features,
    get_highly_corr,
    filter_iv_corr,
    calculate_vif,
    rank_features_by_worst_tail_bad_vol,
    rank_features_by_worst_tail_bad_bal,
)


@dataclass
class FeatureSelectionConfig:
    # --- IV ---
    iv_bins: int = 10
    iv_threshold: float = 0.02

    # --- Correlation filter (numeric only) ---
    corr_threshold: float = 0.85

    # --- VIF (numeric only; optional) ---
    use_vif: bool = False
    vif_threshold: float = 10.0
    vif_drop_high_vif: bool = True
    vif_max_iter: int = 100
    vif_fillna: str = "median"  # {"drop","median","zero"}

    # --- Worst-tail ranking (optional diagnostics) ---
    run_worst_tail_rank: bool = True
    worst_pct: float = 0.05
    min_non_missing: int = 200
    missing_sentinels: Tuple[float, ...] = (-9999,)

    # --- Output control ---
    keep_categorical: bool = True
    max_features: Optional[int] = None  # cap the total features (after filtering)


def _normalize_data_dictionary(data_dictionary: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Normalise a data dictionary into predictable lower-case columns.
    Expected (any case): variable, definition, direction, valid min, valid max
    """
    if data_dictionary is None or data_dictionary.empty:
        return None

    dd = data_dictionary.copy()
    dd.columns = [c.strip().lower() for c in dd.columns]

    # common aliases
    rename = {}
    if "var" in dd.columns and "variable" not in dd.columns:
        rename["var"] = "variable"
    if "def" in dd.columns and "definition" not in dd.columns:
        rename["def"] = "definition"
    dd = dd.rename(columns=rename)

    if "variable" in dd.columns:
        dd["variable"] = dd["variable"].astype(str).str.strip()

    return dd


def _clean_var_name_for_dict_match(col: str) -> str:
    """
    Mirrors your earlier rule: split by '_' and take last token, strip trailing ' -'
    """
    s = str(col)
    cleaned = s.split("_")[-1] if "_" in s else s
    cleaned = cleaned.rstrip(" -") if " -" in cleaned else cleaned
    return cleaned


def _build_variable_spec(
    selected_num: Sequence[str],
    selected_cat: Sequence[str],
    data_dictionary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a rule-ready variable specification table.

    IMPORTANT:
    - This is a *generic* schema that you can extend to match your `rules_searching.py`
      expected `variable_dateframe` schema.
    - Keeps key metadata: definition, direction, valid min/max (if provided).
    """
    dd = _normalize_data_dictionary(data_dictionary)
    dd_lookup = dd.set_index("variable", drop=False) if dd is not None and "variable" in dd.columns else None

    rows: List[Dict[str, Any]] = []

    def add_one(var: str, vtype: str):
        cleaned = _clean_var_name_for_dict_match(var)
        definition = None
        direction = 0
        valid_min = None
        valid_max = None

        if dd_lookup is not None and cleaned in dd_lookup.index:
            r = dd_lookup.loc[cleaned]
            if "definition" in dd_lookup.columns:
                definition = r.get("definition", None)
            if "direction" in dd_lookup.columns:
                try:
                    direction = int(float(r.get("direction", 0)))
                except Exception:
                    direction = 0
            # valid min/max columns can be in several formats
            for k in ("valid min", "valid_min", "min"):
                if k in dd_lookup.columns:
                    v = r.get(k, None)
                    if pd.notnull(v):
                        valid_min = float(v)
                        break
            for k in ("valid max", "valid_max", "max"):
                if k in dd_lookup.columns:
                    v = r.get(k, None)
                    if pd.notnull(v):
                        valid_max = float(v)
                        break

        rows.append(
            {
                "variable": var,            # raw dataset column
                "cleaned_name": cleaned,    # dictionary match key
                "type": vtype,              # "num" / "cat"
                "definition": definition,
                "direction": direction,     # -1 / 0 / +1 (0 = unknown)
                "valid_min": valid_min,
                "valid_max": valid_max,
            }
        )

    for v in selected_num:
        add_one(v, "num")
    for v in selected_cat:
        add_one(v, "cat")

    return pd.DataFrame(rows)


def run_feature_selection_pipeline(
    data: pd.DataFrame,
    *,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    config: FeatureSelectionConfig = FeatureSelectionConfig(),
) -> Dict[str, Any]:
    """
    Main pipeline entry-point.

    Returns dict with:
      - selected_num: List[str]
      - selected_cat: List[str]
      - variable_spec: pd.DataFrame
      - artifacts: dict of intermediate tables
    """
    _require_columns(data, [bad_flag, bal_variable, bad_bal_variable])

    # 1) Split numeric/categorical
    num_list, cat_list = num_cat_list(data, bad_flag, bal_variable, bad_bal_variable)

    # 2) IV table (optionally enriched)
    iv_table = information_value_calculation_dictionary(
        data=data,
        bad_flag=bad_flag,
        num_list=num_list,
        cat_list=cat_list,
        data_dictionary=data_dictionary,
        bins=config.iv_bins,
    )

    survived_iv_set, _, _, _ = iv_survived(
        iv_table=iv_table,
        iv_threshold=config.iv_threshold,
        num_list=num_list,
        cat_list=cat_list,
    )

    # Candidate lists after IV
    num_after_iv = [c for c in num_list if c in survived_iv_set]
    cat_after_iv = [c for c in cat_list if c in survived_iv_set] if config.keep_categorical else []

    # 3) Correlation filter (numeric only)
    corr_matrix = correlation_between_features(data, num_after_iv)
    highly_corr_pairs = get_highly_corr(corr_matrix, config.corr_threshold)

    final_after_iv_corr = filter_iv_corr(iv_table, highly_corr_pairs, config.iv_threshold)
    # filter_iv_corr returns a list across all types, but corr pairs are numeric-only,
    # so this effectively leaves cat untouched (cats not in corr_matrix).
    num_after_iv_corr = [c for c in num_after_iv if c in final_after_iv_corr]
    cat_after_iv_corr = [c for c in cat_after_iv if c in final_after_iv_corr]

    # 4) Optional VIF (numeric only)
    vif_table = None
    if config.use_vif and len(num_after_iv_corr) >= 2:
        # calculate_vif auto-excludes bad/bal/bad_bal, but it uses *all numeric columns*.
        # If you want VIF only on the current candidate list, you can adapt fs_filters later.
        vif_table, remaining_after_vif = calculate_vif(
            data=data[[*num_after_iv_corr, bad_flag, bal_variable, bad_bal_variable]].copy(),
            bad_flag=bad_flag,
            bal_variable=bal_variable,
            bad_bal_variable=bad_bal_variable,
            vif_threshold=config.vif_threshold,
            drop_high_vif=config.vif_drop_high_vif,
            max_iter=config.vif_max_iter,
            fillna=config.vif_fillna,
        )
        selected_num = [c for c in num_after_iv_corr if c in remaining_after_vif]
    else:
        selected_num = list(num_after_iv_corr)

    selected_cat = list(cat_after_iv_corr) if config.keep_categorical else []

    # 5) Optional cap on number of features (keep highest IV)
    if config.max_features is not None:
        iv_map = (
            iv_table[["Variable", "Information Value"]]
            .dropna()
            .set_index("Variable")["Information Value"]
            .to_dict()
        )

        combined = selected_num + selected_cat
        combined_sorted = sorted(combined, key=lambda v: float(iv_map.get(v, -1)), reverse=True)
        combined_sorted = combined_sorted[: int(config.max_features)]

        selected_num = [c for c in selected_num if c in combined_sorted]
        selected_cat = [c for c in selected_cat if c in combined_sorted]

    # 6) Optional worst-tail diagnostics
    worst_tail_bad_vol = None
    worst_tail_bad_bal = None
    if config.run_worst_tail_rank and len(selected_num) > 0:
        worst_tail_bad_vol = rank_features_by_worst_tail_bad_vol(
            data=data,
            bad_flag=bad_flag,
            num_list=selected_num,
            data_dictionary=data_dictionary,
            worst_pct=config.worst_pct,
            min_non_missing=config.min_non_missing,
            missing_sentinels=config.missing_sentinels,
            direction_default=0,
        )

        worst_tail_bad_bal = rank_features_by_worst_tail_bad_bal(
            data=data,
            bad_flag=bad_flag,
            bad_bal=bad_bal_variable,
            num_list=selected_num,
            data_dictionary=data_dictionary,
            worst_pct=config.worst_pct,
            min_non_missing=config.min_non_missing,
            missing_sentinels=config.missing_sentinels,
            direction_default=0,
        )

    # 7) Build variable spec (contract output for rules_searching)
    variable_spec = _build_variable_spec(selected_num, selected_cat, data_dictionary=data_dictionary)

    artifacts = {
        "num_list": num_list,
        "cat_list": cat_list,
        "iv_table": iv_table,
        "survived_iv_set": survived_iv_set,
        "corr_matrix": corr_matrix,
        "highly_corr_pairs": highly_corr_pairs,
        "vif_table": vif_table,
        "worst_tail_bad_vol": worst_tail_bad_vol,
        "worst_tail_bad_bal": worst_tail_bad_bal,
    }

    return {
        "selected_num": selected_num,
        "selected_cat": selected_cat,
        "variable_spec": variable_spec,
        "artifacts": artifacts,
    }
