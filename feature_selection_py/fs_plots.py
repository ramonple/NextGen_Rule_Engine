from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Set, List, Dict, Union

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional, used in good_bad_distribution
import plotly.graph_objects as go  # optional, used in plot_information_value
import plotly.express as px        # optional, used in funnel + heatmap + worst-tail plots


# =============================================
#      Information Value Plots
# =============================================
def plot_information_value(information_value_table: pd.DataFrame, top_x: int = 30):
    """
    Plot the top_x variables by Information Value using Plotly.
    Returns the Plotly Figure.

    Requires: plotly
    """
    if go is None:
        raise ImportError("plotly is required for plot_information_value (plotly.graph_objects not found).")

    if top_x <= 0:
        raise ValueError("top_x must be > 0")

    iv_table = (
        information_value_table[["Variable", "Information Value"]]
        .sort_values("Information Value", ascending=False)
        .head(top_x)
        .copy()
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=iv_table["Variable"],
                y=iv_table["Information Value"],
                text=iv_table["Information Value"].round(3),
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        height=500,
        width=max(450, int(top_x * 40)),
        title="Information Value",
        xaxis_title="Variable",
        yaxis_title="Information Value",
        template="plotly_white",
        xaxis=dict(tickangle=90),
    )
    fig.show()
    return fig


# =============================================
#   Worst-tail Bad Rate Plots
# =============================================
def plot_top_worst_tail_bad_rates(
    rank_table,
    top_n=20,
    bad_metric = "BR Bal" # "BR Vol" or "BR Bal"
    use_cleaned_name=False,
    show_pct=True
):
    """
    Plot the top N features by worst-tail bad rate using Plotly.
    """

    if rank_table is None or rank_table.empty:
        raise ValueError("rank_table is empty.")

    label_col = "cleaned_name" if use_cleaned_name else "feature"

    dfp = rank_table.head(top_n).copy()
    dfp = dfp.sort_values("tail_bad_rate", ascending=True)

    # Format display values
    if show_pct:
        dfp["bad_rate_display"] = (dfp["tail_bad_rate"] * 100).round(2).astype(str) + "%"
        x_vals = dfp["tail_bad_rate"] * 100
        x_label = "Bad rate in worst tail (%)"
    else:
        dfp["bad_rate_display"] = dfp["tail_bad_rate"].round(4).astype(str)
        x_vals = dfp["tail_bad_rate"]
        x_label = "Bad rate in worst tail"

    fig = px.bar(
        dfp,
        x=x_vals,
        y=dfp[label_col].astype(str),
        orientation="h",
        text="bad_rate_display",
        hover_data={
            "tail_n": True,              # volume
            "tail_bad_n": True,          # bad volume
            "cutoff": True,
            "definition": True,
            "direction_used": True,
            "tail_side": True,
        },
        labels={
            "x": x_label,
            "y": "Feature"
        },
        title=f"Top {min(top_n, len(rank_table))} features by worst-tail {bad_metric}"
        
    )

    # Make labels look nice
    fig.update_traces(textposition="outside")

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_tickformat=".2f" if show_pct else ".3f",
        margin=dict(l=200, r=50, t=60, b=50),
        height=max(500, 35 * len(dfp))
    )

    fig.show()


# =============================================
#      IV Correlation Heatmap
# =============================================
def top_iv_correlation(
    iv_table: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_x: int,
    figsize: Tuple[int, int] = (8, 6),
    print_num: bool = True,
):
    """
    Plot the correlation heatmap for the top_x features by IV.

    Requires: plotly
    Returns the Plotly Figure.
    """
    if px is None:
        raise ImportError("plotly is required for top_iv_correlation (plotly.express not found).")

    if top_x <= 0:
        raise ValueError("top_x must be > 0")

    top_features = (
        iv_table.sort_values(by="Information Value", ascending=False)
        .head(top_x)["Variable"]
        .astype(str)
        .tolist()
    )
    top_features = [f for f in top_features if f in corr_matrix.columns]

    if len(top_features) == 0:
        raise ValueError("None of top features by IV found in correlation matrix.")

    filt = corr_matrix.loc[top_features, top_features]

    fig = px.imshow(
        filt,
        text_auto=".2f" if print_num else False,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"Correlation Heatmap (top {len(top_features)} by IV)",
    )
    fig.update_layout(
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        xaxis=dict(tickangle=90),
    )
    fig.show()
    return fig


# =============================================
#      IV + Correlation Filters
# =============================================
def filter_iv_corr(
    iv_table: pd.DataFrame,
    highly_correlated_pairs: pd.DataFrame,
    iv_threshold: float,
) -> List[str]:
    """
    Filter features based on IV and correlation.
    For each highly correlated pair, drop the one with lower IV.
    """
    iv_dict: Dict[str, float] = iv_table.set_index("Variable")["Information Value"].to_dict()

    drop_variables: Set[str] = set()

    for _, row in highly_correlated_pairs.iterrows():
        var1, var2 = str(row["Variable 1"]), str(row["Variable 2"])
        if var1 not in iv_dict or var2 not in iv_dict:
            continue

        if iv_dict[var1] >= iv_dict[var2]:
            drop_variables.add(var2)
        else:
            drop_variables.add(var1)

    final_variables = sorted(
        [v for v, iv in iv_dict.items() if v not in drop_variables and iv > iv_threshold],
        key=lambda v: iv_dict[v],
        reverse=True,
    )
    return final_variables

# =============================================
#      Count survived features after IV & Correlation Filters
# =============================================
def corr_survived(
    iv_table: pd.DataFrame,
    highly_correlated_pairs: pd.DataFrame,
    num_list: Sequence[str],
    cat_list: Sequence[str],
    iv_threshold: float,
) -> Tuple[int, int, int]:
    """
    Count the number of features survived after IV & Correlation filters.

    Returns: (num_survived, cat_survived, total_survived)
    """
    final_variables = filter_iv_corr(iv_table, highly_correlated_pairs, iv_threshold)

    num_set, cat_set = set(num_list), set(cat_list)
    num_sur = sum(v in num_set for v in final_variables)
    cat_sur = sum(v in cat_set for v in final_variables)
    return num_sur, cat_sur, num_sur + cat_sur



# =============================================
#     Numeric distribution plots
# =============================================

def distribution_by_decile_logic(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    bins: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Logic checks for numeric features + plots bad-rate by deciles.

    Returns the grouped summary table (or None if skipped).
    """
    _require_columns(data, [target_variable, bad_flag, bal_variable, bad_bal_variable])

    df = data[[target_variable, bad_flag, bal_variable, bad_bal_variable]].copy()
    df = df.dropna(subset=[target_variable])

    if df[target_variable].nunique() < 3:
        return None

    baseline_br_vol = (df[bad_flag].fillna(0) > 0).mean()
    denom_bal = df[bal_variable].sum()
    baseline_br_bal = (df[bad_bal_variable].sum() / denom_bal) if denom_bal != 0 else 0.0

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    min_value, max_value, direction, definition = -np.inf, np.inf, 0.0, "N/A"

    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_\
            ar].iloc[0]
            min_value = float(row.get("Valid Min", -np.inf)) if pd.notnull(row.get("Valid Min", np.nan)) else -np.inf
            max_value = float(row.get("Valid Max", np.inf)) if pd.notnull(row.get("Valid Max", np.nan)) else np.inf
            direction = float(row.get("Direction", 0.0)) if pd.notnull(row.get("Direction", np.nan)) else 0.0
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"

    # out-of-range as "missing group"
    oob = df[(df[target_variable] < min_value) | (df[target_variable] > max_value)]
    if not oob.empty:
        br_oob = (oob[bad_flag].fillna(0) > 0).mean() * 100
        pct_oob = (len(oob) / len(df)) * 100
        over_baseline = (br_oob / (baseline_br_vol * 100) * 100) if baseline_br_vol > 0 else np.nan
        print(
            f"{target_variable}: Out-of-range Vol%={pct_oob:.2f}%, "
            f"BR(out-of-range)={br_oob:.2f}%, "
            f"over baseline={over_baseline:.2f}%"
        )

    df = df[(df[target_variable] >= min_value) & (df[target_variable] <= max_value)].copy()
    if len(df) < 5:
        return None

    try:
        df["decile"] = pd.qcut(df[target_variable], q=bins, duplicates="drop")
    except Exception:
        df["decile"] = pd.cut(df[target_variable], bins=min(bins, max(2, df[target_variable].nunique())), duplicates="drop")

    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    results = df.groupby("decile", observed=True).agg(
        Total_Volume=("is_bad", "size"),
        Bad_Volume=("is_bad", "sum"),
        Bad_Vol_Rate=("is_bad", "mean"),
        Total_Balance=(bal_variable, "sum"),
        Bad_Balance=(bad_bal_variable, "sum"),
    )
    results["Bad Vol%"] = results["Bad_Vol_Rate"] * 100
    results["Bad Bal%"] = np.where(
        results["Total_Balance"] != 0,
        (results["Bad_Balance"] / results["Total_Balance"]) * 100,
        np.nan,
    )

    # direction sanity check (optional)
    smallest_group_br = results["Bad Vol%"].iloc[0]
    largest_group_br = results["Bad Vol%"].iloc[-1]
    if direction == -1 and smallest_group_br > largest_group_br:
        print(f"{target_variable}: Counterintuitive - lower value has higher bad rate (direction=-1).")
    elif direction == 1 and smallest_group_br < largest_group_br:
        print(f"{target_variable}: Counterintuitive - larger value has higher bad rate (direction=1).")

    # --- plots ---
    results[["Bad Vol%"]].plot.bar()
    plt.axhline(y=baseline_br_vol * 100, linestyle="dotted", color="red", label="Baseline BR Vol")
    plt.ylabel("Bad Vol%")
    plt.title(f"Bad Vol% by {target_variable}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    results[["Bad Bal%"]].plot.bar()
    plt.axhline(y=baseline_br_bal * 100, linestyle="dotted", color="red", label="Baseline BR Bal")
    plt.ylabel("Bad Bal%")
    plt.title(f"Bad Bal% by {target_variable}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if definition != "N/A":
        print(f"{target_variable}: {definition}")

    return results.reset_index()

# =============================================
#      Categorical distribution plots
# =============================================
def distribution_by_group(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    max_categories: int = 30,
) -> Optional[pd.DataFrame]:
    """
    Plot volume + bad-rate and balance + bad-balance-rate for a categorical variable.

    Returns the grouped summary table (or None if skipped).
    """
    _require_columns(data, [target_variable, bad_flag, bal_variable, bad_bal_variable])

    df = data[[target_variable, bad_flag, bal_variable, bad_bal_variable]].copy()
    df[target_variable] = df[target_variable].astype("object").fillna("__MISSING__")
    df["volume"] = 1
    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    group = df.groupby(target_variable, dropna=False).agg(
        Total_Volume=("volume", "sum"),
        Bad_Volume=("is_bad", "sum"),
        Total_Balance=(bal_variable, "sum"),
        Bad_Balance=(bad_bal_variable, "sum"),
    )
    group["Bad Vol%"] = np.where(group["Total_Volume"] != 0, (group["Bad_Volume"] / group["Total_Volume"]) * 100, np.nan)
    group["Bad Bal%"] = np.where(group["Total_Balance"] != 0, (group["Bad_Balance"] / group["Total_Balance"]) * 100, np.nan)

    # Keep plots readable
    if group.shape[0] > max_categories:
        group = group.sort_values("Total_Volume", ascending=False).head(max_categories)

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_var].iloc[0]
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"
            if definition != "N/A":
                print(f"{target_variable}: {definition}")

    baseline_br_vol = df["is_bad"].mean() * 100
    denom_bal = df[bal_variable].sum()
    baseline_br_bal = (df[bad_bal_variable].sum() / denom_bal * 100) if denom_bal != 0 else 0.0

    # --- Bad Volume plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(group.index.astype(str), group["Total_Volume"])
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel("Total Volume")
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(group.index.astype(str), group["Bad Vol%"], marker="o")
    ax2.set_ylabel("Bad Vol%")
    ax2.axhline(y=baseline_br_vol, linestyle="dotted", label="Baseline")
    fig.suptitle(f"{target_variable} - Bad Volume Distribution")
    fig.tight_layout()
    plt.show()

    # --- Bad Balance plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(group.index.astype(str), group["Total_Balance"])
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel("Total Balance")
    ax1.tick_params(axis="x", rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(group.index.astype(str), group["Bad Bal%"], marker="o")
    ax2.set_ylabel("Bad Bal%")
    ax2.axhline(y=baseline_br_bal, linestyle="dotted", label="Baseline")
    fig.suptitle(f"{target_variable} - Bad Balance Distribution")
    fig.tight_layout()
    plt.show()

    return group.reset_index()


def distribution_plot_num_cat(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    bal_variable: str,
    bad_bal_variable: str,
    data_dictionary: Optional[pd.DataFrame] = None,
):
    """
    Route to numeric/categorical distribution plot.
    """
    if _is_categorical(data[target_variable]):
        return distribution_by_group(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)
    return distribution_by_decile_logic(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)


# =============================================
#    Good vs. Bad Distribution
# =============================================

def good_bad_distribution(
    data: pd.DataFrame,
    target_variable: str,
    bad_flag: str,
    data_dictionary: Optional[pd.DataFrame] = None,
    narrow: bool = False,
    bins: int = 10,
):
    """
    Plot the Good vs. Bad distribution for a single variable.

    Notes:
    - For numeric variables uses seaborn if available; otherwise falls back to matplotlib hist.
    - For categorical variables uses countplot-style bar charts if seaborn is available; otherwise matplotlib.
    """
    _require_columns(data, [target_variable, bad_flag])

    cleaned_var = target_variable.split("_")[-1] if "_" in target_variable else target_variable
    definition = "N/A"
    valid_min = -np.inf

    if data_dictionary is not None and "Variable" in data_dictionary.columns:
        if cleaned_var in set(data_dictionary["Variable"].astype(str).values):
            row = data_dictionary.loc[data_dictionary["Variable"].astype(str) == cleaned_var].iloc[0]
            definition = row.get("Definition", "N/A") if pd.notnull(row.get("Definition", np.nan)) else "N/A"
            vm = row.get("Valid Min", np.nan)
            valid_min = float(vm) if pd.notnull(vm) else -np.inf
            if definition != "N/A":
                print(f"{target_variable}: {definition}")

    df = data[[target_variable, bad_flag]].copy()
    df["is_bad"] = (df[bad_flag].fillna(0) > 0).astype(int)

    # Filter extremes
    if _is_numeric(df[target_variable]):
        if narrow:
            df = df[df[target_variable] >= valid_min].copy()
        else:
            threshold = df[target_variable].quantile(0.999)
            df = df[(df[target_variable] >= valid_min) & (df[target_variable] < threshold)].copy()

    if df.empty:
        return

    # Numeric
    if _is_numeric(df[target_variable]):
        good = df.loc[df["is_bad"] == 0, target_variable].dropna()
        bad = df.loc[df["is_bad"] == 1, target_variable].dropna()

        fig, ax1 = plt.subplots(figsize=(8, 4))
        if sns is not None:
            sns.histplot(good, kde=True, bins=bins, ax=ax1, alpha=0.5, label="Good")
            ax2 = ax1.twinx()
            sns.histplot(bad, kde=True, bins=bins, ax=ax2, alpha=0.5, label="Bad")
            ax1.set_ylabel("Good Count")
            ax2.set_ylabel("Bad Count")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax1.hist(good, bins=bins, alpha=0.5, label="Good")
            ax1.hist(bad, bins=bins, alpha=0.5, label="Bad")
            ax1.set_ylabel("Count")
            ax1.legend()

        ax1.set_xlabel(target_variable)
        ax1.set_title(f"Distribution of {target_variable} by {bad_flag}")
        fig.tight_layout()
        plt.show()
        return

    # Categorical
    df[target_variable] = df[target_variable].astype("object").fillna("__MISSING__")
    flags = sorted(df["is_bad"].unique().tolist())
    n = len(flags)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, flag in enumerate(flags):
        sub = df[df["is_bad"] == flag]
        if sns is not None:
            sns.countplot(data=sub, x=target_variable, ax=axes[i])
        else:
            counts = sub[target_variable].value_counts()
            axes[i].bar(counts.index.astype(str), counts.values)
        axes[i].set_title(f"{bad_flag}={flag} - {target_variable}")
        axes[i].tick_params(axis="x", rotation=90)
        axes[i].set_xlabel(target_variable)
        axes[i].set_ylabel("Count" if i == 0 else "")

    fig.tight_layout()
    plt.show()


# =============================================
#    Funnel plot - feature selection process
# =============================================

def funnel_feature_selection(
    original_feature: int,
    iv_survived_n: int,
    corr_survived_n: int,
    iv_corr_survived_n: int,
    logic_survived_n: int,
    expert_judgement_n: Optional[int] = None,
):
    """
    Funnel plot for feature selection stages.

    Requires: plotly
    Returns Plotly Figure.
    """
    if px is None:
        raise ImportError("plotly is required for funnel_feature_selection (plotly.express not found).")

    funnel_dictionary: Dict[str, int] = {
        "Initial": int(original_feature),
        "Information Value Filter": int(iv_survived_n),
        "Correlation Filter": int(corr_survived_n),
        "IV + Corr Filter": int(iv_corr_survived_n),
        "Logic Checks": int(logic_survived_n),
    }
    if expert_judgement_n is not None:
        funnel_dictionary["Expert Judgement"] = int(expert_judgement_n)

    funnel_data = pd.DataFrame({"Stage": list(funnel_dictionary.keys()), "Feature Number": list(funnel_dictionary.values())})

    fig = px.funnel(
        funnel_data,
        x="Feature Number",
        y="Stage",
        title="Funnel plot for Feature Selection Process",
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(funnel_dictionary.keys()),
            tickfont=dict(size=14),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=600,
    )
    fig.show()
    return fig


