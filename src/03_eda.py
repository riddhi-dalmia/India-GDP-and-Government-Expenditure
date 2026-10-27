"""
03_eda.py — Exploratory Data Analysis
======================================
We generate four publication-quality charts:

  1. GDP Growth over time (line chart)   — see the trend and shocks
  2. Government Expenditure over time    — see spending trajectory
  3. Scatter: Expenditure vs GDP Growth  — coloured by growth period
  4. Correlation heatmap                 — quick view of all relationships

Each chart is saved to outputs/figures/ and returned for use in Streamlit.

READING THE CHARTS
------------------
• A downward slope in Chart 3 would mean higher spending → lower growth
  (the "crowding out" hypothesis).
• An upward slope would support the Keynesian multiplier view.
• No slope would mean spending and growth are unrelated in the data.
We let the data decide — we do NOT pre-assume any direction.
"""

import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FIG_DIR, COLOR_HIGH, COLOR_LOW, COLOR_ALL, ensure_dirs

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Helper: consistent figure saving ─────────────────────────────────────────
def _save(fig, name: str) -> str:
    ensure_dirs()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: GDP Growth Rate over Time
# ─────────────────────────────────────────────────────────────────────────────
def plot_gdp_growth(df: pd.DataFrame) -> str:
    """
    Line chart of India's real GDP growth (%) from 1992 to 2023.
    Shaded band marks the median growth threshold used to split periods.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["Year"], df["GDP_Growth"], color=COLOR_ALL,
            linewidth=2.5, marker="o", markersize=5, label="GDP Growth (%)")

    # Mark recession/slow years in red
    low_df = df[df["Growth_Period"] == "Low"]
    ax.scatter(low_df["Year"], low_df["GDP_Growth"],
               color=COLOR_LOW, zorder=5, s=70, label="Low-Growth Year")

    # Median threshold line
    median_g = df["GDP_Growth"].median()
    ax.axhline(median_g, color="gray", linestyle="--", linewidth=1.2,
               label=f"Median threshold ({median_g:.1f}%)")

    # Annotate notable events
    annotations = {
        1998: "Asian\nCrisis",
        2003: "Dotcom\nSlump",
        2009: "GFC\nSpillover",
        2021: "COVID-19",
    }
    for yr, label in annotations.items():
        row = df[df["Year"] == yr]
        if not row.empty:
            ax.annotate(label,
                        xy=(yr, row["GDP_Growth"].values[0]),
                        xytext=(yr + 0.3, row["GDP_Growth"].values[0] - 1.5),
                        fontsize=8, color="gray",
                        arrowprops=dict(arrowstyle="-", color="lightgray"))

    ax.set_title("India: Real GDP Growth Rate (1992–2023)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year (fiscal year ending)")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return _save(fig, "01_gdp_growth_over_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Government Expenditure (Revenue + Capital) over Time
# ─────────────────────────────────────────────────────────────────────────────
def plot_govt_expenditure(df: pd.DataFrame) -> str:
    """
    Stacked area chart showing how Revenue and Capital Expenditure
    (as % of GDP) have changed over time.
    Helps visualise the composition of government spending.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(df["Year"], 0, df["Revenue_Exp_GDP"],
                    alpha=0.6, color="#1976D2", label="Revenue Expenditure (% GDP)")
    ax.fill_between(df["Year"], df["Revenue_Exp_GDP"],
                    df["Revenue_Exp_GDP"] + df["Capital_Exp_GDP"],
                    alpha=0.7, color="#FF9800", label="Capital Expenditure (% GDP)")

    ax.plot(df["Year"], df["Govt_Exp_GDP"],
            color="black", linewidth=2, linestyle="--",
            label="Total Expenditure (% GDP)")

    ax.set_title("India: Composition of Central Govt Expenditure (1992–2023)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year (fiscal year ending)")
    ax.set_ylabel("Expenditure as % of GDP")
    ax.legend(loc="upper right")
    fig.tight_layout()

    return _save(fig, "02_govt_expenditure_over_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Scatter — Govt Expenditure vs GDP Growth (by period)
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter_exp_vs_growth(df: pd.DataFrame) -> str:
    """
    Scatter plot of Government Expenditure (% GDP) on the X-axis
    versus GDP Growth (%) on the Y-axis.

    Each point is coloured by growth period. A regression line is fitted
    separately for each period so we can visually compare slopes.

    KEY INSIGHT: If the red (low-growth) line is steeper than the blue
    (high-growth) line, it suggests that an additional unit of government
    spending has a larger growth impact during low periods — which is
    what our hypothesis predicts. If slopes are similar, the hypothesis
    is not supported by the data.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"High": COLOR_HIGH, "Low": COLOR_LOW}
    for period, grp in df.groupby("Growth_Period"):
        ax.scatter(grp["Govt_Exp_GDP"], grp["GDP_Growth"],
                   color=colors[period], label=f"{period}-Growth Years",
                   s=80, alpha=0.85, edgecolors="white", linewidth=0.5)

        # Fit and draw a linear regression line for this group
        import numpy as np
        m, b = np.polyfit(grp["Govt_Exp_GDP"], grp["GDP_Growth"], 1)
        x_line = pd.Series(sorted(grp["Govt_Exp_GDP"]))
        ax.plot(x_line, m * x_line + b, color=colors[period],
                linewidth=2, linestyle="--",
                label=f"{period} trend (slope={m:.2f})")

    ax.set_title("Govt Expenditure vs GDP Growth by Growth Period\n"
                 "(India 1992–2023, RBI/MoSPI data)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Central Govt Expenditure (% of GDP)")
    ax.set_ylabel("Real GDP Growth (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    return _save(fig, "03_scatter_exp_vs_growth.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Heatmap of Pearson correlation coefficients between all numeric variables.

    Pearson correlation ranges from -1 to +1:
      • +1 = perfect positive relationship
      •  0 = no linear relationship
      • -1 = perfect negative relationship

    This is a quick diagnostic — it does NOT prove causation.
    """
    numeric_cols = ["GDP_Growth", "Govt_Exp_GDP", "Fiscal_Deficit",
                    "Revenue_Exp_GDP", "Capital_Exp_GDP"]
    corr = df[numeric_cols].corr()

    # Rename for nicer display
    labels = {
        "GDP_Growth":      "GDP Growth",
        "Govt_Exp_GDP":    "Govt Exp / GDP",
        "Fiscal_Deficit":  "Fiscal Deficit / GDP",
        "Revenue_Exp_GDP": "Revenue Exp / GDP",
        "Capital_Exp_GDP": "Capital Exp / GDP",
    }
    corr = corr.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap — India Economic Variables (1992–2023)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "04_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all EDA charts
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> dict:
    """Generate all EDA figures and return their file paths."""
    paths = {
        "gdp_growth":   plot_gdp_growth(df),
        "govt_exp":     plot_govt_expenditure(df),
        "scatter":      plot_scatter_exp_vs_growth(df),
        "heatmap":      plot_correlation_heatmap(df),
    }
    return paths


if __name__ == "__main__":
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("dl", os.path.join(os.path.dirname(__file__), "01_data_loader.py"))
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    df = _m.load_data()
    run_eda(df)
    print("All EDA charts saved.")
