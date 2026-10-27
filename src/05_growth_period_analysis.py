"""
05_growth_period_analysis.py — Compare High vs Low Growth Periods
==================================================================
This module answers our core research question:

  "Does government expenditure have a DIFFERENT relationship with GDP growth
   during high-growth vs low-growth periods?"

We use THREE analytical tools:

  1. Descriptive statistics — Mean, std, min/max of each variable for each period
  2. t-test               — Is the difference in means statistically significant?
  3. Box plots + bar chart— Visual comparison of distributions

STATISTICAL TOOL: Independent Samples t-test
---------------------------------------------
The t-test asks: "Could the difference between group means be just random chance?"
  • t-statistic : how many standard errors apart the means are
  • p-value     : probability of observing this large a difference by chance
    – p < 0.05 → statistically significant difference (the groups ARE different)
    – p ≥ 0.05 → the difference could be random chance

IMPORTANT CAVEAT
----------------
Even if we find a significant difference, this is CORRELATION, not CAUSATION.
We cannot claim government spending CAUSED growth to change without a more
sophisticated econometric design (e.g., instrumental variables, VAR models).
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FIG_DIR, RES_DIR, COLOR_HIGH, COLOR_LOW, ensure_dirs

sns.set_theme(style="whitegrid", font_scale=1.1)


def compare_periods(df: pd.DataFrame) -> dict:
    """
    Compare economic variables across High-Growth and Low-Growth periods.

    Parameters
    ----------
    df : DataFrame from 01_data_loader.load_data()

    Returns
    -------
    dict with keys 'stats', 'ttest_results', 'figure_paths'
    """
    ensure_dirs()

    high_df = df[df["Growth_Period"] == "High"]
    low_df  = df[df["Growth_Period"] == "Low"]

    numeric_cols = ["GDP_Growth", "Govt_Exp_GDP", "Fiscal_Deficit",
                    "Revenue_Exp_GDP", "Capital_Exp_GDP"]

    # ── 1. Descriptive statistics per period ──────────────────────────────────
    stats_high = high_df[numeric_cols].describe().round(3)
    stats_low  = low_df[numeric_cols].describe().round(3)

    print("=" * 65)
    print("  PERIOD COMPARISON REPORT")
    print("=" * 65)
    print(f"\nHigh-Growth Period ({len(high_df)} years — GDP growth ≥ median):")
    print(stats_high.to_string())
    print(f"\nLow-Growth Period ({len(low_df)} years — GDP growth < median):")
    print(stats_low.to_string())

    # ── 2. t-tests for each variable ─────────────────────────────────────────
    print("\n── Independent Samples t-test Results ──────────────────────────────")
    ttest_results = {}
    for col in numeric_cols:
        t_stat, p_val = stats.ttest_ind(
            high_df[col].dropna(),
            low_df[col].dropna(),
            equal_var=False   # Welch's t-test — does NOT assume equal variances
        )
        sig = "SIGNIFICANT ✅" if p_val < 0.05 else "Not significant ❌"
        mean_h = high_df[col].mean()
        mean_l = low_df[col].mean()
        diff   = mean_h - mean_l

        print(f"  {col:20s}  High={mean_h:.2f}  Low={mean_l:.2f}  "
              f"Δ={diff:+.2f}  t={t_stat:.3f}  p={p_val:.4f}  [{sig}]")

        ttest_results[col] = {
            "mean_high": mean_h,
            "mean_low":  mean_l,
            "diff":      diff,
            "t_stat":    t_stat,
            "p_val":     p_val,
            "significant": p_val < 0.05,
        }

    # ── 3. Visualisations ──────────────────────────────────────────────────────
    figure_paths = []

    # 3a. Box plots side by side
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    col_labels = {
        "GDP_Growth":      "GDP Growth (%)",
        "Govt_Exp_GDP":    "Govt Exp (% GDP)",
        "Fiscal_Deficit":  "Fiscal Deficit (% GDP)",
        "Revenue_Exp_GDP": "Revenue Exp (% GDP)",
        "Capital_Exp_GDP": "Capital Exp (% GDP)",
    }

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        plot_data = df[["Growth_Period", col]].copy()
        plot_data["Growth_Period"] = pd.Categorical(
            plot_data["Growth_Period"], categories=["High", "Low"])

        sns.boxplot(data=plot_data, x="Growth_Period", y=col,
                    palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                    ax=ax, width=0.5)
        sns.stripplot(data=plot_data, x="Growth_Period", y=col,
                      palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                      ax=ax, size=5, alpha=0.6, jitter=True)

        # Significance annotation
        p = ttest_results[col]["p_val"]
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        y_max = df[col].max()
        ax.text(0.5, 1.03, f"p={p:.3f} {star}", transform=ax.transAxes,
                ha="center", fontsize=9, color="black")

        ax.set_title(col_labels[col], fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove the 6th empty subplot
    axes[5].set_visible(False)

    fig.suptitle("Distribution Comparison: High-Growth vs Low-Growth Periods\n"
                 "(India 1992–2023, RBI/MoSPI data)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "06_period_boxplots.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path1)
    print(f"\n[Period Analysis] Boxplots saved: {path1}")

    # 3b. Bar chart of mean Govt Expenditure by growth period
    fig, ax = plt.subplots(figsize=(7, 5))
    periods = ["High-Growth", "Low-Growth"]
    means   = [ttest_results["Govt_Exp_GDP"]["mean_high"],
               ttest_results["Govt_Exp_GDP"]["mean_low"]]
    bars = ax.bar(periods, means, color=[COLOR_HIGH, COLOR_LOW],
                  alpha=0.8, width=0.4, edgecolor="white")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f"{m:.2f}%",
                ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Mean Govt Expenditure (% of GDP)")
    ax.set_title("Average Government Expenditure by Growth Period\n"
                 "(Does the government spend more in low-growth years?)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3)
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "07_mean_exp_by_period.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path2)
    print(f"[Period Analysis] Mean exp bar chart saved: {path2}")

    # ── Save text summary ─────────────────────────────────────────────────────
    summary_path = os.path.join(RES_DIR, "period_comparison.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PERIOD COMPARISON REPORT\n" + "=" * 65 + "\n\n")
        f.write("Independent Samples Welch t-test (High vs Low Growth years)\n\n")
        for col, res in ttest_results.items():
            sig = "SIGNIFICANT" if res["significant"] else "Not significant"
            f.write(
                f"{col:22s}  High={res['mean_high']:.2f}  Low={res['mean_low']:.2f}  "
                f"Δ={res['diff']:+.2f}  t={res['t_stat']:.3f}  "
                f"p={res['p_val']:.4f}  [{sig}]\n"
            )
    print(f"[Period Analysis] Text summary saved: {summary_path}")

    return {
        "stats": {"High": stats_high, "Low": stats_low},
        "ttest": ttest_results,
        "figure_paths": figure_paths,
    }


if __name__ == "__main__":
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("dl", os.path.join(os.path.dirname(__file__), "01_data_loader.py"))
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    df = _m.load_data()
    compare_periods(df)
