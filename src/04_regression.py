"""
04_regression.py — OLS Regression Analysis
============================================
WHAT IS OLS?
------------
Ordinary Least Squares (OLS) is the most common statistical tool for
understanding relationships between variables. It fits a straight line:

    GDP_Growth = β₀ + β₁ × Govt_Exp_GDP + ε

where:
  β₀ = intercept (baseline growth when expenditure = 0, not meaningful alone)
  β₁ = slope (the change in GDP growth for a 1-percentage-point rise in expenditure)
  ε  = error term (unexplained variation)

WHY TWO REGRESSIONS?
--------------------
We fit one model on ALL years and then separate models for HIGH-GROWTH
and LOW-GROWTH periods. If government spending truly has a stronger effect
during recessions or slowdowns (our hypothesis), the β₁ coefficient should
be more negative (or more strongly related) in the Low period model.

KEY OUTPUTS YOUR TEACHER WILL ASK ABOUT
----------------------------------------
  • p-value     : probability of observing this slope by chance alone.
                  If p < 0.05, the effect is statistically significant.
  • R-squared   : proportion of variation in GDP growth explained by the model.
                  Ranges 0–1; higher = better fit.
  • Coefficient : the actual slope β₁. Its sign and magnitude matter.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FIG_DIR, RES_DIR, COLOR_HIGH, COLOR_LOW, COLOR_ALL, ensure_dirs


def _run_ols(df: pd.DataFrame, label: str) -> dict:
    """
    Fit a simple OLS model: GDP_Growth ~ Govt_Exp_GDP

    Returns a dict with regression results and the fitted model object.
    """
    X = sm.add_constant(df["Govt_Exp_GDP"])   # adds the intercept column
    y = df["GDP_Growth"]

    model  = sm.OLS(y, X).fit()
    coef   = model.params["Govt_Exp_GDP"]
    pval   = model.pvalues["Govt_Exp_GDP"]
    r2     = model.rsquared
    n      = int(model.nobs)
    ci_low, ci_high = model.conf_int().loc["Govt_Exp_GDP"]

    sig = "✅ SIGNIFICANT (p < 0.05)" if pval < 0.05 else "❌ NOT significant (p ≥ 0.05)"

    print(f"\n── OLS Results: {label} ({'n=' + str(n)}) ──")
    print(f"   Coefficient (β₁) : {coef:+.4f}")
    print(f"   p-value           : {pval:.4f}  {sig}")
    print(f"   R-squared         : {r2:.4f}")
    print(f"   95% CI            : [{ci_low:.4f}, {ci_high:.4f}]")

    return {
        "label":   label,
        "n":       n,
        "coef":    coef,
        "pval":    pval,
        "r2":      r2,
        "ci":      (ci_low, ci_high),
        "model":   model,
    }


def run_regression(df: pd.DataFrame) -> dict:
    """
    Run three OLS regressions:
      1. Full dataset
      2. High-Growth period only
      3. Low-Growth  period only

    Also generates a side-by-side coefficient comparison plot.

    Returns dict of results for each period.
    """
    ensure_dirs()

    high_df = df[df["Growth_Period"] == "High"].copy()
    low_df  = df[df["Growth_Period"] == "Low" ].copy()

    results = {
        "All":  _run_ols(df,      "All Years (1992–2023)"),
        "High": _run_ols(high_df, "High-Growth Period"),
        "Low":  _run_ols(low_df,  "Low-Growth Period"),
    }

    # ── Save full regression summaries ────────────────────────────────────────
    report_path = os.path.join(RES_DIR, "regression_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for key, res in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"  {res['label']}\n")
            f.write(f"{'='*60}\n")
            f.write(res["model"].summary().as_text())
            f.write("\n")
    print(f"\n[Regression] Full summaries saved to: {report_path}")

    # ── Comparison chart: Coefficients with confidence intervals ──────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    labels  = ["All Years",   "High-Growth", "Low-Growth"]
    coefs   = [results["All"]["coef"], results["High"]["coef"], results["Low"]["coef"]]
    ci_lo   = [results["All"]["ci"][0], results["High"]["ci"][0], results["Low"]["ci"][0]]
    ci_hi   = [results["All"]["ci"][1], results["High"]["ci"][1], results["Low"]["ci"][1]]
    colors  = [COLOR_ALL, COLOR_HIGH, COLOR_LOW]
    pvals   = [results["All"]["pval"], results["High"]["pval"], results["Low"]["pval"]]

    x = np.arange(len(labels))
    bars = ax.bar(x, coefs, color=colors, alpha=0.75, width=0.5)

    # Error bars (95% CI)
    yerr_low  = [c - lo for c, lo in zip(coefs, ci_lo)]
    yerr_high = [hi - c for c, hi in zip(coefs, ci_hi)]
    ax.errorbar(x, coefs, yerr=[yerr_low, yerr_high],
                fmt="none", color="black", capsize=6, linewidth=1.5)

    # Significance stars
    for i, (bar, p) in enumerate(zip(bars, pvals)):
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * (1 if bar.get_height() >= 0 else -1),
                star, ha="center", va="bottom", fontsize=12)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("OLS Coefficient β₁\n(change in GDP growth per 1% rise in Govt Exp)")
    ax.set_title("Effect of Govt Expenditure on GDP Growth\n"
                 "by Growth Period (OLS with 95% CI)",
                 fontsize=13, fontweight="bold")

    # R² annotation inside bars
    for i, (bar, res) in enumerate(zip(bars, [results["All"], results["High"], results["Low"]])):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"R²={res['r2']:.2f}\np={res['pval']:.3f}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "05_regression_coefficients.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Regression] Coefficient chart saved to: {path}")

    return results


if __name__ == "__main__":
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("dl", os.path.join(os.path.dirname(__file__), "01_data_loader.py"))
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    df = _m.load_data()
    results = run_regression(df)
