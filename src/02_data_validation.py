"""
02_data_validation.py — Check the quality and integrity of the dataset
=======================================================================
Why this step matters
---------------------
Before drawing any conclusions from data, we must verify that:
  • There are no missing values (NaN) that could silently distort results.
  • The numeric ranges are plausible (e.g. GDP growth shouldn't be 500%).
  • There are no duplicate year entries.
  • The distributions look reasonable (check via basic statistics).

Your teacher may ask: "How do you know your data is correct?"
This module gives you a documented, reproducible answer.
"""

import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RES_DIR, ensure_dirs


# ── Expected plausible ranges (based on India's historical record) ─────────────
EXPECTED_RANGES = {
    "GDP_Growth":      (-10,  15),   # India never saw below -10 or above 15 (modern era)
    "Govt_Exp_GDP":    (  8,  25),   # Central govt exp as % GDP typically 8–25%
    "Fiscal_Deficit":  (  0,  12),   # Deficit as % GDP
    "Revenue_Exp_GDP": (  5,  20),
    "Capital_Exp_GDP": (  1,   8),
}


def validate(df: pd.DataFrame) -> dict:
    """
    Run a battery of data quality checks and return a results dictionary.

    Parameters
    ----------
    df : DataFrame produced by 01_data_loader.load_data()

    Returns
    -------
    dict with keys: 'passed', 'warnings', 'summary_stats'
    """
    ensure_dirs()
    warnings = []
    passed   = []

    print("=" * 60)
    print("  DATA VALIDATION REPORT")
    print("=" * 60)

    # ── Check 1: No missing values ─────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.sum() == 0:
        passed.append("No missing values found.")
        print("✅  No missing values.")
    else:
        warnings.append(f"Missing values detected:\n{missing[missing > 0]}")
        print(f"⚠️  Missing values:\n{missing[missing > 0]}")

    # ── Check 2: No duplicate years ────────────────────────────────────────────
    dupes = df["Year"].duplicated().sum()
    if dupes == 0:
        passed.append("No duplicate year entries.")
        print("✅  No duplicate years.")
    else:
        warnings.append(f"{dupes} duplicate year(s) found.")
        print(f"⚠️  {dupes} duplicate year(s).")

    # ── Check 3: Years are consecutive ────────────────────────────────────────
    years = df["Year"].sort_values().values
    gaps  = [years[i+1] - years[i] for i in range(len(years)-1) if years[i+1] - years[i] != 1]
    if not gaps:
        passed.append("Years are consecutive (no gaps).")
        print("✅  Year sequence is continuous — no gaps.")
    else:
        warnings.append(f"Gaps in year sequence: {gaps}")
        print(f"⚠️  Year gaps detected: {gaps}")

    # ── Check 4: Numeric ranges are plausible ─────────────────────────────────
    range_ok = True
    for col, (lo, hi) in EXPECTED_RANGES.items():
        out_of_range = df[(df[col] < lo) | (df[col] > hi)][["Year", col]]
        if not out_of_range.empty:
            range_ok = False
            warnings.append(
                f"'{col}' has {len(out_of_range)} out-of-range value(s): "
                f"{out_of_range.to_dict('records')}"
            )
            print(f"⚠️  '{col}' out-of-range rows:\n{out_of_range}\n")
    if range_ok:
        passed.append("All numeric columns within expected historical ranges.")
        print("✅  All numeric columns within expected plausible ranges.")

    # ── Check 5: Growth_Period balance ────────────────────────────────────────
    counts = df["Growth_Period"].value_counts()
    diff   = abs(counts.get("High", 0) - counts.get("Low", 0))
    if diff <= 2:
        passed.append(f"Growth periods roughly balanced: {counts.to_dict()}")
        print(f"✅  Growth period split: {counts.to_dict()} (balanced ✔)")
    else:
        warnings.append(f"Growth period split imbalanced: {counts.to_dict()}")
        print(f"⚠️  Growth period split: {counts.to_dict()} — imbalanced!")

    # ── Summary statistics ─────────────────────────────────────────────────────
    numeric_cols = ["GDP_Growth", "Govt_Exp_GDP", "Fiscal_Deficit",
                    "Revenue_Exp_GDP", "Capital_Exp_GDP"]
    summary = df[numeric_cols].describe().round(2)

    print("\n── Summary Statistics ──────────────────────────────────────────")
    print(summary.to_string())

    # ── Correlation quick-look ─────────────────────────────────────────────────
    corr = df[numeric_cols].corr().round(3)
    print("\n── Pairwise Correlations ───────────────────────────────────────")
    print(corr.to_string())

    # ── Save validation report ─────────────────────────────────────────────────
    report_path = os.path.join(RES_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATA VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("PASSED CHECKS:\n")
        for p in passed:
            f.write(f"  ✅  {p}\n")
        f.write("\nWARNINGS:\n")
        if warnings:
            for w in warnings:
                f.write(f"  ⚠️  {w}\n")
        else:
            f.write("  (none)\n")
        f.write("\nSUMMARY STATISTICS:\n")
        f.write(summary.to_string())
        f.write("\n\nCORRELATION MATRIX:\n")
        f.write(corr.to_string())

    print(f"\n[Validation] Report saved to: {report_path}")
    print("=" * 60)

    return {
        "passed":        passed,
        "warnings":      warnings,
        "summary_stats": summary,
        "correlation":   corr,
    }


if __name__ == "__main__":
    # Run standalone: python src/02_data_validation.py
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "data_loader",
        os.path.join(os.path.dirname(__file__), "01_data_loader.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    df = _mod.load_data()
    validate(df)
