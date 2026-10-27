"""
06_ml_models.py — GDP Growth Prediction: Linear Regression & Random Forest
===========================================================================
WHY MACHINE LEARNING?
----------------------
OLS regression (Module 04) tests *whether* a relationship exists and
quantifies it. Machine learning goes further: can we *predict* GDP growth
from spending indicators with reasonable accuracy?

We train two models:
  1. Linear Regression  — simple, interpretable, baseline model
  2. Random Forest      — captures non-linear patterns, robust to outliers

CRITICAL: TIME-ORDERED EVALUATION
-----------------------------------
For time-series data, we must NEVER randomly split train/test sets.
Why? Because that would let the model "see the future" — e.g., training
on 2010 data to predict 1998 growth. This would give artificially good
results that don't represent real forecasting.

Instead, we use Walk-Forward (Expanding Window) Cross-Validation:
  • Fold 1: Train on years 1–10, test on year 11
  • Fold 2: Train on years 1–11, test on year 12
  • ...and so on.
This simulates a real-world scenario where you only know the past.

FEATURES USED
-------------
  X1: Govt_Exp_GDP      — total central govt expenditure (% GDP)
  X2: Fiscal_Deficit    — fiscal deficit (% GDP)
  X3: Revenue_Exp_GDP   — revenue expenditure (% GDP)
  X4: Capital_Exp_GDP   — capital expenditure (% GDP)
  X5: GDP_Growth_Lag1   — GDP growth in the *previous* year (momentum)

METRICS
-------
  • MAE  (Mean Absolute Error)      — average prediction error in % points
  • RMSE (Root Mean Squared Error)  — penalises large errors more
  • R²   (R-squared, if > 0, good)  — proportion of variance explained
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FIG_DIR, RES_DIR, COLOR_HIGH, COLOR_LOW, COLOR_ALL, ensure_dirs

sns.set_theme(style="whitegrid", font_scale=1.1)

FEATURES = [
    "Govt_Exp_GDP",
    "Fiscal_Deficit",
    "Revenue_Exp_GDP",
    "Capital_Exp_GDP",
    "GDP_Growth_Lag1",   # lagged GDP growth (momentum feature)
]
TARGET = "GDP_Growth"

# Minimum training window before we start predicting
MIN_TRAIN = 10


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the lag feature and drop the first row (which has no lag).
    We sort by year to ensure time order is correct.
    """
    df = df.sort_values("Year").copy()
    df["GDP_Growth_Lag1"] = df["GDP_Growth"].shift(1)
    df = df.dropna(subset=["GDP_Growth_Lag1"]).reset_index(drop=True)
    return df


def _walk_forward_cv(df: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    Returns a DataFrame with columns:
      Year, Actual, Predicted, Error, AbsError, Growth_Period
    """
    results = []
    df = df.reset_index(drop=True)

    for test_idx in range(MIN_TRAIN, len(df)):
        train = df.iloc[:test_idx]
        test  = df.iloc[[test_idx]]

        X_train = train[FEATURES].values
        y_train = train[TARGET].values
        X_test  = test[FEATURES].values
        y_actual = test[TARGET].values[0]

        # Scale features (important for Linear Regression; harmless for RF)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)[0]

        results.append({
            "Year":          test["Year"].values[0],
            "Actual":        y_actual,
            "Predicted":     round(y_pred, 3),
            "Error":         round(y_pred - y_actual, 3),
            "AbsError":      round(abs(y_pred - y_actual), 3),
            "Growth_Period": test["Growth_Period"].values[0],
        })

    return pd.DataFrame(results)


def _compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, R² from walk-forward results."""
    y_true = results_df["Actual"]
    y_pred = results_df["Predicted"]
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
        "n":    len(results_df),
    }


def run_ml_models(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression and Random Forest using walk-forward CV.
    Generate comparison plots. Return results dict.
    """
    ensure_dirs()

    df_feat = _prepare_features(df)

    # ── Model definitions ─────────────────────────────────────────────────────
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
                                n_estimators=100,
                                max_depth=3,      # keep shallow to avoid overfitting on small data
                                random_state=42,
                             ),
    }

    all_results = {}
    all_metrics = {}

    for name, model in models.items():
        print(f"\n── Walk-Forward CV: {name} ──────────────────────────────")
        res_df  = _walk_forward_cv(df_feat, model, name)
        metrics = _compute_metrics(res_df)
        all_results[name] = res_df
        all_metrics[name] = metrics

        print(f"   MAE  : {metrics['MAE']:.4f} percentage points")
        print(f"   RMSE : {metrics['RMSE']:.4f} percentage points")
        print(f"   R²   : {metrics['R2']:.4f}")
        print(f"   (n={metrics['n']} walk-forward predictions)")

        # Save per-model CSV
        csv_path = os.path.join(RES_DIR, f"predictions_{name.replace(' ', '_').lower()}.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"   Predictions saved to: {csv_path}")

    # ── Chart 1: Actual vs Predicted for both models ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    colors_model = {"Linear Regression": "#7B1FA2", "Random Forest": "#E65100"}

    for ax, (name, res_df) in zip(axes, all_results.items()):
        ax.plot(res_df["Year"], res_df["Actual"],
                color=COLOR_ALL, linewidth=2.5, marker="o", markersize=5,
                label="Actual GDP Growth")
        ax.plot(res_df["Year"], res_df["Predicted"],
                color=colors_model[name], linewidth=2, linestyle="--",
                marker="s", markersize=5, label=f"{name} Forecast")

        # Shade by growth period
        for _, row in res_df.iterrows():
            ax.axvspan(row["Year"] - 0.4, row["Year"] + 0.4,
                       alpha=0.05,
                       color=COLOR_HIGH if row["Growth_Period"] == "High" else COLOR_LOW)

        metrics = all_metrics[name]
        ax.set_title(f"{name} | MAE={metrics['MAE']:.2f}pp  RMSE={metrics['RMSE']:.2f}pp  R²={metrics['R2']:.2f}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("GDP Growth (%)")
        ax.legend(loc="upper right")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    axes[1].set_xlabel("Year (fiscal year ending)")
    fig.suptitle("Walk-Forward CV: Actual vs Predicted GDP Growth\n"
                 "(India 1992–2023, time-ordered evaluation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "08_actual_vs_predicted.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ML] Actual vs Predicted chart saved: {path1}")

    # ── Chart 2: Error distribution by growth period ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, res_df) in zip(axes, all_results.items()):
        sns.boxplot(data=res_df, x="Growth_Period", y="AbsError",
                    palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                    order=["High", "Low"], ax=ax, width=0.45)
        sns.stripplot(data=res_df, x="Growth_Period", y="AbsError",
                      palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                      order=["High", "Low"], ax=ax, size=6, alpha=0.7)

        # Mean absolute error per period
        mae_high = res_df[res_df["Growth_Period"] == "High"]["AbsError"].mean()
        mae_low  = res_df[res_df["Growth_Period"] == "Low"]["AbsError"].mean()
        ax.set_title(f"{name}\nMAE High={mae_high:.2f}pp  |  MAE Low={mae_low:.2f}pp",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Growth Period")
        ax.set_ylabel("|Prediction Error| (% points)")

    fig.suptitle("Prediction Error by Growth Period\n"
                 "(Is the model more accurate in high-growth years?)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "09_error_by_period.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Error-by-period chart saved: {path2}")

    # ── Chart 3: Random Forest feature importances ─────────────────────────────
    # Retrain on FULL dataset just for importance inspection
    rf_final = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    X_all = df_feat[FEATURES].values
    y_all = df_feat[TARGET].values
    scaler = StandardScaler()
    rf_final.fit(scaler.fit_transform(X_all), y_all)

    importance_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": rf_final.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color="#FF7043", alpha=0.85, edgecolor="white")
    ax.set_title("Random Forest: Feature Importances\n"
                 "(Which inputs matter most for predicting GDP growth?)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    path3 = os.path.join(FIG_DIR, "10_rf_feature_importance.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ML] Feature importance chart saved: {path3}")

    # ── Metrics comparison table ───────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = os.path.join(RES_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"[ML] Metrics table saved: {metrics_path}")
    print("\n── Model Comparison ─────────────────────────────────────────────")
    print(metrics_df.to_string())

    return {
        "results":  all_results,
        "metrics":  all_metrics,
        "features": FEATURES,
        "rf_importances": importance_df,
        "figure_paths": [path1, path2, path3],
    }


if __name__ == "__main__":
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("dl", os.path.join(os.path.dirname(__file__), "01_data_loader.py"))
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    df = _m.load_data()
    run_ml_models(df)
