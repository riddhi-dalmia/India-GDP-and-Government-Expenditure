"""
app.py — Streamlit Dashboard
Run with: streamlit run app.py
"""

import sys, os, importlib.util

def _load(filename, alias):
    path = os.path.join(os.path.dirname(__file__), "src", filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

loader  = _load("01_data_loader.py",           "data_loader_mod")
val_mod = _load("02_data_validation.py",       "validation_mod")
reg_mod = _load("04_regression.py",            "regression_mod")
per_mod = _load("05_growth_period_analysis.py","period_mod")
ml_mod  = _load("06_ml_models.py",             "ml_mod")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import COLOR_HIGH, COLOR_LOW, COLOR_ALL

st.set_page_config(
    page_title="India GDP & Govt Spending",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; color: #333; }
    h3 { font-size: 1.1rem !important; color: #444; }
    .stMetric label { font-size: 0.8rem; color: #666; }
    div[data-testid="stSidebarNav"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return loader.load_data()

@st.cache_data
def get_validation(_df):
    return val_mod.validate(_df)

@st.cache_data
def get_regression(_df):
    return reg_mod.run_regression(_df)

@st.cache_data
def get_period_analysis(_df):
    return per_mod.compare_periods(_df)

@st.cache_data
def get_ml(_df):
    return ml_mod.run_ml_models(_df)

df = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### India GDP Study")
    st.markdown("---")
    page = st.radio("", [
        "Overview",
        "Data & Validation",
        "Exploratory Charts",
        "Regression",
        "High vs Low Growth",
        "ML Prediction",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Data: RBI & MoSPI (1992–2023)")


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("Government Spending and GDP Growth in India")
    st.markdown("*An empirical study using RBI and MoSPI data, 1992–2023*")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### What this project is about")
        st.markdown("""
        This study looks at whether the **relationship between government spending
        and India's GDP growth** changes depending on how well the economy is doing.

        The core question is: does an increase in government expenditure lead to
        stronger growth when the economy is already slowing down, compared to when
        it is growing fast?

        This idea comes from **Keynesian economics** — the argument that government
        spending fills a gap when private investment is weak. But whether that
        actually holds in India's data is what we test here. The hypothesis is not
        assumed to be true — the data decides.
        """)

        st.markdown("#### How growth periods are defined")
        median_g = df["GDP_Growth"].median()
        st.markdown(f"""
        We split all years into two groups using the **median GDP growth rate
        ({median_g:.1f}%)** as the dividing line:

        - **High-growth years** — GDP growth at or above {median_g:.1f}%
        - **Low-growth years** — GDP growth below {median_g:.1f}%

        Using the median means both groups have roughly the same number of years,
        which makes comparisons fair.
        """)

    with col2:
        st.markdown("#### At a glance")
        m1, m2 = st.columns(2)
        m1.metric("Years of data", f"{len(df)}")
        m2.metric("Period covered", "1992 – 2023")
        m3, m4 = st.columns(2)
        m3.metric("High-growth years", f"{(df['Growth_Period']=='High').sum()}")
        m4.metric("Low-growth years",  f"{(df['Growth_Period']=='Low').sum()}")

        st.markdown("#### Data sources")
        st.markdown("""
        | Variable | Source |
        |----------|--------|
        | GDP growth (% YoY) | MoSPI National Accounts |
        | Govt expenditure (% GDP) | RBI Handbook of Statistics |
        | Fiscal deficit (% GDP) | RBI Handbook of Statistics |
        | Revenue & capital exp | RBI Handbook of Statistics |

        World Bank GDP figures were checked against MoSPI for 2004 onwards.
        Small differences exist due to rounding. MoSPI is the primary source.
        """)

    st.markdown("---")
    st.markdown("#### Full dataset")

    disp = df.copy()
    disp["Period"] = disp["Growth_Period"].map({"High": "High", "Low": "Low"})
    disp = disp.drop(columns=["Growth_Period"])
    st.dataframe(
        disp.style.format({
            "GDP_Growth":      "{:.1f}%",
            "Govt_Exp_GDP":    "{:.1f}%",
            "Fiscal_Deficit":  "{:.1f}%",
            "Revenue_Exp_GDP": "{:.1f}%",
            "Capital_Exp_GDP": "{:.1f}%",
        }).apply(
            lambda row: ["background-color: #e8f4fd" if row["Period"] == "High"
                         else "background-color: #fdecea"] * len(row),
            axis=1
        ),
        use_container_width=True,
        height=400,
    )
    st.caption("Blue rows = high-growth years. Red rows = low-growth years.")


# =============================================================================
# PAGE: DATA & VALIDATION
# =============================================================================
elif page == "Data & Validation":
    st.title("Data & Validation")
    st.markdown("Before analysing anything, we check the data is clean and makes sense.")

    with st.spinner("Running checks..."):
        val = get_validation(df)

    st.markdown("#### Quality checks")
    for p in val["passed"]:
        st.success(p)
    if val["warnings"]:
        for w in val["warnings"]:
            st.warning(w)

    st.markdown("---")
    st.markdown("#### Summary statistics")
    st.dataframe(val["summary_stats"].style.format("{:.2f}"), use_container_width=True)
    st.caption("""
    count = number of observations · mean = average · std = standard deviation ·
    min/max = range · 50% = median
    """)

    st.markdown("---")
    st.markdown("#### Correlation between variables")
    st.markdown("""
    The heatmap below shows Pearson correlation coefficients.
    A value near −1 means a strong negative relationship; near +1 means a strong
    positive relationship; near 0 means little linear relationship.

    Note that a negative correlation between government spending and GDP growth
    does not mean spending causes low growth. Governments typically *increase*
    spending when growth is already weak — the causality likely runs the other way.
    """)

    corr_labels = {
        "GDP_Growth":      "GDP Growth",
        "Govt_Exp_GDP":    "Govt Exp / GDP",
        "Fiscal_Deficit":  "Fiscal Deficit / GDP",
        "Revenue_Exp_GDP": "Revenue Exp / GDP",
        "Capital_Exp_GDP": "Capital Exp / GDP",
    }
    corr = df[list(corr_labels.keys())].corr().rename(index=corr_labels, columns=corr_labels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlations — all years (1992–2023)", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig); plt.close()


# =============================================================================
# PAGE: EXPLORATORY CHARTS
# =============================================================================
elif page == "Exploratory Charts":
    st.title("Exploratory Charts")

    tab1, tab2, tab3 = st.tabs(["GDP Growth", "Government Spending", "Spending vs Growth"])

    with tab1:
        st.markdown("#### India's real GDP growth rate, 1992–2023")
        st.markdown("Red dots mark the years classified as low-growth (below the median).")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(df["Year"], df["GDP_Growth"], color=COLOR_ALL,
                linewidth=2, marker="o", markersize=4, label="GDP growth (%)")
        low_df = df[df["Growth_Period"] == "Low"]
        ax.scatter(low_df["Year"], low_df["GDP_Growth"],
                   color=COLOR_LOW, zorder=5, s=60, label="Low-growth year")
        median_g = df["GDP_Growth"].median()
        ax.axhline(median_g, color="gray", linestyle="--", linewidth=1,
                   label=f"Median ({median_g:.1f}%)")

        for yr, lbl in {1998:"Asian crisis", 2003:"Dotcom slowdown",
                        2009:"GFC spillover", 2021:"COVID-19"}.items():
            row = df[df["Year"] == yr]
            if not row.empty:
                ax.annotate(lbl, xy=(yr, row["GDP_Growth"].values[0]),
                            xytext=(yr + 0.5, row["GDP_Growth"].values[0] - 1.8),
                            fontsize=7.5, color="#777",
                            arrowprops=dict(arrowstyle="-", color="#ccc"))

        ax.set_xlabel("Fiscal year ending")
        ax.set_ylabel("GDP growth (%)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Source: MoSPI National Accounts Statistics")

    with tab2:
        st.markdown("#### Central government expenditure as % of GDP")
        st.markdown("Stacked area shows the split between revenue and capital spending.")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(df["Year"], 0, df["Revenue_Exp_GDP"],
                        alpha=0.55, color="#1976D2", label="Revenue expenditure")
        ax.fill_between(df["Year"], df["Revenue_Exp_GDP"],
                        df["Revenue_Exp_GDP"] + df["Capital_Exp_GDP"],
                        alpha=0.65, color="#FF9800", label="Capital expenditure")
        ax.plot(df["Year"], df["Govt_Exp_GDP"], color="#333",
                linewidth=1.5, linestyle="--", label="Total expenditure")
        ax.set_xlabel("Fiscal year ending")
        ax.set_ylabel("% of GDP")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Source: RBI Handbook of Statistics on Indian Economy")

    with tab3:
        st.markdown("#### Government spending vs GDP growth, split by period")
        st.markdown("""
        Each dot is one year. The dashed lines are simple linear trend lines fitted
        separately for high-growth (blue) and low-growth (red) years.

        If the slopes are noticeably different, it suggests the relationship between
        spending and growth changes depending on economic conditions.
        """)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"High": COLOR_HIGH, "Low": COLOR_LOW}
        for period, grp in df.groupby("Growth_Period"):
            ax.scatter(grp["Govt_Exp_GDP"], grp["GDP_Growth"],
                       color=colors[period], label=f"{period}-growth",
                       s=65, alpha=0.85, edgecolors="white", linewidth=0.4)
            m, b = np.polyfit(grp["Govt_Exp_GDP"], grp["GDP_Growth"], 1)
            xs = pd.Series(sorted(grp["Govt_Exp_GDP"]))
            ax.plot(xs, m*xs+b, color=colors[period], linewidth=1.8,
                    linestyle="--", label=f"{period} trend (slope = {m:.2f})")

        ax.set_xlabel("Govt expenditure (% of GDP)")
        ax.set_ylabel("GDP growth (%)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()


# =============================================================================
# PAGE: REGRESSION
# =============================================================================
elif page == "Regression":
    st.title("Regression Analysis")
    st.markdown("""
    We fit a simple OLS regression: **GDP growth = β₀ + β₁ × Govt expenditure**

    β₁ is the slope — how many percentage points of GDP growth are associated with
    a 1 percentage-point rise in government expenditure as a share of GDP.
    We fit this separately for all years, high-growth years, and low-growth years.
    """)

    with st.spinner("Running regressions..."):
        reg = get_regression(df)

    # Three metric cards
    c1, c2, c3 = st.columns(3)
    for col_ui, (key, label) in zip([c1, c2, c3],
        [("All","All years"), ("High","High-growth"), ("Low","Low-growth")]):
        r = reg[key]
        sig = "p < 0.05 — significant" if r["pval"] < 0.05 else "p ≥ 0.05 — not significant"
        col_ui.metric(label, f"β₁ = {r['coef']:+.3f}")
        col_ui.caption(f"R² = {r['r2']:.3f} · {sig}")

    st.markdown("---")

    # Coefficient plot
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["All years", "High-growth", "Low-growth"]
    coefs  = [reg[k]["coef"] for k in ["All","High","Low"]]
    pvals  = [reg[k]["pval"] for k in ["All","High","Low"]]
    ci_lo  = [reg[k]["ci"][0] for k in ["All","High","Low"]]
    ci_hi  = [reg[k]["ci"][1] for k in ["All","High","Low"]]
    colors = [COLOR_ALL, COLOR_HIGH, COLOR_LOW]

    x = np.arange(len(labels))
    bars = ax.bar(x, coefs, color=colors, alpha=0.7, width=0.45)
    ax.errorbar(x, coefs,
                yerr=[[c-lo for c,lo in zip(coefs,ci_lo)],
                      [hi-c  for c,hi in zip(coefs,ci_hi)]],
                fmt="none", color="#333", capsize=5, linewidth=1.2)
    ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("β₁  (with 95% confidence interval)")
    ax.set_title("OLS slope by period", fontsize=11)

    for i, (bar, p) in enumerate(zip(bars, pvals)):
        star = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.1 else "n.s."))
        ypos = max(bar.get_height(), 0) + 0.02
        ax.text(bar.get_x()+bar.get_width()/2, ypos, star,
                ha="center", va="bottom", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.caption("* p<0.1  · ** p<0.05  · *** p<0.01  · n.s. = not significant")

    st.markdown("---")
    st.markdown("""
    #### A note on interpretation

    A negative β₁ does not mean government spending *causes* slower growth. The
    more likely explanation is reverse causality: when growth slows, the government
    responds by increasing spending. This is standard counter-cyclical fiscal policy.
    To separate cause and effect, you would need a more advanced design (instrumental
    variables, or a natural experiment). This is an important limitation to flag.
    """)

    with st.expander("Full OLS output tables"):
        for key, label in [("All","All years"),("High","High-growth"),("Low","Low-growth")]:
            st.markdown(f"**{label}**")
            st.text(reg[key]["model"].summary().as_text())


# =============================================================================
# PAGE: HIGH vs LOW GROWTH
# =============================================================================
elif page == "High vs Low Growth":
    st.title("High-growth vs Low-growth Periods")
    st.markdown("""
    We use a **Welch's t-test** to check whether the average value of each variable
    is meaningfully different between high-growth and low-growth years.
    Welch's version is used because it does not assume the two groups have equal variance.
    """)

    with st.spinner("Running comparison..."):
        per = get_period_analysis(df)

    ttest = per["ttest"]

    # Clean results table
    rows = []
    for col, res in ttest.items():
        rows.append({
            "Variable":        col.replace("_", " "),
            "High-growth avg": f"{res['mean_high']:.2f}%",
            "Low-growth avg":  f"{res['mean_low']:.2f}%",
            "Difference":      f"{res['diff']:+.2f}%",
            "p-value":         f"{res['p_val']:.4f}",
            "Result":          "Significant" if res["significant"] else "Not significant",
        })
    results_df = pd.DataFrame(rows).set_index("Variable")
    st.dataframe(
        results_df.style.apply(
            lambda col: ["color: green; font-weight: bold"
                         if v == "Significant" else "color: #888"
                         for v in col]
            if col.name == "Result" else [""] * len(col),
            axis=0
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Distribution comparison")

    cols5 = ["GDP_Growth","Govt_Exp_GDP","Fiscal_Deficit","Revenue_Exp_GDP","Capital_Exp_GDP"]
    nice  = ["GDP Growth (%)","Govt Exp (% GDP)","Fiscal Deficit (% GDP)",
             "Revenue Exp (% GDP)","Capital Exp (% GDP)"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col, lbl in zip(axes, cols5, nice):
        sns.boxplot(data=df, x="Growth_Period", y=col,
                    palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                    order=["High","Low"], ax=ax, width=0.45,
                    linewidth=1, flierprops=dict(marker="o", markersize=4))
        sns.stripplot(data=df, x="Growth_Period", y=col,
                      palette={"High": COLOR_HIGH, "Low": COLOR_LOW},
                      order=["High","Low"], ax=ax, size=4, alpha=0.55, jitter=True)
        p = ttest[col]["p_val"]
        star = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        ax.set_title(f"{lbl}\np = {p:.3f}  {star}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Distribution of variables: high-growth vs low-growth years",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("""
    #### What does this tell us about the hypothesis?

    The t-test here checks whether spending *levels* differ between the two groups.
    If they do, it is consistent with counter-cyclical fiscal policy — the government
    spends more when growth is low.

    To properly evaluate the hypothesis, we also need to look at whether the
    *slope* of the relationship (the β₁ from the regression page) differs between
    the two periods. Both pieces of evidence together give a fuller picture.
    """)


# =============================================================================
# PAGE: ML PREDICTION
# =============================================================================
elif page == "ML Prediction":
    st.title("ML Prediction")
    st.markdown("""
    We train two models to predict India's GDP growth rate from government spending
    indicators. The goal is to see whether spending data has predictive value.

    **Walk-forward cross-validation** is used throughout — when predicting year T,
    the model is only trained on years before T. This avoids letting the model
    see future data, which would give artificially good results.

    Input features: govt expenditure, fiscal deficit, revenue and capital expenditure
    (all as % of GDP), plus last year's GDP growth as a momentum variable.
    """)

    with st.spinner("Training models..."):
        ml = get_ml(df)

    # Metrics
    st.markdown("#### Model performance")
    mrows = []
    for name, m in ml["metrics"].items():
        mrows.append({
            "Model": name,
            "MAE (pp)":  m["MAE"],
            "RMSE (pp)": m["RMSE"],
            "R²":        m["R2"],
            "n":         m["n"],
        })
    mdf = pd.DataFrame(mrows).set_index("Model")
    st.dataframe(
        mdf.style
           .highlight_min(subset=["MAE (pp)","RMSE (pp)"], color="#d4edda")
           .highlight_max(subset=["R²"], color="#d4edda")
           .format({"MAE (pp)": "{:.3f}", "RMSE (pp)": "{:.3f}", "R²": "{:.3f}"}),
        use_container_width=True,
    )
    st.caption("MAE and RMSE are in percentage points of GDP growth. Lower is better. R² closer to 1 is better.")

    st.markdown("---")
    st.markdown("#### Actual vs predicted GDP growth")

    colors_m = {"Linear Regression": "#7B1FA2", "Random Forest": "#E65100"}
    for name, res_df in ml["results"].items():
        m = ml["metrics"][name]
        st.markdown(f"**{name}** — MAE = {m['MAE']:.2f} pp · R² = {m['R2']:.3f}")
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(res_df["Year"], res_df["Actual"],
                color=COLOR_ALL, linewidth=2, marker="o", markersize=4,
                label="Actual")
        ax.plot(res_df["Year"], res_df["Predicted"],
                color=colors_m[name], linewidth=1.8, linestyle="--",
                marker="s", markersize=4, label="Predicted")
        ax.axhline(0, color="#ccc", linewidth=0.6)
        ax.set_ylabel("GDP growth (%)")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Random Forest — feature importances")
    st.markdown("Which inputs does the model rely on most?")

    imp = ml["rf_importances"].sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(imp["Feature"], imp["Importance"],
            color="#FF7043", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Importance score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    If last year's GDP growth (`GDP_Growth_Lag1`) scores highest, it means
    economic momentum is more predictive than spending levels alone. That is a
    meaningful finding in itself.
    """)

    with st.expander("Detailed predictions table"):
        for name, res_df in ml["results"].items():
            st.markdown(f"**{name}**")
            st.dataframe(res_df.style.format({
                "Actual":    "{:.2f}%",
                "Predicted": "{:.2f}%",
                "Error":     "{:+.2f}%",
                "AbsError":  "{:.2f}%",
            }), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: RBI Handbook of Statistics on Indian Economy · MoSPI National Accounts Statistics · 1992–2023")
