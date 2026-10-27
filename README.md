# 🇮🇳 Government Spending and GDP Growth in India

**An empirical academic study using RBI and MoSPI data (1992–2023)**

---

## 🎯 Research Question

> Does the relationship between government expenditure and India's GDP growth differ between high-growth and low-growth economic periods?

## 💡 Hypothesis *(tested objectively — not pre-assumed true)*

Government expenditure has a **stronger positive association** with GDP growth during **low-growth periods**. This is inspired by the Keynesian fiscal multiplier — when private investment is weak, public spending may fill the demand gap more effectively.

---

## 📂 Project Structure

```
XAI/
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   ├── raw/                        # Raw data files (if added later)
│   └── processed/                  # Auto-generated cleaned CSV
├── src/
│   ├── utils.py                    # Shared paths and constants
│   ├── 01_data_loader.py           # Loads RBI/MoSPI data
│   ├── 02_data_validation.py       # Data quality checks
│   ├── 03_eda.py                   # Exploratory charts
│   ├── 04_regression.py            # OLS regression analysis
│   ├── 05_growth_period_analysis.py# t-tests: High vs Low periods
│   └── 06_ml_models.py             # Linear Regression + Random Forest
└── outputs/
    ├── figures/                    # Auto-saved charts (.png)
    └── results/                    # Auto-saved result tables (.csv/.txt)
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

### 3. Or run individual modules

```bash
python src/01_data_loader.py
python src/02_data_validation.py
python src/03_eda.py
python src/04_regression.py
python src/05_growth_period_analysis.py
python src/06_ml_models.py
```

---

## 📊 Data Sources

| Source | Variables | Link |
|--------|-----------|------|
| **RBI Handbook of Statistics on Indian Economy** | Govt Expenditure, Fiscal Deficit (% of GDP) | https://rbi.org.in |
| **MoSPI National Accounts Statistics** | Real GDP Growth (% YoY, base year 2011-12) | https://mospi.gov.in |
| World Bank (cross-check only) | GDP growth 2004–2023 | https://data.worldbank.org |

---

## 📐 Methodology

| Step | Module | Technique |
|------|--------|-----------|
| 1 | `01_data_loader.py` | Data ingestion, median-split classification |
| 2 | `02_data_validation.py` | Missing values, range checks, year continuity |
| 3 | `03_eda.py` | Time-series plots, scatter, correlation heatmap |
| 4 | `04_regression.py` | OLS with statsmodels, β coefficient by period |
| 5 | `05_growth_period_analysis.py` | Welch's t-test, box plots |
| 6 | `06_ml_models.py` | Linear Regression + Random Forest, walk-forward CV |

---

## 🧠 Key Concepts for Your Defence

| Concept | What it means |
|---------|---------------|
| **Median split** | We classify years where GDP growth ≥ median as "High", below as "Low" — ensures balanced groups |
| **OLS β₁ coefficient** | Estimated slope: change in GDP growth per 1pp rise in Govt Exp |
| **p-value** | Probability of observing this slope by chance; p < 0.05 = statistically significant |
| **R-squared** | Fraction of GDP growth variation explained by the model |
| **Walk-forward CV** | Train only on past data to predict each future year — prevents data leakage |
| **Welch's t-test** | Tests if group means differ significantly without assuming equal variances |
| **Counter-cyclical spending** | Governments boost spending in bad years — creates negative correlation between spending and growth |
| **Endogeneity** | When X causes Y and Y causes X simultaneously — a key limitation here |

---

## ⚠️ Limitations

1. **Small sample**: 32 annual observations limit statistical power.
2. **Endogeneity**: Government spending responds to growth, not just the other way.
3. **Omitted variables**: Monetary policy, external trade, global shocks not modelled.
4. **Aggregation**: Uses central government data only; state spending omitted.
5. **Causation vs correlation**: All findings are associative, not causal.
