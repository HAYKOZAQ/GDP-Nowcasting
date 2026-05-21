# Armenia Real GDP Nowcasting & Short-Horizon Forecasting Framework

This repository hosts a production-grade, modular python framework for real-time tracking, nowcasting, and recursive forecasting of Armenia's Real GDP growth. The project integrates traditional macroeconomic statistical indicators with alternative daily/monthly datasets—such as Google Search Trends, international commodity indices, exchange rates, remittance inflows, and high-frequency fintech-administrative data—processed within a stage-aware, mixed-frequency machine learning pipeline.

---

## 🎯 Project Motivation & Objectives

National accounts data (GDP) are published with a substantial delay—typically 60 to 90 days after the end of the reference quarter. In small, highly open economies exposed to structural transformations and external geopolitical/remittance shocks, this reporting lag creates an information vacuum for policymakers (e.g., Central Bank of Armenia, Ministry of Finance) and private enterprises.

To resolve this, this project builds a **stage-aware, mixed-frequency nowcasting and forecasting pipeline** that:
1. **Handles Mixed-Frequency Data**: Integrates daily, monthly, and quarterly series into a unified modeling framework.
2. **Mitigates the "Ragged-Edge" Problem**: Models real-time data releases where indicators are updated asynchronously with varying lags (e.g., immediate daily prices vs. monthly indicators delayed by 20–30 days).
3. **Compares Structural Econometrics with Machine Learning**: Evaluates state-space Dynamic Factor Models (DFM) against modern regularized regressions, tree-based ensembles (Random Forest, LightGBM), and stacking meta-regressors.
4. **Validates Alternative Data Value**: Quantifies the marginal predictive contribution of search engine query indices (Google Trends) and real-time transaction data.

---

## 🗂️ Core Methodologies & Mathematical Formulations

### 1. Mixed-Frequency State-Space Dynamic Factor Model (DFM)
The structural econometric benchmark is a DFM estimated via the Kalman Filter/Smoother and the Expectation-Maximization (EM) algorithm. 

#### System Formulation
Let $\mathbf{Y}_t$ be an $N \times 1$ vector of observed monthly indicators (standardized to have mean zero and unit variance). We assume $\mathbf{Y}_t$ is driven by a small number of latent common factors $\mathbf{f}_t$ (here, $r=1$) and an $N \times 1$ vector of idiosyncratic noise components $\mathbf{e}_t$:
$$\mathbf{Y}_t = \mathbf{\Lambda} \mathbf{f}_t + \mathbf{e}_t, \quad \mathbf{e}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$$
where $\mathbf{\Lambda}$ is the $N \times r$ factor loading matrix, and the idiosyncratic covariance matrix $\mathbf{R}$ is diagonal: $\mathbf{R} = \text{diag}(\sigma_1^2, \dots, \sigma_N^2)$.

The latent factor is modeled as a Vector Autoregressive process of order $p$:
$$\mathbf{f}_t = \mathbf{A}_1 \mathbf{f}_{t-1} + \dots + \mathbf{A}_p \mathbf{f}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$

#### Mariano-Murasawa Quarterly Mapping
To incorporate quarterly GDP YoY index ($Y_{Q, t}$) observed only on the 3rd month of each quarter ($t = 3, 6, 9, \dots$), we use the Mariano-Murasawa linear approximation mapping the monthly latent factor to quarterly variables:
$$Y_{Q, t} = \frac{1}{3} Y_{M, t} + \frac{2}{3} Y_{M, t-1} + Y_{M, t-2} + \frac{2}{3} Y_{M, t-3} + \frac{1}{3} Y_{M, t-4}$$
Assuming the monthly unobserved GDP counterpart $Y_{M, t}$ loaded on the factor $\mathbf{f}_t$:
$$Y_{M, t} = \lambda_Q \mathbf{f}_t + e_{Q, t}$$
This creates a state-space representation where the state vector $\mathbf{S}_t$ contains current and lagged monthly factors:
$$\mathbf{S}_t = \begin{bmatrix} \mathbf{f}_t^T & \mathbf{f}_{t-1}^T & \mathbf{f}_{t-2}^T & \mathbf{f}_{t-3}^T & \mathbf{f}_{t-4}^T \end{bmatrix}^T$$
Measurement equations are adjusted to place zero weight on missing values (ragged-edge) or non-quarterly months, and the Kalman Filter recursively updates estimates of $\mathbf{S}_t$:
$$\mathbf{S}_{t|t} = \mathbf{S}_{t|t-1} + \mathbf{K}_t (\mathbf{Y}_t - \mathbf{H}_t \mathbf{S}_{t|t-1})$$
where $\mathbf{K}_t$ is the Kalman gain matrix, and $\mathbf{H}_t$ is the time-varying measurement selection matrix.

---

### 2. Machine Learning & Ensemble Stacking Nowcasting
For non-linear and high-dimensional prediction, we train a regularized stacking model.

#### Base Estimators:
- **Ridge Regression**: Employs $L_2$ regularization to prevent overfitting on collinear indicators:
  $$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
- **ElasticNet Regression**: Blends $L_1$ and $L_2$ penalties:
  $$\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$$
- **Random Forest**: Builds decorrelated decision trees, splitting on bootstrap samples to reduce variance.
- **LightGBM**: Gradient boosting framework using leaf-wise tree growth, optimized for fast training on high-dimensional tables.

#### Stacking Meta-Regression:
The base model predictions are compiled into a meta-feature matrix. A meta-estimator (Ridge Regressor with cross-validation) is trained on out-of-fold predictions to determine the optimal weights:
$$\hat{y}_{\text{stacked}} = \beta_0 + \sum_{m \in \mathcal{M}} \beta_m \hat{y}_{m}$$
This enables the framework to dynamically shift weight from purely structural predictions (like the DFM) to data-driven non-linear indicators as the information set expands.

---

### 3. Stage-Based Information Flow
To replicate real-time forecasting, the backtest evaluates three distinct intra-quarterly information stages:
1. **Early (Month 1)**: Available data includes daily commodities, financial asset prices, Google Trends, and previous quarter lags. Official real-sector data for the current quarter is completely absent.
2. **Mid (Month 2)**: Staggered release of Month 1 economic indicators (EAI, CPI, trade).
3. **Late (Month 3)**: Nearly complete monthly datasets for Months 1 and 2, with immediate alternative indicators extending into Month 3.

---

## 📈 Performance Summary & Empirical Results

The models are tested using a rolling walk-forward backtest (minimum training window of 40 quarters). 

### 1. Model Accuracy Comparison (Mean Absolute Percentage Error - MAPE)
Historically, the ensemble StackingNowcast model outperforms single structural benchmarks, showing substantial accuracy gains as the information set expands:

| Model Family | Model Name | Early Stage MAPE | Mid Stage MAPE | Late Stage MAPE | Average MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ensemble** | `StackingNowcast` | **2.382%** | **2.369%** | **1.875%** | **2.209%** |
| **Machine Learning** | `Ridge` | 2.502% | 2.488% | 1.942% | 2.311% |
| **Machine Learning** | `ElasticNet` | 2.554% | 2.520% | 1.996% | 2.357% |
| **Structural/DFM** | `DFM` | 2.923% | 2.802% | 2.766% | 2.830% |

- **Late-Stage Dominance**: At the `Late` stage (Month 3), the prediction error falls below **1.9%**, demonstrating that integrating high-frequency variables provides a highly accurate preview of GDP.
- **Ensemble Advantage**: Stacking base models yields a **22% reduction in error** over the standard structural Dynamic Factor Model.
- **Crisis Robustness**: Under extreme macroeconomic shocks (e.g., the 2020-Q2 COVID-19 contraction), the `EarlyShockAdjusted` model successfully captures structural breaks, avoiding the lagged adjustment error typical of standard Kalman-filter formulations.

### 2. Google Trends Marginal Value (Ablation Analysis)
An ablation analysis is executed by comparing a base structural macroeconomic model against a counterpart augmented with Google Search Trends queries.

- **Early Stage Benefit**: Query composites (e.g., search keywords reflecting local tourism interest, immigration, employment, and inflation anxiety) provide key leading signals during Month 1.
- **Statistical Significance**: Diebold-Mariano (DM) tests yield a $p$-value of **~0.40** for the early stage. This suggests that while alternative search data improves point forecasts, it acts as a valuable secondary layer rather than a standalone driver, and its significance is secondary to official fast-indicators once they are published.

---

## 📂 Codebase Architecture & Structure

```
GDP_NOWCASTING_DASHBOARD/
│
├── config.yaml                     # Centralized hyperparameters & DFM orders
├── requirements.txt                # Python environment specifications
├── main.py                         # End-to-end forecasting pipeline entrypoint
├── dashboard.py                    # Clean Streamlit sidebar router
│
├── 📁 src/                         # Core Python modules
│   ├── 📄 nowcast_config.py        # YAML configuration parser
│   ├── 📁 data/                    # Mixed-frequency panel builders
│   ├── 📁 data_acquisition/        # Armstat and Central Bank API scrapers
│   ├── 📁 evaluation/              # Walk-forward rolling backtest engine
│   ├── 📁 forecasting/             # Recursive forward forecast modules
│   ├── 📁 models/                  # DFM & Machine Learning model wrappers
│   ├── 📁 diagnostics/             # Residual tracking and hotspot metrics
│   ├── 📁 reporting/               # Automated thesis markdown report generator
│   └── 📁 visualization/           # Decoupled visualization/plotting library
│
├── 📁 views/                       # Dashboard presentation layer
│   ├── 📄 utils.py                 # Data loaders, translations, and templates
│   └── 📄 nowcast.py               # GDP Nowcast charts and analysis panels
│
├── 📁 data/                        # Cleaned data directories
│   ├── 📁 data_xlsx/               # Dynamic Excel workbooks (e.g. Translated_Cleaned_Nowcasting_Data.xlsx)
│   ├── 📁 processed/               # Intermediate parsed CSV datasets
│   └── 📁 raw/                     # Raw database outputs
│
├── 📁 results/                     # Dynamic modeling outputs (Gitignored)
│   ├── 📁 backtests/               # Summaries, predictions, and DM tests
│   ├── 📁 forecasts/               # Forward-looking quarterly predictions
│   └── 📁 figures/                 # Auto-generated PNG analytical plots
│
├── 📁 figures/                     # Static thesis LaTeX figures
├── 📁 legacy/                      # Redundant/legacy codebase backups
├── 📁 txt/                         # Thesis chapters and TeX compiler script
└── 📁 tests/                       # Unit test suites (all 32 tests passing)
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisite
Ensure you have **Python 3.10 to 3.12** installed on your operating system.

### 2. Install Packages
Clone this repository and run pip installation from the root directory:
```powershell
pip install -r requirements.txt
```

### 3. Pipeline Execution
Run the full backtest modeling pipeline and generate report artifacts:
```powershell
python main.py
```
To calculate forward forecasts through the next three quarters (e.g., 2026-Q2 to 2026-Q4) and output prediction charts:
```powershell
python main.py --future-forecast
```

### 4. Launching the Interactive Dashboard
To launch the Streamlit dashboard visualization layer locally:
```powershell
streamlit run dashboard.py
```
This opens the dark-themed dashboard in your default browser at `http://localhost:8501`, featuring three main sections:
- **Գլխավոր էջ (Home)**: Forward GDP forecast trajectory, uncertainty bands, and metrics.
- **Մոդելների վերլուծություն (Model Analysis)**: Historical backtest errors, ranking heatmaps, and ablation results.
- **Գծապատկերներ (Charts)**: Complete interactive directory containing all 15 walk-forward performance plots.
