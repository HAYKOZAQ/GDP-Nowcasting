# Armenia GDP Nowcasting Codebase

This repository contains the full code pipeline for Armenia GDP nowcasting, backtesting, feature construction, alternative-data integration, and short-horizon GDP forecasting through 2026-Q4.

## Scope

- stage-based GDP nowcasting: `Early`, `Mid`, `Late`
- backtesting across multiple model families
- alternative-data blocks including market variables, Google-based signals, and curated fintech-admin proxies
- forward GDP forecasting for 2026-Q2 to 2026-Q4
- exported results for the thesis and dashboard layers

## Main Structure

- `src/` core data, feature, modeling, and acquisition code
- `data/` processed local datasets used by the pipeline
- `results/` generated backtests, figures, and forecast outputs
- `tests/` validation tests for the framework
- `main.py` end-to-end pipeline entry point

## Environment

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Run the main pipeline:

```powershell
python main.py
```

Run the pipeline including forward forecasts:

```powershell
python main.py --future-forecast
```

## Key Outputs

- `results/backtests/backtest_summary.csv`
- `results/backtests/backtest_predictions.csv`
- `results/backtests/google_trends_ablation_summary.csv`
- `results/backtests/google_trends_ablation_dm.csv`
- `results/forecasts/future_gdp_forecast.csv`
- `results/figures/`

## Current Forward Forecast

- `2026-Q2`: `105.10`
- `2026-Q3`: `104.25`
- `2026-Q4`: `104.10`

Selected forward model: `Ridge`

## Notes

- This codebase contains both the modeling pipeline and the Streamlit dashboard.
- Run `streamlit run dashboard.py` from the root directory to launch the visualization layer.
- Thesis tables and figures are generated from the exported outputs in `results/`.
