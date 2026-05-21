import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOWCAST_RESULTS_DIR = os.path.join(BASE_DIR, "results", "backtests")
NOWCAST_FALLBACK_DIR = os.path.join(BASE_DIR, "nowcasting_results")
FORECAST_DIR = os.path.join(BASE_DIR, "results", "forecasts")
FORECAST_FALLBACK_DIR = os.path.join(BASE_DIR, "nowcasting_results")

@st.cache_data
def load_data(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))

@st.cache_data
def load_nowcasting_results():
    candidate_dirs = [NOWCAST_RESULTS_DIR, NOWCAST_FALLBACK_DIR]
    summary_path = None
    predictions_path = None
    for candidate_dir in candidate_dirs:
        candidate_summary = os.path.join(candidate_dir, "backtest_summary.csv")
        candidate_predictions = os.path.join(candidate_dir, "backtest_predictions.csv")
        if os.path.exists(candidate_summary) and os.path.exists(candidate_predictions):
            summary_path = candidate_summary
            predictions_path = candidate_predictions
            break

    if not summary_path or not predictions_path:
        return None, None

    summary = pd.read_csv(summary_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["prediction_date", "train_end"])

    for column in ["mape", "mae", "rmse", "coverage_50", "coverage_90", "avg_width_50", "avg_width_90"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    for column in ["prediction", "actual", "abs_pct_error"]:
        if column in predictions.columns:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    return summary, predictions

@st.cache_data
def load_future_forecast():
    candidate_paths = [
        os.path.join(FORECAST_DIR, "future_gdp_forecast.csv"),
        os.path.join(FORECAST_FALLBACK_DIR, "future_gdp_forecast.csv"),
    ]
    forecast_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if not forecast_path:
        return None

    forecast = pd.read_csv(forecast_path, parse_dates=["forecast_date", "last_observed_quarter"])
    for column in ["forecast", "interval_lo_50", "interval_hi_50", "interval_lo_90", "interval_hi_90"]:
        if column in forecast.columns:
            forecast[column] = pd.to_numeric(forecast[column], errors="coerce")

    return forecast.sort_values("horizon").reset_index(drop=True)

@st.cache_data
def load_google_ablation():
    candidate_dirs = [NOWCAST_RESULTS_DIR, NOWCAST_FALLBACK_DIR]
    summary_path = None
    dm_path = None
    for candidate_dir in candidate_dirs:
        candidate_summary = os.path.join(candidate_dir, "google_trends_ablation_summary.csv")
        candidate_dm = os.path.join(candidate_dir, "google_trends_ablation_dm.csv")
        if os.path.exists(candidate_summary) and os.path.exists(candidate_dm):
            summary_path = candidate_summary
            dm_path = candidate_dm
            break

    if not summary_path or not dm_path:
        return None, None

    ablation_summary = pd.read_csv(summary_path)
    ablation_dm = pd.read_csv(dm_path)
    for frame in [ablation_summary, ablation_dm]:
        for column in frame.columns:
            if column not in {"model", "stage", "loss", "model_a", "model_b", "better_model"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return ablation_summary, ablation_dm

@st.cache_data
def load_recent_actual_quarters():
    candidate_dirs = [NOWCAST_RESULTS_DIR, NOWCAST_FALLBACK_DIR]
    predictions_path = None
    for candidate_dir in candidate_dirs:
        candidate_predictions = os.path.join(candidate_dir, "backtest_predictions.csv")
        if os.path.exists(candidate_predictions):
            predictions_path = candidate_predictions
            break

    if not predictions_path:
        return None

    predictions = pd.read_csv(predictions_path)
    actuals = (
        predictions[["target_quarter", "actual"]]
        .dropna()
        .drop_duplicates()
        .sort_values("target_quarter")
    )
    actuals["actual"] = pd.to_numeric(actuals["actual"], errors="coerce")
    return actuals[actuals["target_quarter"].astype(str).str.startswith(("2025", "2026-Q1"))].reset_index(drop=True)

PERIOD_MAP = {
    1: "Հունվար", 2: "Փետրվար", 3: "Մարտ", 4: "Ապրիլ", 5: "Մայիս", 6: "Հունիս",
    7: "Հուլիս", 8: "Օգոստոս", 9: "Սեպտեմբեր", 10: "Հոկտեմբեր", 11: "Նոյեմբեր", 12: "Դեկտեմբեր",
    "I": "Հունվար", "II": "Փետրվար", "III": "Մարտ", "IV": "Ապրիլ", "V": "Մայիս", "VI": "Հունիս",
    "VII": "Հուլիս", "VIII": "Օգոստոս", "IX": "Սեպտեմբեր", "X": "Հոկտեմբեր", "XI": "Նոյեմբեր", "XII": "Դեկտեմբեր"
}

def translate_p(p):
    if isinstance(p, str):
        if "-" in p:
            parts = p.split("-")
            if parts[1] in ["I", "II", "III", "IV"]:
                return f"{parts[0]}-{parts[1]} եռ."
        return PERIOD_MAP.get(p, p)
    return PERIOD_MAP.get(p, p)

def S(fig, h=500):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,34,0.95)",
        height=h, margin=dict(l=40, r=40, t=60, b=60), font=dict(family="Noto Sans Armenian", size=13),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#30363d", gridwidth=0.5, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#30363d", gridwidth=0.5, zeroline=False)
    return fig
