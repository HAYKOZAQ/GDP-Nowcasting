import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Õ€Ô±Õ…Ô±ÕÕÔ±Õ† 2025: ÕÕˆÕ‘Ô»Ô±Ô¼-ÕÕ†ÕÔµÕÔ±Ô¿Ô±Õ†", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOWCAST_RESULTS_DIR = os.path.join(BASE_DIR, "nowcasting_results")
NOWCAST_FALLBACK_DIR = os.path.join(os.path.dirname(BASE_DIR), "nowcasting-CODE", "results", "backtests")
FORECAST_DIR = os.path.join(BASE_DIR, "nowcasting_results")
FORECAST_FALLBACK_DIR = os.path.join(os.path.dirname(BASE_DIR), "nowcasting-CODE", "results", "forecasts")

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
    1: "Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€", 2: "Õ“Õ¥Õ¿Ö€Õ¾Õ¡Ö€", 3: "Õ„Õ¡Ö€Õ¿", 4: "Ô±ÕºÖ€Õ«Õ¬", 5: "Õ„Õ¡ÕµÕ«Õ½", 6: "Õ€Õ¸Ö‚Õ¶Õ«Õ½",
    7: "Õ€Õ¸Ö‚Õ¬Õ«Õ½", 8: "Õ•Õ£Õ¸Õ½Õ¿Õ¸Õ½", 9: "ÕÕ¥ÕºÕ¿Õ¥Õ´Õ¢Õ¥Ö€", 10: "Õ€Õ¸Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€", 11: "Õ†Õ¸ÕµÕ¥Õ´Õ¢Õ¥Ö€", 12: "Ô´Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€",
    "I": "Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€", "II": "Õ“Õ¥Õ¿Ö€Õ¾Õ¡Ö€", "III": "Õ„Õ¡Ö€Õ¿", "IV": "Ô±ÕºÖ€Õ«Õ¬", "V": "Õ„Õ¡ÕµÕ«Õ½", "VI": "Õ€Õ¸Ö‚Õ¶Õ«Õ½",
    "VII": "Õ€Õ¸Ö‚Õ¬Õ«Õ½", "VIII": "Õ•Õ£Õ¸Õ½Õ¿Õ¸Õ½", "IX": "ÕÕ¥ÕºÕ¿Õ¥Õ´Õ¢Õ¥Ö€", "X": "Õ€Õ¸Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€", "XI": "Õ†Õ¸ÕµÕ¥Õ´Õ¢Õ¥Ö€", "XII": "Ô´Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€"
}

def translate_p(p):
    if isinstance(p, str):
        if "-" in p:
            parts = p.split("-")
            if parts[1] in ["I", "II", "III", "IV"]:
                return f"{parts[0]}-{parts[1]} Õ¥Õ¼."
        return PERIOD_MAP.get(p, p)
    return PERIOD_MAP.get(p, p)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Armenian:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans Armenian', sans-serif; }
.main { background: #0d1117; }
h1, h2, h3 { color: #58a6ff !important; font-weight: 700; }
.stMarkdown p, .stMarkdown li { color: #c9d1d9; font-size: 1.1rem; line-height: 1.6; }
.stMetric [data-testid="stMetricValue"] { font-size: 2.25rem; line-height: 1.1; }
.stMetric [data-testid="stMetricLabel"] p { font-size: 1rem; }
.sidebar .sidebar-content { background-image: linear-gradient(#161b22, #0d1117); }
</style>""", unsafe_allow_html=True)

def S(fig, h=500):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,34,0.95)",
        height=h, margin=dict(l=40, r=40, t=60, b=60), font=dict(family="Noto Sans Armenian", size=13),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#30363d", gridwidth=0.5, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#30363d", gridwidth=0.5, zeroline=False)
    return fig

page = st.sidebar.radio("Ô¸Õ¶Õ¿Ö€Õ¥Ö„ Õ¢Õ¡ÕªÕ«Õ¶Õ¨", [
    "Õ€Õ†Ô± nowcasting",
    "Õ„Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ¥Ö€Õ« Õ·Õ¡Ö€ÕªÕ¨Õ¶Õ©Õ¡ÖÕ¨", "Õ€Õ€ Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶", "Ô´Ö€Õ¡Õ´Õ¡Õ¯Õ¡Õ¶ ÖƒÕ¸Õ­Õ¡Õ¶ÖÕ¸Ö‚Õ´Õ¶Õ¥Ö€Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶", "ÕÕ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¯Õ¿Õ«Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶",
    "Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô±Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ·Õ¸Ö‚Õ¯Õ¡Õ¶ Ö‡ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¡Õ¯Õ¡Õ¶ Õ©Õ¸Ö‚ÕµÕ¬Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ ÔµÖ€Ö‡Õ¡Õ¶Õ¸Ö‚Õ´",
    "Ô¶Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô³Õ¸Ö€Õ®Õ¡Õ¦Ö€Õ¯Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¼Õ¥Õ½Õ¸Ö‚Ö€Õ½Õ¶Õ¥Ö€", "Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ± Ö‡ Õ¾Õ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€",
    "Ô±Ö€Õ¿Õ¡Ö„Õ«Õ¶ Õ¡Õ¼Ö‡Õ¿Ö€Õ¡Õ·Ö€Õ»Õ¡Õ¶Õ¡Õ¼Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô´Ö€Õ¡Õ´Õ¡Õ¾Õ¡Ö€Õ¯Õ¡ÕµÕ«Õ¶ Õ¯Õ¡ÕµÕ¸Ö‚Õ¶Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Ô³Õ¶Õ¡Õ³", "Ô·Õ¶Õ¥Ö€Õ£Õ¥Õ¿Õ«Õ¯Õ¡ Ö‡ Õ„Õ¡Õ¯Ö€Õ¸-Õ¡Õ¼Õ¡Õ»Õ¡Õ¶ÖÕ«Õ¯ ÖÕ¸Ö‚ÖÕ«Õ¹", "Õ€Õ¡Ö€Õ¯Õ¡Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ¿Õ¡ÕµÕ«Õ¶ ÖÕ¸Ö‚ÖÕ¡Õ¶Õ«Õ·Õ¶Õ¥Ö€", "Ô²Õ¡Õ¶Õ¯Õ¡ÕµÕ«Õ¶ Õ°Õ¡Õ´Õ¡Õ¯Õ¡Ö€Õ£ Ö‡ ÕŽÕ¡Ö€Õ¯Õ¡Õ¾Õ¸Ö€Õ¸Ö‚Õ´",
    "Õ„Õ¡Ö€Õ¦Õ¡ÕµÕ«Õ¶ Õ¿Õ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ ÕºÕ¡Õ¿Õ¯Õ¥Ö€", "ÕÕ Ö‡ Ô²Õ¡Ö€Õ±Ö€ Õ¿Õ¥Õ­Õ¶Õ¸Õ¬Õ¸Õ£Õ«Õ¡Õ¶Õ¥Ö€", "ÔºÕ¸Õ²Õ¸Õ¾Ö€Õ¤Õ¡Õ£Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡"
])

if page == "Õ€Õ†Ô± nowcasting":
    st.title("Õ€Õ¡ÕµÕ¡Õ½Õ¿Õ¡Õ¶Õ« Õ€Õ¡Õ¶Ö€Õ¡ÕºÕ¥Õ¿Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ€Õ†Ô±-Õ« Nowcasting")
    nowcast_section = st.radio(
        "Ô¸Õ¶Õ¿Ö€Õ¥Ö„ nowcasting Õ¢Õ¡ÕªÕ¶Õ« Õ§Õ»Õ¨",
        ["Ô³Õ¬Õ­Õ¡Õ¾Õ¸Ö€ Õ§Õ»", "Õ„Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ« Õ¾Õ¥Ö€Õ¬Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶"],
        horizontal=True,
        key="nowcast_section",
    )

    if nowcast_section == "Ô³Õ¬Õ­Õ¡Õ¾Õ¸Ö€ Õ§Õ»":
        future_forecast = load_future_forecast()
        recent_actuals = load_recent_actual_quarters()
        if future_forecast is not None and not future_forecast.empty:
            st.markdown(
                """
                <div style="background:rgba(46,160,67,0.14); border-left:4px solid #2ea043; padding:14px 18px; border-radius:10px; margin:8px 0 18px 0;">
                <strong>2026Õ©. Q2-Q4 Õ€Õ†Ô±-Õ« Õ¡Õ¼Õ¡Õ»Õ¨Õ¶Õ©Õ¡Ö Õ¯Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´.</strong> Ô±ÕµÕ½ Õ¢Õ¡ÕªÕ«Õ¶Õ¨ Õ¶Õ¥Ö€Õ¯Õ¡ÕµÕ¡ÖÕ¶Õ¸Ö‚Õ´ Õ§ 2026Õ©. Õ¡Õ¼Õ¡Õ»Õ«Õ¶ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ« Õ¿Õ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¡Õ¯Õ¡Õ¶ ÖƒÕ¡Õ©Õ¥Õ©Õ« Õ°Õ«Õ´Õ¡Õ¶ Õ¾Ö€Õ¡ Õ½Õ¿Õ¡ÖÕ¾Õ¡Õ® Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¡ÕµÕ«Õ¶ Õ¯Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¨Ö‰
                </div>
                """,
                unsafe_allow_html=True,
            )

            forecast_metric_cols = st.columns(len(future_forecast))
            for idx, (_, row) in enumerate(future_forecast.iterrows()):
                forecast_metric_cols[idx].metric(
                    row["target_quarter"],
                    f"{row['forecast']:.2f}",
                    f"50% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„Õ {row['interval_lo_50']:.2f} â€“ {row['interval_hi_50']:.2f}",
                )

            forecast_table = future_forecast.copy()
            forecast_table["Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´"] = forecast_table["forecast"].map(lambda x: f"{x:.3f}")
            forecast_table["50% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„"] = forecast_table.apply(
                lambda row: f"{row['interval_lo_50']:.3f} - {row['interval_hi_50']:.3f}", axis=1
            )
            forecast_table["90% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„"] = forecast_table.apply(
                lambda row: f"{row['interval_lo_90']:.3f} - {row['interval_hi_90']:.3f}", axis=1
            )
            st.dataframe(
                forecast_table[["target_quarter", "Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´", "50% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„", "90% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„"]],
                width="stretch",
                hide_index=True,
            )

            forecast_chart = go.Figure()
            if recent_actuals is not None and not recent_actuals.empty:
                forecast_chart.add_trace(
                    go.Scatter(
                        x=recent_actuals["target_quarter"],
                        y=recent_actuals["actual"],
                        mode="lines+markers+text",
                        name="Õ“Õ¡Õ½Õ¿Õ¡ÖÕ« Õ€Õ†Ô±",
                        line=dict(color="#c9d1d9", width=3),
                        marker=dict(size=9, color="#c9d1d9"),
                        text=[f"{value:.1f}" for value in recent_actuals["actual"]],
                        textposition="top center",
                    )
                )
            forecast_chart.add_trace(
                go.Scatter(
                    x=future_forecast["target_quarter"],
                    y=future_forecast["interval_hi_90"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name="90% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„",
                )
            )
            forecast_chart.add_trace(
                go.Scatter(
                    x=future_forecast["target_quarter"],
                    y=future_forecast["interval_lo_90"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(88,166,255,0.14)",
                    hoverinfo="skip",
                    name="90% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„",
                )
            )
            forecast_chart.add_trace(
                go.Scatter(
                    x=future_forecast["target_quarter"],
                    y=future_forecast["interval_hi_50"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name="50% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„",
                )
            )
            forecast_chart.add_trace(
                go.Scatter(
                    x=future_forecast["target_quarter"],
                    y=future_forecast["interval_lo_50"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(46,160,67,0.24)",
                    hoverinfo="skip",
                    name="50% Õ´Õ«Õ»Õ¡Õ¯Õ¡ÕµÖ„",
                )
            )
            forecast_chart.add_trace(
                go.Scatter(
                    x=future_forecast["target_quarter"],
                    y=future_forecast["forecast"],
                    mode="lines+markers+text",
                    name="Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¾Õ¡Õ® Õ€Õ†Ô±",
                    line=dict(color="#f2cc60", width=4),
                    marker=dict(size=11, color="#f2cc60"),
                    text=[f"{value:.2f}" for value in future_forecast["forecast"]],
                    textposition="top center",
                )
            )
            if recent_actuals is not None and not recent_actuals.empty:
                last_actual = recent_actuals.iloc[-1]
                forecast_chart.add_trace(
                    go.Scatter(
                        x=[last_actual["target_quarter"]] + future_forecast["target_quarter"].tolist(),
                        y=[last_actual["actual"]] + future_forecast["forecast"].tolist(),
                        mode="lines",
                        line=dict(color="#f2cc60", width=4),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            forecast_chart.update_layout(
                title="2026Õ©. Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¡ÕµÕ«Õ¶ Õ€Õ†Ô±-Õ« Õ¯Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ´Õ¡Õ¶ Õ¸Ö‚Õ²Õ¥Õ£Õ«Õ®",
                xaxis_title="ÔµÕ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯",
                yaxis_title="Õ€Õ†Ô± YoY Õ«Õ¶Õ¤Õ¥Ö„Õ½",
            )
            st.plotly_chart(S(forecast_chart, h=460), width="stretch")

        st.stop()

    summary, predictions = load_nowcasting_results()
    ablation_summary, ablation_dm = load_google_ablation()
    if summary is None or predictions is None:
        st.error("Nowcasting Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¶Õ¥Ö€Õ¨ Õ¹Õ¥Õ¶ Õ£Õ¿Õ¶Õ¾Õ¥Õ¬Ö‰ ÕÕºÕ¡Õ½Õ¾Õ¸Ö‚Õ´ Õ¥Õ¶ `nowcasting_results/backtest_summary.csv` Ö‡ `nowcasting_results/backtest_predictions.csv` Ö†Õ¡ÕµÕ¬Õ¥Ö€Õ¨Ö‰")
    else:
        stage_order = ["Early", "Mid", "Late"]
        stage_names = {"Early": "ÕŽÕ¡Õ² ÖƒÕ¸Ö‚Õ¬", "Mid": "Õ„Õ«Õ»Õ«Õ¶ ÖƒÕ¸Ö‚Õ¬", "Late": "ÕˆÖ‚Õ· ÖƒÕ¸Ö‚Õ¬"}
        stage_colors = {"Early": "#1f6feb", "Mid": "#2ea043", "Late": "#ffb703"}

        best_by_stage = (
            summary.sort_values(["stage", "mape"])
            .groupby("stage", as_index=False)
            .first()
        )
        best_by_stage["stage"] = pd.Categorical(best_by_stage["stage"], categories=stage_order, ordered=True)
        best_by_stage = best_by_stage.sort_values("stage")

        model_stage_coverage = (
            summary.groupby("model", as_index=False)["stage"]
            .nunique()
            .rename(columns={"stage": "stage_count"})
        )
        summary_with_coverage = summary.merge(model_stage_coverage, on="model", how="left")
        operational_summary = summary_with_coverage[summary_with_coverage["stage_count"] == len(stage_order)].copy()

        overall_ranking = (
            operational_summary.groupby("model", as_index=False)["mape"]
            .mean()
            .rename(columns={"mape": "avg_mape"})
            .sort_values("avg_mape")
        )
        overall_winner = overall_ranking.iloc[0]
        model_options = (
            summary_with_coverage.groupby(["model", "stage_count"], as_index=False)["mape"]
            .mean()
            .rename(columns={"mape": "avg_mape"})
            .sort_values(["stage_count", "avg_mape"], ascending=[False, True])
        )
        default_model = (
            model_options.sort_values(["stage_count", "avg_mape"], ascending=[False, True]).iloc[0]["model"]
        )
        dfm_summary = summary[summary["model"] == "DFM"].copy().set_index("stage")
        early_winner = best_by_stage[best_by_stage["stage"] == "Early"].iloc[0]
        mid_winner = best_by_stage[best_by_stage["stage"] == "Mid"].iloc[0]
        late_winner = best_by_stage[best_by_stage["stage"] == "Late"].iloc[0]
        if mid_winner["model"] == late_winner["model"]:
            mid_late_summary = (
                f"<strong>Õ´Õ«Õ»Õ«Õ¶</strong> Ö‡ <strong>Õ¸Ö‚Õ· ÖƒÕ¸Ö‚Õ¬Õ¥Ö€Õ¸Ö‚Õ´</strong> Õ¡Õ¼Õ¡Õ»Õ¡Õ¿Õ¡Ö€ Õ§ "
                f"<strong>{mid_winner['model']}</strong>-Õ¨ Õ°Õ¡Õ´Õ¡ÕºÕ¡Õ¿Õ¡Õ½Õ­Õ¡Õ¶Õ¡Õ¢Õ¡Ö€ "
                f"<strong>{mid_winner['mape']:.2f}%</strong> Ö‡ <strong>{late_winner['mape']:.2f}%</strong> MAPE-Õ¸Õ¾Ö‰"
            )
        else:
            mid_late_summary = (
                f"<strong>Õ´Õ«Õ»Õ«Õ¶ ÖƒÕ¸Ö‚Õ¬Õ¸Ö‚Õ´</strong> Õ¡Õ¼Õ¡Õ»Õ¡Õ¿Õ¡Ö€ Õ§ <strong>{mid_winner['model']}</strong>-Õ¨ "
                f"(<strong>{mid_winner['mape']:.2f}%</strong> MAPE), Õ«Õ½Õ¯ "
                f"<strong>Õ¸Ö‚Õ· ÖƒÕ¸Ö‚Õ¬Õ¸Ö‚Õ´</strong>Õ <strong>{late_winner['model']}</strong>-Õ¨ "
                f"(<strong>{late_winner['mape']:.2f}%</strong> MAPE)Ö‰"
            )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Ô¼Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ£Õ¸Ö€Õ®Õ¡Õ¼Õ¶Õ¡Õ¯Õ¡Õ¶ Õ´Õ¸Õ¤Õ¥Õ¬", overall_winner["model"], f"MAPE {overall_winner['avg_mape']:.2f}%")
        for idx, (_, row) in enumerate(best_by_stage.iterrows(), start=1):
            metric_cols[idx].metric(stage_names[row["stage"]], row["model"], f"MAPE {row['mape']:.2f}%")

        st.markdown(
            f"""
            <div style="background:rgba(31,111,235,0.16); border-left:4px solid #58a6ff; padding:14px 18px; border-radius:10px; margin:8px 0 18px 0;">
            <strong>Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¶Õ¥Ö€Õ« Õ°Õ¡Õ´Õ¡Õ¼Õ¸Õ¿ Õ¢Õ¡ÖÕ¡Õ¿Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶.</strong> Backtest Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¶Õ¥Ö€Õ¸Õ¾ <strong>{overall_winner['model']}</strong>-Õ¨ Õ¬Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ£Õ¸Ö€Õ®Õ¡Õ¼Õ¶Õ¡Õ¯Õ¡Õ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶ Õ§,
            Ö„Õ¡Õ¶Õ« Õ¸Ö€ Õ¡ÕµÕ¶ Õ°Õ¡Õ½Õ¡Õ¶Õ¥Õ¬Õ« Õ§ Õ¢Õ¸Õ¬Õ¸Ö€ Õ¥Ö€Õ¥Ö„ ÖƒÕ¸Ö‚Õ¬Õ¥Ö€Õ¸Ö‚Õ´ Ö‡ Õ¸Ö‚Õ¶Õ« <strong>{overall_winner['avg_mape']:.2f}%</strong> Õ´Õ«Õ»Õ«Õ¶ MAPEÖ‰
            <strong>ÕŽÕ¡Õ² ÖƒÕ¸Ö‚Õ¬Õ¸Ö‚Õ´</strong> Õ¬Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¨ Õ£Ö€Õ¡Õ¶ÖÕ¥Õ¬ Õ§ <strong>{early_winner['model']}</strong>-Õ¨ ({early_winner['mape']:.2f}%),
            Õ«Õ½Õ¯ {mid_late_summary}
            <strong>DFM</strong>-Õ¨ ÕºÕ¡Õ°ÕºÕ¡Õ¶Õ¾Õ¸Ö‚Õ´ Õ§ Õ¸Ö€ÕºÕ¥Õ½ Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¡ÕµÕ«Õ¶ benchmark, Õ½Õ¡Õ¯Õ¡ÕµÕ¶ Õ¤Ö€Õ¡ Õ½Õ­Õ¡Õ¬Õ¨ Õ¡Õ¾Õ¥Õ¬Õ« Õ¢Õ¡Ö€Õ±Ö€ Õ§
            ({dfm_summary.loc['Early', 'mape']:.2f}%, {dfm_summary.loc['Mid', 'mape']:.2f}%, {dfm_summary.loc['Late', 'mape']:.2f}%),
            Õ¸Ö‚Õ½Õ¿Õ« Õ£Õ¸Ö€Õ®Õ¡Õ¼Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¯Õ«Ö€Õ¡Õ¼Õ´Õ¡Õ¶ Õ°Õ¡Õ´Õ¡Ö€ Õ¶Õ¡Õ­Õ¨Õ¶Õ¿Ö€Õ¥Õ¬Õ« Õ§ ensemble Õ´Õ¸Õ¿Õ¥ÖÕ¸Ö‚Õ´Õ¨Ö‰
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ablation_summary is not None and not ablation_summary.empty:
            early_ablation = (
                ablation_summary[ablation_summary["stage"] == "Early"]
                .copy()
                .sort_values("mape")
                .reset_index(drop=True)
            )
            dm_pair = ablation_dm[
                (ablation_dm["stage"] == "Early")
                & (
                    ((ablation_dm["model_a"] == "Base+Market") & (ablation_dm["model_b"] == "Base+Market+Google"))
                    | ((ablation_dm["model_a"] == "Base+Market+Google") & (ablation_dm["model_b"] == "Base+Market"))
                )
            ]
            dm_p_value = None if dm_pair.empty else float(dm_pair.iloc[0]["p_value"])
            market_mape = float(early_ablation.loc[early_ablation["model"] == "Base+Market", "mape"].iloc[0])
            full_mape = float(early_ablation.loc[early_ablation["model"] == "Base+Market+Google", "mape"].iloc[0])
            google_gain = market_mape - full_mape

            st.markdown("### Ô±ÕµÕ¬Õ¨Õ¶Õ¿Ö€Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ« Õ¡Õ¢Õ¬Õ¡ÖÕ«Õ¸Õ¶ Õ©Õ¥Õ½Õ¿")
            ablation_table = early_ablation.copy()
            ablation_table["ÕÕ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¬Õ¸Õ¯"] = ablation_table["model"].replace(
                {
                    "Base": "Ô²Õ¡Õ¦Õ¡ÕµÕ«Õ¶ Õ´Õ¸Õ¤Õ¥Õ¬",
                    "Base+Google": "Ô²Õ¡Õ¦Õ¡ÕµÕ«Õ¶ + Google Õ¯Õ¸Õ´ÕºÕ¸Õ¦Õ«Õ¿Õ¶Õ¥Ö€",
                    "Base+Market": "Ô²Õ¡Õ¦Õ¡ÕµÕ«Õ¶ + Õ·Õ¸Ö‚Õ¯Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ Õ¡Ö€Õ¡Õ£ ÖƒÕ¸ÖƒÕ¸Õ­Õ¡Õ¯Õ¡Õ¶Õ¶Õ¥Ö€",
                    "Base+Market+Google": "Ô²Õ¡Õ¦Õ¡ÕµÕ«Õ¶ + Õ·Õ¸Ö‚Õ¯Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ + Google Õ¯Õ¸Õ´ÕºÕ¸Õ¦Õ«Õ¿Õ¶Õ¥Ö€",
                }
            )
            ablation_table["MAPE"] = ablation_table["mape"].map(lambda x: f"{x:.3f}%")
            ablation_table["RMSE"] = ablation_table["rmse"].map(lambda x: f"{x:.3f}")
            st.dataframe(
                ablation_table[["ÕÕ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¬Õ¸Õ¯", "MAPE", "RMSE"]],
                width="stretch",
                hide_index=True,
            )

            dm_text = "n/a" if dm_p_value is None else f"{dm_p_value:.3f}"
            st.markdown(
                f"""
                <div style="background:rgba(46,160,67,0.10); border-left:4px solid #2ea043; padding:14px 18px; border-radius:10px; margin:4px 0 18px 0;">
                <strong>Ô¹Õ¡Ö€Õ´Õ¡ÖÕ¾Õ¡Õ® Õ´Õ¥Õ¯Õ¶Õ¡Õ¢Õ¡Õ¶Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶.</strong> Ô¹Õ¡Ö€Õ´Õ¡ÖÕ¾Õ¡Õ® <em>Early</em> Õ¡Õ¢Õ¬Õ¡ÖÕ«Õ¸Õ¶ Õ©Õ¥Õ½Õ¿Õ¸Ö‚Õ´
                Õ·Õ¸Ö‚Õ¯Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ Õ¡Ö€Õ¡Õ£ Õ¢Õ¬Õ¸Õ¯Õ¨ Õ¦Õ£Õ¡Õ¬Õ«Õ¸Ö€Õ¥Õ¶ Õ¢Õ¡Ö€Õ¥Õ¬Õ¡Õ¾Õ¸Ö‚Õ´ Õ§ Õ¢Õ¡Õ¦Õ¡ÕµÕ«Õ¶ Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¡ÕµÕ«Õ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¨,
                Õ«Õ½Õ¯ Google Õ¯Õ¸Õ´ÕºÕ¸Õ¦Õ«Õ¿Õ¶Õ¥Ö€Õ« Õ¡Õ¾Õ¥Õ¬Õ¡ÖÕ¸Ö‚Õ´Õ¨ Õ¤Ö€Õ¡ Õ¾Ö€Õ¡ Õ¿Õ¡Õ¬Õ«Õ½ Õ§ Õ´Õ«Õ¡ÕµÕ¶ ÖƒÕ¸Ö„Ö€ Õ¬Ö€Õ¡ÖÕ¸Ö‚ÖÕ«Õ¹ Õ·Õ¡Õ°Õ¸Ö‚ÕµÕ©`
                <strong>{google_gain:.3f}</strong> MAPE Õ¿Õ¸Õ¯Õ¸Õ½Õ¡ÕµÕ«Õ¶ Õ¯Õ¥Õ¿Ö‰
                Ô±ÕµÕ¤ Õ¡Õ¦Õ¤Õ¥ÖÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¨ Õ¤Õ¥Õ¼ Õ¾Õ«Õ³Õ¡Õ¯Õ¡Õ£Ö€Õ¸Ö€Õ¥Õ¶ Õ¾Õ³Õ¼Õ¡Õ¯Õ¡Õ¶ Õ¹Õ§, Ö„Õ¡Õ¶Õ« Õ¸Ö€ Diebold-Mariano Õ©Õ¥Õ½Õ¿Õ«
                <strong>p = {dm_text}</strong>Ö‰
                Ô±ÕµÕ½Õ«Õ¶Ö„Õ¶` Õ¡ÕµÕ¬Õ¨Õ¶Õ¿Ö€Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ« ÕºÕ¡Õ¿Õ´Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¨ Õ¤Õ¡Ö€Õ±Õ¥Õ¬ Õ§ Õ´Õ« ÖƒÕ¸Ö„Ö€ Õ¡Õ¾Õ¥Õ¬Õ« Õ¸Ö‚ÕªÕ¥Õ²,
                Õ¢Õ¡ÕµÖ Õ¡ÕµÕ¶ Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¸Ö‚Õ´ Õ§ Õ´Õ¶Õ¡Õ¬ Õ·Õ¥Ö€Õ¿Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¬Ö€Õ¡ÖÕ¸Ö‚Õ´, Õ¸Õ¹ Õ©Õ¥ Õ¸Ö€Õ¸Õ·Õ«Õ¹ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¦Õ¤Õ¡Õ¯Ö‰
                </div>
                """,
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns([1.2, 1])
        with col1:
            top_models = overall_ranking.head(6).sort_values("avg_mape", ascending=False)
            f_rank = go.Figure(
                go.Bar(
                    x=top_models["avg_mape"],
                    y=top_models["model"],
                    orientation="h",
                    marker_color="#58a6ff",
                    text=[f"{v:.2f}%" for v in top_models["avg_mape"]],
                    textposition="outside",
                )
            )
            f_rank.update_layout(title="Ô³Õ¸Ö€Õ®Õ¡Õ¼Õ¶Õ¡Õ¯Õ¡Õ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ« Õ¾Õ¡Ö€Õ¯Õ¡Õ¶Õ«Õ·Õ¨", xaxis_title="Õ„Õ«Õ»Õ«Õ¶ MAPE, %", yaxis_title="")
            st.plotly_chart(S(f_rank, h=420), width="stretch")

        with col2:
            f_stage = go.Figure(
                go.Bar(
                    x=[stage_names[s] for s in best_by_stage["stage"]],
                    y=best_by_stage["mape"],
                    marker_color=[stage_colors[s] for s in best_by_stage["stage"]],
                    text=[f"{v:.2f}%" for v in best_by_stage["mape"]],
                    textposition="outside",
                )
            )
            f_stage.update_layout(title="Ô¼Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¨ Õ¨Õ½Õ¿ ÖƒÕ¸Ö‚Õ¬Õ«", yaxis_title="MAPE, %", xaxis_title="")
            st.plotly_chart(S(f_stage, h=420), width="stretch")

        view_mode = st.radio(
            "Ô³Ö€Õ¡Ö†Õ«Õ¯Õ« Õ¼Õ¥ÕªÕ«Õ´Õ¨",
            ["Ô¼Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ¶ Õ¨Õ½Õ¿ ÖƒÕ¸Ö‚Õ¬Õ«", "Ô¸Õ¶Õ¿Ö€Õ¾Õ¡Õ® Õ´Õ¸Õ¤Õ¥Õ¬"],
            horizontal=True,
        )
        selected_model = st.selectbox(
            "Ô¸Õ¶Õ¿Ö€Õ¥Ö„ Õ´Õ¸Õ¤Õ¥Õ¬Õ¨",
            model_options["model"].tolist(),
            index=model_options["model"].tolist().index(default_model),
        )

        if view_mode == "Ô¼Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ¶ Õ¨Õ½Õ¿ ÖƒÕ¸Ö‚Õ¬Õ«":
            selected_predictions = predictions.merge(best_by_stage[["stage", "model"]].drop_duplicates(), on=["stage", "model"], how="inner")
            chart_title = "ÕŽÕ¥Ö€Õ»Õ«Õ¶ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¶Õ¥Ö€Õ« Õ¬Õ¡Õ¾Õ¡Õ£Õ¸Ö‚ÕµÕ¶ nowcast-Õ¥Ö€Õ¨"
            table_title = "ÕŽÕ¥Ö€Õ»Õ«Õ¶ Õ°Õ¡Õ½Õ¡Õ¶Õ¥Õ¬Õ« snapshot-Õ¨"
        else:
            selected_predictions = predictions[predictions["model"] == selected_model].copy()
            chart_title = f"{selected_model} Õ´Õ¸Õ¤Õ¥Õ¬Õ« nowcast-Õ¥Ö€Õ¨"
            table_title = f"{selected_model} Õ´Õ¸Õ¤Õ¥Õ¬Õ« Õ¾Õ¥Ö€Õ»Õ«Õ¶ snapshot-Õ¨"

            model_stage_summary = summary[summary["model"] == selected_model].copy()
            if not model_stage_summary.empty:
                model_stage_summary["Õ“Õ¸Ö‚Õ¬"] = model_stage_summary["stage"].map(stage_names)
                model_stage_summary["MAPE"] = model_stage_summary["mape"].map(lambda x: f"{x:.2f}%")
                model_stage_summary["RMSE"] = model_stage_summary["rmse"].map(lambda x: f"{x:.2f}")
                model_stage_summary["90% cover"] = model_stage_summary["coverage_90"].map(lambda x: f"{x * 100:.1f}%")
                st.dataframe(
                    model_stage_summary[["Õ“Õ¸Ö‚Õ¬", "MAPE", "RMSE", "90% cover"]],
                    width="stretch",
                    hide_index=True,
                )
                missing_stages = [stage_names[s] for s in stage_order if s not in model_stage_summary["stage"].tolist()]
                if missing_stages:
                    st.caption(f"Ô±ÕµÕ½ Õ´Õ¸Õ¤Õ¥Õ¬Õ¨ Õ°Õ¡Õ½Õ¡Õ¶Õ¥Õ¬Õ« Õ¹Õ§ Õ°Õ¥Õ¿Ö‡ÕµÕ¡Õ¬ ÖƒÕ¸Ö‚Õ¬Õ¥Ö€Õ¸Ö‚Õ´Õ {', '.join(missing_stages)}Ö‰")

        selected_predictions = selected_predictions.sort_values(["prediction_date", "stage"])
        available_quarters = selected_predictions["target_quarter"].dropna().drop_duplicates().tolist()
        default_quarters = available_quarters[-8:] if len(available_quarters) > 8 else available_quarters
        selected_quarters = st.multiselect(
            "Ô¸Õ¶Õ¿Ö€Õ¥Ö„ ÖÕ¸Ö‚ÖÕ¡Õ¤Ö€Õ¾Õ¸Õ² Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¶Õ¥Ö€Õ¨",
            available_quarters,
            default=default_quarters,
        )
        chart_predictions = selected_predictions[selected_predictions["target_quarter"].isin(selected_quarters)].copy()
        actual_recent = chart_predictions.sort_values("prediction_date").drop_duplicates("target_quarter")

        if chart_predictions.empty:
            st.warning("Ô³Ö€Õ¡Ö†Õ«Õ¯Õ¨ ÖÕ¸Ö‚ÖÕ¡Õ¤Ö€Õ¥Õ¬Õ¸Ö‚ Õ°Õ¡Õ´Õ¡Ö€ Õ¨Õ¶Õ¿Ö€Õ¥Ö„ Õ¡Õ¼Õ¶Õ¾Õ¡Õ¦Õ¶ Õ´Õ¥Õ¯ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Ö‰")
        else:
            f_recent = go.Figure()
            f_recent.add_trace(
                go.Scatter(
                    x=actual_recent["target_quarter"],
                    y=actual_recent["actual"],
                    name="Õ“Õ¡Õ½Õ¿Õ¡ÖÕ« Õ€Õ†Ô±",
                    line=dict(color="#c9d1d9", width=4),
                )
            )
            for stage in stage_order:
                stage_recent = chart_predictions[chart_predictions["stage"] == stage].sort_values("prediction_date")
                if stage_recent.empty:
                    continue
                f_recent.add_trace(
                    go.Scatter(
                        x=stage_recent["target_quarter"],
                        y=stage_recent["prediction"],
                        name=f"{stage_names[stage]} nowcast",
                        line=dict(color=stage_colors[stage], width=3),
                        mode="lines+markers",
                    )
                )
            f_recent.update_layout(title=chart_title, yaxis_title="Õ€Õ†Ô± YoY Õ«Õ¶Õ¤Õ¥Ö„Õ½", xaxis_title="")
            st.plotly_chart(S(f_recent, h=480), width="stretch")

        explorer_predictions = predictions.sort_values(["target_quarter", "stage", "model"]).copy()
        explorer_cols = st.columns([1, 1])
        quarter_filter_options = ["Ô²Õ¸Õ¬Õ¸Ö€ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¶Õ¥Ö€Õ¨"] + explorer_predictions["target_quarter"].dropna().drop_duplicates().tolist()
        with explorer_cols[0]:
            selected_quarter_filter = st.selectbox(
                "Ô´Õ«Õ¿Õ¥Õ¬ Õ¯Õ¸Õ¶Õ¯Ö€Õ¥Õ¿ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯",
                quarter_filter_options,
                index=max(0, len(quarter_filter_options) - 1),
            )

        available_model_rows = explorer_predictions.copy()
        if selected_quarter_filter != "Ô²Õ¸Õ¬Õ¸Ö€ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¶Õ¥Ö€Õ¨":
            available_model_rows = available_model_rows[available_model_rows["target_quarter"] == selected_quarter_filter]
        model_filter_options = ["Ô²Õ¸Õ¬Õ¸Ö€ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ¨"] + available_model_rows["model"].dropna().drop_duplicates().tolist()
        with explorer_cols[1]:
            selected_model_filter = st.selectbox(
                "Ô´Õ«Õ¿Õ¥Õ¬ Õ¯Õ¸Õ¶Õ¯Ö€Õ¥Õ¿ Õ´Õ¸Õ¤Õ¥Õ¬",
                model_filter_options,
                index=model_filter_options.index(selected_model) if selected_model in model_filter_options else 0,
            )

        latest_snapshot = explorer_predictions.copy()
        if selected_quarter_filter != "Ô²Õ¸Õ¬Õ¸Ö€ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¶Õ¥Ö€Õ¨":
            latest_snapshot = latest_snapshot[latest_snapshot["target_quarter"] == selected_quarter_filter]
        if selected_model_filter != "Ô²Õ¸Õ¬Õ¸Ö€ Õ´Õ¸Õ¤Õ¥Õ¬Õ¶Õ¥Ö€Õ¨":
            latest_snapshot = latest_snapshot[latest_snapshot["model"] == selected_model_filter]

        latest_snapshot = latest_snapshot.sort_values(["target_quarter", "stage", "model"]).copy()
        latest_snapshot["Õ“Õ¸Ö‚Õ¬"] = latest_snapshot["stage"].map(stage_names)
        latest_snapshot["Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ´Õ¡Õ¶ Õ¡Õ´Õ½Õ¡Õ©Õ«Õ¾"] = latest_snapshot["prediction_date"].dt.strftime("%Y-%m-%d")
        latest_snapshot["Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´"] = latest_snapshot["prediction"].map(lambda x: f"{x:.2f}")
        latest_snapshot["Õ“Õ¡Õ½Õ¿Õ¡ÖÕ«"] = latest_snapshot["actual"].map(lambda x: f"{x:.2f}")
        latest_snapshot["ÕÕ­Õ¡Õ¬"] = latest_snapshot["abs_pct_error"].map(lambda x: f"{x:.2f}%")
        latest_snapshot = latest_snapshot[["Õ“Õ¸Ö‚Õ¬", "model", "target_quarter", "Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ´Õ¡Õ¶ Õ¡Õ´Õ½Õ¡Õ©Õ«Õ¾", "Ô¿Õ¡Õ¶Õ­Õ¡Õ¿Õ¥Õ½Õ¸Ö‚Õ´", "Õ“Õ¡Õ½Õ¿Õ¡ÖÕ«", "ÕÕ­Õ¡Õ¬"]]
        latest_snapshot = latest_snapshot.rename(columns={"model": "Õ„Õ¸Õ¤Õ¥Õ¬", "target_quarter": "ÔµÕ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯"})
        st.subheader(table_title)
        if latest_snapshot.empty:
            st.info("Ô¸Õ¶Õ¿Ö€Õ¾Õ¡Õ® Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ« Ö‡ Õ´Õ¸Õ¤Õ¥Õ¬Õ« Õ°Õ¡Õ´Õ¡Ö€ Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€ Õ¹Õ¯Õ¡Õ¶Ö‰ Õ“Õ¸Õ­Õ¥Ö„ Ö†Õ«Õ¬Õ¿Ö€Õ¥Ö€Õ¨Ö‰")
        else:
            st.dataframe(latest_snapshot, width="stretch", hide_index=True)

elif page == "Õ„Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ¥Ö€Õ« Õ·Õ¡Ö€ÕªÕ¨Õ¶Õ©Õ¡ÖÕ¨":
    st.title(page)
    st.info("2025Õ©. Õ¿Õ¡Ö€Õ¥Õ¾Õ¥Ö€Õ»Õ«Õ¶ ÕºÕ¡Õ°ÕºÕ¡Õ¶Õ¾Õ¥Õ¬ Õ§ Õ°Õ¡Õ´Õ¡Õ·Õ­Õ¡Ö€Õ°Õ¡ÕµÕ«Õ¶ Õ©Õ¸Ö‚ÕµÕ¬ ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¡Ö€Õ¯Õ« Ö‡ Õ£Õ¥Ö€Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯Õ« ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¶Õ¥Ö€Õ¸Ö‚Õ´ Õ¶Õ¡Õ¾Õ©Õ« Õ£Õ¶Õ« Õ¶Õ¾Õ¡Õ¦Õ´Õ¡Õ¶ Õ´Õ«Õ¿Õ¸Ö‚Õ´Õ¨...\n\n2026Õ©â€¤ Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€Õ«Õ¶ Õ£Ö€Õ¡Õ¶ÖÕ¾Õ¡Õ® Õ¯Õ¿Ö€Õ¸Ö‚Õ¯ Õ¡Õ³Õ¨ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¡Õ·Õ­Õ¡Ö€Õ°Õ¡Ö„Õ¡Õ²Õ¡Ö„Õ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¸Õ¶Õ¶Õ¥Ö€Õ¸Õ¾ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ´Õ¡Õ¿Õ¡Õ¯Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ« Õ­Õ¡ÖƒÕ¡Õ¶Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ« Õ´Õ¿Õ¡Õ°Õ¸Õ£Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ¸Õ¾ (Ô±Õ„Õ†-ÕŽÕ¥Õ¶Õ¥Õ½Õ¸Ö‚Õ¥Õ¬Õ¡, Ô±Õ„Õ†-Ô»Ö€Õ¡Õ¶, Õ¡Õ¶Ö…Õ¤Õ¡Õ¹Õ¸Ö‚ Õ©Õ¼Õ¹Õ¸Õ² Õ½Õ¡Ö€Ö„Õ¥Ö€Õ« Õ°Õ¡Ö€Õ±Õ¡Õ¯Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¨ Ö‡ Õ¿Õ¥Õ­Õ¶Õ«Õ¯Õ¡Õ¯Õ¡Õ¶ Õ­Õ¶Õ¤Õ«Ö€Õ¶Õ¥Ö€Õ¨ Õ¶Õ¾Õ¡Õ¦Õ¥ÖÖ€Õ¥Õ¬ Õ¥Õ¶ Õ‚Õ¡Õ¦Õ¡Õ­Õ½Õ¿Õ¡Õ¶Õ« Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¨)Ö‰ 2026Õ©â€¤-Õ« Õ¡Õ¼Õ¡Õ»Õ«Õ¶ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ« Õ°Õ¡Õ´Õ¡Ö€ Õ•ÕŠÔµÔ¿+-Õ¨ Õ¤Õ¡Õ¤Õ¡Ö€Õ¥ÖÖ€Õ¥Õ¬ Õ§ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ­Õ©Õ¡Õ¶Õ¸Ö‚Õ´Õ¨..\n\n2025Õ©â€¤ ÕºÕ²Õ¶Õ±Õ« Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ« Õ¡Õ³Õ¨ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¡Õ¼Ö‡Õ¿Ö€Õ¡ÕµÕ«Õ¶ Ö„Õ¡Õ²Õ¡Ö„Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ¶Õ¸Ö€Õ¸Õ·Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ¸Õ¾, ÕºÕ²Õ¶Õ±Õ« Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯Õ« ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¶Õ¥Ö€Õ« Õ­Õ¡Õ©Õ¡Ö€Õ´Õ¡Õ´Õ¢ Ö‡ Õ¤Ö€Õ¡ Õ·Õ¸Ö‚Ö€Õ» Õ´Õ¿Õ¡Õ°Õ¸Õ£Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ¸Õ¾, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ Õ¤Õ¸Õ¬Õ¡Ö€Õ« Õ¤Õ«Ö€Ö„Õ« Õ©Õ¸Ö‚Õ¬Õ¡ÖÕ´Õ¡Õ´Õ¢: Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€Õ« Õ¯Õ¿Ö€Õ¸Ö‚Õ¯ Õ¡Õ³Õ¨Õ Ô±Õ„Õ†-Õ«Ö Õ¤Õ¸Ö‚Ö€Õ½ ÕºÕ¡Õ·Õ¡Ö€Õ¶Õ¥Ö€Õ« Õ½Õ¡Õ°Õ´Õ¡Õ¶Õ¡ÖƒÕ¡Õ¯Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶, Ô±Õ„Õ†-Õ« Õ¯Õ¸Õ²Õ´Õ«Ö Õ¥Õ¾Ö€Õ¸ÕºÕ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¨Õ¶Õ¯Õ¥Ö€Õ¶Õ¥Ö€Õ« Õ¶Õ¯Õ¡Õ¿Õ´Õ¡Õ´Õ¢ Õ´Õ¡Ö„Õ½Õ¡Õ¿Õ¸Ö‚Ö€Ö„Õ¥Ö€Õ« Õ¯Õ«Ö€Õ¼Õ¡Õ´Õ¡Õ¶ Õ·Õ¸Ö‚Ö€Õ» Õ¡Õ¶Õ°Õ¡Õ¶Õ£Õ½Õ¿Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¥Ö€, Õ´ÕµÕ¸Ö‚Õ½ Õ¯Õ¸Õ²Õ´Õ«Ö Õ£Õ¶Õ¥Ö€Õ« Õ¡Õ³Õ¨ Õ¡Õ¦Õ¤Õ¥Õ¬ Õ§ Õ‰Õ«Õ¶Õ¡Õ½Õ¿Õ¡Õ¶Õ« Õ¯Õ¸Õ²Õ´Õ«Ö Õ´Õ¥Õ¿Õ¡Õ²Õ« Õ¶Õ¯Õ¡Õ¿Õ´Õ¡Õ´Õ¢ ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¡Ö€Õ¯Õ« Õ¯Ö€Õ³Õ¡Õ¿Õ´Õ¡Õ¶Õ¨Ö‰\n\nÕˆÕ½Õ¯Õ¸Ö‚ Õ£Õ¶Õ« Õ¡Õ³Õ¨ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¥Õ²Õ¥Õ¬ Õ´Õ¡Ö„Õ½Õ¡Õ¿Õ¸Ö‚Ö€Ö„Õ¥Ö€Õ« Õ¡Õ¶Õ¸Ö€Õ¸Õ·Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ¸Õ¾, Ô±Õ„Õ† Õ¤Õ¸Õ¬Õ¡Ö€Õ« Õ¶Õ¯Õ¡Õ¿Õ´Õ¡Õ´Õ¢ ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¡Ö€Õ¯Õ« Õ¶Õ¾Õ¡Õ¦Õ´Õ¡Õ´Õ¢, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ Õ¢Õ¸Ö€Õ½Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ Ö†Õ¸Õ¶Õ¤Õ¥Ö€Õ« Ö‡ Õ¯Õ¥Õ¶Õ¿Ö€Õ¸Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¡Õ¶Õ¯Õ¥Ö€Õ« Õ¯Õ¸Õ²Õ´Õ«Ö Õ¸Õ½Õ¯Õ¸Ö‚ Õ´Õ¥Õ® ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¡Õ¯Õ¸Õ¾Ö‰\n\nÕŠÕ¡Ö€Õ¥Õ¶Õ« Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ« Õ£Õ¶Õ¡Õ¶Õ¯Õ¸Ö‚Õ´Õ¡ÕµÕ«Õ¶ Õ´Õ«Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¨ Õ©Õ¸Ö‚Õ¬Õ¡ÖÕ¥Õ¬ Õ¥Õ¶Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯Õ« Õ£Õ¸Ö€Õ®Õ¸Õ¶Õ¶Õ¥Ö€Õ¸Õ¾Ö‰")
    c1, c2, c3 = st.columns([1, 1, 1.5])
    df1_1 = load_data('p1_commodities.csv')
    idx37 = pd.date_range("2023-01-01", periods=len(df1_1), freq="MS")
    # Generate ticktext for dates
    date_ticks = [f"{translate_p(d.month)} {d.year}Õ©." for d in idx37]
    
    cu = df1_1['cu']
    oil = df1_1['oil']
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=idx37, y=cu, name="ÕŠÕ²Õ«Õ¶Õ± ($/Õ¿)", line=dict(color="#1f6feb", width=3)))
    f1.add_trace(go.Scatter(x=idx37, y=oil, name="Õ†Õ¡Õ¾Õ© ($/Õ¢)", yaxis="y2", line=dict(color="#ff9f43", width=3)))
    f1.update_layout(title="Õ†Õ¡Õ¾Õ©Õ« Ö‡ ÕŠÕ²Õ¶Õ±Õ« Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ¥Ö€", yaxis=dict(title="ÕŠÕ²Õ«Õ¶Õ±"), yaxis2=dict(title="Õ†Õ¡Õ¾Õ©", overlaying="y", side="right"),
                     xaxis=dict(tickmode="array", tickvals=idx37[::4], ticktext=date_ticks[::4]))
    c1.plotly_chart(S(f1), width="stretch")
    au = load_data('p1_commodities.csv')['gold']
    f2 = go.Figure(go.Scatter(x=idx37, y=au, line=dict(color="#f2cc60", width=4), name="ÕˆÕ½Õ¯Õ« ($/Õ¸Ö‚Õ¶ÖÕ«Õ¡)"))
    f2.update_layout(title="ÕˆÕ½Õ¯Õ¸Ö‚ Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ«Õ¶", xaxis=dict(tickmode="array", tickvals=idx37[::4], ticktext=date_ticks[::4]))
    c2.plotly_chart(S(f2), width="stretch")
    m = [f"{i}" for i in [1,3,5,7,9,11,1,3,5,7,9,11,1]]
    df1_2 = load_data('p1_food.csv')
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['meat'], name="Õ„Õ«Õ½", line=dict(color="#d73027", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['oil'], name="Ô²Õ¸Ö‚Õ½Õ¡Õ¯Õ¡Õ¶ ÕµÕ¸Ö‚Õ²Õ¥Ö€", line=dict(color="#00ffff", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['food'], name="ÕŠÕ¡Ö€Õ¥Õ¶Õ« Õ£Õ¶Õ« Õ°Õ¡Õ´Õ¡Õ©Õ«Õ¾", line=dict(color="#4575b4", width=4)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['dairy'], name="Ô¿Õ¡Õ©Õ¶Õ¡Õ´Õ©Õ¥Ö€Ö„", line=dict(color="#74add1", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['sugar'], name="Õ‡Õ¡Ö„Õ¡Ö€Õ¡Õ¾Õ¡Õ¦", line=dict(color="#fdae61", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['cereals'], name="Õ€Õ¡ÖÕ¡Õ°Õ¡Õ¿Õ«Õ¯", line=dict(color="#b2abd2", width=2)))
    f3.update_layout(title="ÕŠÕ¡Ö€Õ¥Õ¶Õ« Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ£Õ¶Õ¥Ö€", xaxis=dict(tickvals=list(range(13)), ticktext=m), yaxis_title="Ô»Õ¶Õ¤Õ¥Ö„Õ½, 2024Õ©. Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€=100")
    c3.plotly_chart(S(f3), width="stretch")

elif page == "Õ€Õ€ Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶":
    st.title(page)
    st.info("2025Õ©.-Õ«Õ¶ Õ€Õ€ Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 2.5%-Õ¸Õ¾, Õ¸Ö€Õ¨ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¥Õ²Õ¥Õ¬ Õ¡ÕµÕ¬ Õ¥Ö€Õ¯Ö€Õ¶Õ¥Ö€Õ«Ö Õ€Õ€ Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¡Õ³Õ¸Õ¾:\n\nÔ±ÕµÕ¬ Õ¥Ö€Õ¯Ö€Õ¶Õ¥Ö€Õ«Ö Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ Õ·Õ¸Ö‚Ö€Õ» 5.6%-Õ¸Õ¾Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® ÕŽÖ€Õ¡Õ½Õ¿Õ¡Õ¶Õ«Ö, Õ‰Õ«Õ¶Õ¡Õ½Õ¿Õ¡Õ¶Õ«Ö, Õ–Ö€Õ¡Õ¶Õ½Õ«Õ¡ÕµÕ«Ö, Ô»Ö€Õ¡Õ¶Õ«Ö Ö‡ Õ¡ÕµÕ¬ Õ¥Ö€Õ¯Ö€Õ¶Õ¥Ö€Õ«Ö Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¡Õ³Õ¸Õ¾:\n\nÔ±Õ³Õ«Õ¶ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ°Õ¡Õ¯Õ¡Õ¦Õ¤Õ¥Õ¬ Õ§ Õ€Õ¶Õ¤Õ¯Õ¡Õ½Õ¿Õ¡Õ¶Õ«Ö Ö‡ ÕŒÔ´-Õ«Ö Õ¡ÕµÖÕ¥Õ¬Õ¡Õ® Õ¦Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¶Õ¾Õ¡Õ¦Õ¸Ö‚Õ´Õ¨, Õ°Õ¡Õ´Õ¡ÕºÕ¡Õ¿Õ¡Õ½Õ­Õ¡Õ¶Õ¡Õ¢Õ¡Ö€Õ 0.7 Ö‡ 0.03 Õ¿Õ¸Õ¯Õ¸Õ½Õ¡ÕµÕ«Õ¶ Õ¯Õ¥Õ¿Õ¥Ö€Õ¸Õ¾:")
    c1, c2 = st.columns(2)
    df_t = load_data('p2_tourism_counts.csv')
    df_t['ÕÕ¡Ö€Õ¥Õ©Õ«Õ¾'] = df_t['ÕÕ¡Ö€Õ¥Õ©Õ«Õ¾'].astype(str)
    f1 = px.bar(df_t, x="ÕˆÖ‚Õ²Õ²Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", y="Õ”Õ¡Õ¶Õ¡Õ¯", color="ÕÕ¡Ö€Õ¥Õ©Õ«Õ¾", barmode="group", title="Ô¶Õ¢Õ¸Õ½Õ¡Õ·Ö€Õ»Õ«Õ¯Õ¶Õ¥Ö€ (Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤)")
    f1.update_traces(texttemplate="%{y}", textposition="outside")
    c1.plotly_chart(S(f1), width="stretch")
    df2_2 = load_data('p2_tourism_growth.csv')
    f2 = go.Figure(go.Bar(x=df2_2['ÔµÖ€Õ¯Õ«Ö€'], y=df2_2['Ô±Õ³'], marker_color=["#1f6feb", "#c00000", "#7ee787"]*2, text=[f"{v:+.1f}%" for v in df2_2['Ô±Õ³']], textposition="outside"))
    f2.update_layout(title="Ô±Õ³ 2024â€“2025 (%)")
    c2.plotly_chart(S(f2), width="stretch")

elif page == "Ô´Ö€Õ¡Õ´Õ¡Õ¯Õ¡Õ¶ ÖƒÕ¸Õ­Õ¡Õ¶ÖÕ¸Ö‚Õ´Õ¶Õ¥Ö€Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶":
    st.title(page)
    st.info("2025Õ©. Ö†Õ«Õ¦Õ«Õ¯Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¶Õ±Õ¡Õ¶Ö ÖƒÕ¸Õ­Õ¡Õ¶ÖÕ¸Ö‚Õ´Õ¶Õ¥Ö€Õ« Õ¦Õ¸Ö‚Õ¿ Õ¶Õ¥Ö€Õ°Õ¸Õ½Ö„Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 8.6%-Õ¸Õ¾, Õ¨Õ¶Õ¤ Õ¸Ö€Õ¸Ö‚Õ´ Õ¶Õ¥Ö€Õ°Õ¸Õ½Ö„Õ¶ Õ¡Õ¾Õ¥Õ¬Õ¡ÖÕ¥Õ¬ Õ§ 2.4%-Õ¸Õ¾, Õ«Õ½Õ¯ Õ¡Ö€Õ¿Õ¡Õ°Õ¸Õ½Ö„Õ¨Õ 0.3%-Õ¸Õ¾:")
    c1, c2 = st.columns([1.5, 1])
    df3 = load_data('p3_remittances.csv')
    m = df3['month'].tolist()
    
    # Chart 1: Combined Inflow (Õ†Õ¥Ö€hosq) + Outflow (Artahosq) side by side with vertical divider
    from plotly.subplots import make_subplots
    f1 = make_subplots(rows=1, cols=2, subplot_titles=["Õ†Õ¥Ö€Õ°Õ¸Õ½Ö„", "Ô±Ö€Õ¿Õ¡Õ°Õ¸Õ½Ö„"], horizontal_spacing=0.08)
    
    # Left subplot: Inflow
    f1.add_trace(go.Bar(x=[translate_p(x) for x in m], y=df3['in_2025'], name="2025Õ©.", marker_color="#5b9bd5", showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2024'], name="2024Õ©.", line=dict(color="#adbac7", width=2), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2023'], name="2023Õ©.", line=dict(color="#ffc000", width=2, dash="dash"), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2022'], name="2022Õ©.", line=dict(color="#ff0000", width=2, dash="dot"), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2021'], name="2021Õ©.", line=dict(color="#808080", width=1, dash="dot"), showlegend=True), row=1, col=1)
    
    # Right subplot: Outflow
    f1.add_trace(go.Bar(x=[translate_p(x) for x in m], y=df3['out_2025'], name="2025Õ©.", marker_color="#5b9bd5", showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2024'], name="2024Õ©.", line=dict(color="#adbac7", width=2), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2023'], name="2023Õ©.", line=dict(color="#ffc000", width=2, dash="dash"), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2022'], name="2022Õ©.", line=dict(color="#ff0000", width=2, dash="dot"), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2021'], name="2021Õ©.", line=dict(color="#808080", width=1, dash="dot"), showlegend=False), row=1, col=2)
    
    f1.update_yaxes(range=[0, 7000], row=1, col=1)
    f1.update_yaxes(range=[0, 7000], row=1, col=2)
    f1.update_layout(title="Õ–Õ«Õ¦. Õ¡Õ¶Õ±Õ¡Õ¶Ö Õ¤Ö€Õ¡Õ´Õ¡Õ¯Õ¡Õ¶ ÖƒÕ¸Õ­Õ¡Õ¶ÖÕ¸Ö‚Õ´Õ¶Õ¥Ö€", yaxis_title="Õ„Õ¬Õ¶ Õ¤Õ¸Õ¬Õ¡Ö€", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    c1.plotly_chart(S(f1, h=500), width="stretch")

    # Chart 2: Net Inflow (Ô¶Õ¸Ö‚Õ¿ Õ¶Õ¥Ö€Õ°Õ¸Õ½Ö„)
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=m, y=df3['net_2025'], name="2025", marker_color="#5b9bd5"))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2024'], name="2024", line=dict(color="#adbac7", width=2)))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2023'], name="2023", line=dict(color="#ffc000", width=2, dash="dash")))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2022'], name="2022", line=dict(color="#ff0000", width=2, dash="dot")))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2021'], name="2021", line=dict(color="#808080", width=1, dash="dot")))
    f2.update_layout(title="Õ–Õ«Õ¦. Õ¡Õ¶Õ±Õ¡Õ¶Ö ÖƒÕ¸Õ­Õ¡Õ¶ÖÕ¸Ö‚Õ´Õ¶Õ¥Ö€, Õ¦Õ¸Ö‚Õ¿ Õ¶Õ¥Ö€Õ°Õ¸Õ½Ö„", yaxis=dict(range=[0, 3000]), yaxis_title="Õ„Õ¬Õ¶ Õ¤Õ¸Õ¬Õ¡Ö€", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    c2.plotly_chart(S(f2, h=500), width="stretch")

elif page == "ÕÕ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¯Õ¿Õ«Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. ÕÔ±Õ‘-Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 9.2%-Õ¸Õ¾` ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ®Õ¡Õ¼Õ¡ÕµÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« (Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´Õ Ö†Õ«Õ¶Õ¡Õ¶Õ½Õ¡Õ¯Õ¡Õ¶ Ö‡ Õ¡ÕºÕ¡Õ°Õ¸Õ¾Õ¡Õ£Ö€Õ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ¶Õ¥Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶, Õ¿Õ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Õ¯Õ¡Õº) Ö‡ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¥Ö€Õ¸Õ¾:\n\nÕÕ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³Õ¨ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ¯Õ¥Õ¶Õ¿Ö€Õ¸Õ¶Õ¡ÖÕ¾Õ¡Õ® Õ§ Ö†Õ«Õ¶Õ¡Õ¶Õ½Õ¡Õ¯Õ¡Õ¶ Ö‡ Õ¡ÕºÕ¡Õ°Õ¸Õ¾Õ¡Õ£Ö€Õ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ¶Õ¥Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶, Õ¿Õ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Õ¯Õ¡Õº, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ« Õ·Õ¸Ö‚Ö€Õ»:")
    c1, c2, c3 = st.columns(3)
    df4_1 = load_data('p4_eai_quarterly.csv')
    qu = df4_1['quarter'].tolist()
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=qu, y=df4_1['eai'], name="ÕÔ±Õ‘", marker=dict(size=10, color="#5b9bd5"), line=dict(color="#5b9bd5", width=3), mode="lines+markers+text", text=df4_1['eai'].apply(lambda v: f'{v:.1f}').tolist(), textposition="top center"))
    # Add the 2024 "TAC without gold" dashed comparison line
    if 'eai_nosk' in df4_1.columns:
        f1.add_trace(go.Scatter(x=qu, y=df4_1['eai_nosk'], name="ÕÔ±Õ‘Õ Õ¡Õ¼Õ¡Õ¶Ö Õ¸Õ½Õ¯Õ«", line=dict(color="#c00000", width=3, dash="dash"), mode="lines+markers+text", text=df4_1['eai_nosk'].apply(lambda v: f'{v:.1f}').tolist(), textposition="bottom center"))
    f1.update_layout(title="ÕÕ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¯Õ¿Õ«Õ¾Õ¤Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶<br>Ô»Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³Õ¨, %")
    c1.plotly_chart(S(f1), width="stretch")
    g_lab = ["Õ€Õ†Ô±", "Õ–Õ«Õ¶. Ö‡ Õ¡ÕºÕ¡Õ°Õ¸Õ¾. Õ£Õ¸Ö€Õ®.", "Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Õ£Õ¸Ö‚Õ¿ Õ¡Õ¶Õ¸Ö‚Õ¤Ö€Õ¡Õ¯Õ« Õ°Õ¡Ö€Õ¯Õ¥Ö€", "ÕÕ¥Õ²Õ¥Õ¯. Ö‡ Õ¯Õ¡Õº", "Ô±Õ¶Õ·Õ¡Ö€Õª Õ£.", "Õ„Õ·. Õ¡Ö€Õ¤."]
    g_val = [6.0, 1.4, 1.3, 1.2, 1.1, 0.5, -0.6]
    g_col = ["#1f6feb", "#2e6db4", "#2e6db4", "#00b050", "#2e6db4", "#2e6db4", "#c00000"]
    f2 = go.Figure(go.Bar(x=g_val, y=g_lab, orientation="h", marker_color=g_col, text=g_val, textposition="outside"))
    f2.update_layout(title="Õ€Õ¡Õ´Õ¡Õ­Õ¡Õ¼Õ¶ Õ¶Õ¥Ö€Ö„Õ«Õ¶ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„<br>Ô»Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³Õ¨ (%) Ö‡ Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¨ (Õ¿.Õ¯.)\n(Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ½Õ¥ÕºÕ¿Õ¥Õ´Õ¢Õ¥Ö€)")
    c2.plotly_chart(S(f2), width="stretch")
    df4_3 = load_data('p4_sectors.csv')
    sec = df4_3['sector'].tolist()[::-1]
    ach = df4_3['growth'].tolist()[::-1]
    npas = df4_3['contribution'].tolist()[::-1]
    ach_text = [str(v) if not pd.isna(v) else "" for v in ach]
    npas_text = [str(v) if not pd.isna(v) else "" for v in npas]
    
    f3 = go.Figure()
    f3.add_trace(go.Bar(y=sec, x=ach, name="Õ¡Õ³, %", orientation="h", marker_color="#cc0000", text=ach_text, textposition="outside"))
    f3.add_trace(go.Bar(y=sec, x=npas, name="Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´, Õ¿.Õ¯.", orientation="h", marker_color="#3182bd", text=npas_text, textposition="outside"))
    f3.update_layout(title="ÕÕ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¯Õ¿Õ«Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (Ô±Õ³ Ö‡ Õ†ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´)", barmode="group", xaxis=dict(range=[0, 25]))
    c3.plotly_chart(S(f3), width="stretch")

elif page == "Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¨ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ§ 4.7%Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ´Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¸Õ¾:\n\nÕ„Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¶ Õ«Ö€ Õ°Õ¥Ö€Õ©Õ«Õ¶ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¥Õ²Õ¥Õ¬ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ®Õ­Õ¡Õ­Õ¸Õ¿Õ¡ÕµÕ«Õ¶ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¿Õ¥Õ½Õ¡Õ¯Õ¶Õ¥Ö€Õ« Ö‡ Õ½Õ¶Õ¶Õ¤Õ¡Õ´Õ©Õ¥Ö€Ö„Õ«* Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¸Õ¾:")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    df5_1 = load_data('p5_industry.csv')
    m_lbl = [translate_p(x) for x in df5_1['month'].tolist()]
    y25 = df5_1['val_2025'].tolist()
    y24 = df5_1['val_2024'].tolist()
    
    # Ensure mapping handles any floating inaccuracies by string conversion during plot
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['water'], name="Õ‹Ö€Õ¡Õ´Õ¡Õ¿Õ¡Õ¯Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ´", marker_color="#fcae91"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['energy'], name="Ô·Õ¬Õ¥Õ¯Õ¿Ö€Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", marker_color="#de2d26"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['manuf'], name="Õ„Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€.", marker_color="#ccece6"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['mining'], name="Õ€Õ¡Õ¶Ö„Õ¡Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", marker_color="#183b66"))
    f1.add_trace(go.Scatter(x=m_lbl, y=y24, name="Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢.-2024Õ©.", line=dict(color="#ff9900", width=3, dash="dash")))
    f1.add_trace(go.Scatter(x=m_lbl, y=y25, name="Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢.-2025Õ©.", line=dict(color="#3182bd", width=4), mode="lines+markers+text", text=y25, textposition="top center"))
    f1.update_layout(barmode="relative", title="Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (Ô±Õ³, %)", legend=dict(orientation="h", y=-0.4, font=dict(size=10)))
    c1.plotly_chart(S(f1, h=550), width="stretch")
    
    s_lab = ["Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€.", "Õ„Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€.", "Ô·Õ¬Õ¥Õ¯., Õ£Õ¡Õ¦Õ«... Õ´Õ¡Õ¿.", "Õ€Õ¡Õ¶Ö„Õ¡Õ£Õ¸Ö€Õ®."]
    s_lab.reverse()
    val = [4.7, 2.4, 1.3, 1.0]
    val.reverse()
    f2 = go.Figure(go.Bar(x=val, y=s_lab, orientation="h", marker_color=["#3182bd", "#3182bd", "#3182bd", "#cc0000"], text=val, textposition="outside"))
    f2.update_layout(title="Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€)", xaxis=dict(range=[-3, 6]))
    c2.plotly_chart(S(f2, h=550), width="stretch")
    
    df5_2 = load_data('p5_manufacturing.csv')
    sub = df5_2['sector'].tolist()[::-1]
    v_sub = df5_2['val'].tolist()[::-1]
    n = len(v_sub)
    mfg_colors = ["#cc0000" if v < 0 else "#3182bd" for v in v_sub]
    f3 = go.Figure(go.Bar(x=v_sub, y=sub, orientation="h", marker_color=mfg_colors, text=v_sub, textposition="outside"))
    f3.update_layout(title="Õ„Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶<br>Ô»Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³, % Ö‡ Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´, Õ¿.Õ¯.<br>(Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€)", xaxis=dict(range=["auto", "auto"]))
    c3.plotly_chart(S(f3, h=550), width="stretch")

elif page == "Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. Õ£ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 5.6%-Õ¸Õ¾Õ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¢Õ¸Ö‚Õ½Õ¡Õ¢Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¸Õ¾:\n\nÔ²Õ¸Ö‚Õ½Õ¡Õ¢Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¶ Õ«Ö€ Õ°Õ¥Ö€Õ©Õ«Õ¶ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¥Õ²Õ¥Õ¬ Õ­Õ¡Õ²Õ¸Õ²Õ«, ÕºÕ¿Õ²Õ« Ö‡ Õ°Õ¡Õ¿Õ¡ÕºÕ¿Õ²Õ«, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ Õ°Õ¡ÖÕ¡Õ°Õ¡Õ¿Õ«Õ¯Õ« Ö‡ Õ°Õ¡Õ¿Õ«Õ¯Õ¡Õ¨Õ¶Õ¤Õ¥Õ²Õ¥Õ¶Õ« Õ¡Õ³Õ¸Õ¾:")
    c1, c2 = st.columns([1.5, 1])
    df6_1 = load_data('p6_agriculture.csv')
    per = df6_1['period'].tolist()
    l25 = df6_1['l25'].tolist()
    l24 = df6_1['l24'].tolist()
    
    f1 = go.Figure()
    f1.add_trace(go.Bar(name="Õ¢Õ¸Ö‚Õ½Õ¡Õ¢Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", x=per, y=df6_1['crop'], marker_color="#b3cde3"))
    f1.add_trace(go.Bar(name="Õ¡Õ¶Õ¡Õ½Õ¶Õ¡Õ¢Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", x=per, y=df6_1['animal'], marker_color="#fdd0a2"))
    f1.add_trace(go.Bar(name="Õ¡Õ¶Õ¿Õ¡Õ¼Õ¡ÕµÕ«Õ¶ Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", x=per, y=df6_1['forest'], marker_color="#fbb4b9"))
    f1.add_trace(go.Bar(name="Õ±Õ¯Õ¶Õ¸Ö€Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", x=per, y=df6_1['fish'], marker_color="#ccebc5"))
    
    f1.add_trace(go.Scatter(x=per, y=l24, name="Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶-2024", line=dict(color="#ff9900", width=3, dash="dot")))
    f1.add_trace(go.Scatter(x=per, y=l25, name="Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶-2025", line=dict(color="#3182bd", width=4), mode="lines+markers+text", text=l25, textposition="top center", marker=dict(color="#cc0000", size=8)))
    
    f1.update_layout(barmode="stack", title="Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (Õ¡Õ³ Ö‡ Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€)", legend=dict(orientation="v", y=1, x=0.7, font=dict(size=11)))
    c1.plotly_chart(S(f1, h=550), width="stretch")
    
    df6_2 = load_data('p6_sectors.csv')
    a_lab = df6_2['sector'].tolist()[::-1]
    v_a = df6_2['growth'].tolist()[::-1]
    n6 = len(v_a)
    agr_colors = ["#cc0000" if v < 0 else "#0070c0" for v in v_a]
    # Override: last item (Amboxj gyugh) should be red indicating total
    agr_colors[-1] = "#cc0000"
    f2 = go.Figure(go.Bar(x=v_a, y=a_lab, orientation="h", marker_color=agr_colors, text=[f'{v:.1f}' for v in v_a], textposition="outside"))
    f2.update_layout(title="Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶<br>Ô»Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³, % Ö‡ Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€, Õ¿.Õ¯.<br>(Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€)", xaxis=dict(range=["auto", "auto"]))
    c2.plotly_chart(S(f2, h=550), width="stretch")

elif page == "Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ¨ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ§ 20.2%` ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¨Õ½Õ¿ Ö†Õ«Õ¶Õ¡Õ¶Õ½Õ¡Õ¾Õ¸Ö€Õ´Õ¡Õ¶ Õ¡Õ²Õ¢ÕµÕ¸Ö‚Ö€Õ¶Õ¥Ö€Õ«Õ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ ÕºÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ«, Õ¯Õ¡Õ¦Õ´Õ¡Õ¯Õ¥Ö€ÕºÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ«, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ«Õ»Õ¸ÖÕ¶Õ¥Ö€Õ¸Õ¾ Õ«Ö€Õ¡Õ¯Õ¡Õ¶Õ¡ÖÕ¾Õ¡Õ® Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¶Õ¥Ö€Õ« Õ¡Õ³Õ¸Õ¾, Õ«Õ½Õ¯ Õ¨Õ½Õ¿ Õ¿Õ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ¶Õ¥Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¿Õ¥Õ½Õ¡Õ¯Õ¶Õ¥Ö€Õ«Õ Õ¡Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ°Õ¥Õ¿ Õ¯Õ¡ÕºÕ¾Õ¡Õ® Õ£Õ¸Ö€Õ®Õ¸Ö‚Õ¶Õ¥Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Ö‡ Õ¯Ö€Õ©Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ¸Ö‚Õ´ Õ«Ö€Õ¡Õ¯Õ¡Õ¶Õ¡ÖÕ¾Õ¡Õ® Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¶Õ¥Ö€Õ« Õ¡Õ³Õ¸Õ¾:")
    c1, c2 = st.columns([1.5, 1])
    
    # Chart 1: Stacked Bar Chart for Months I-XII
    df7_1 = load_data('p7_construction_monthly.csv')
    m_lbl = [translate_p(x) for x in df7_1['month'].tolist()]
    y25 = df7_1['gr25'].tolist()
    y24 = df7_1['gr24'].tolist()
    
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['pop'], name="Õ¢Õ¶Õ¡Õ¯Õ¹. Õ´Õ«Õ»Õ¸ÖÕ¶Õ¥Ö€", marker_color="#b3cde3"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['org'], name="Õ¯Õ¡Õ¦Õ´. Õ´Õ«Õ»Õ¸ÖÕ¶Õ¥Ö€", marker_color="#ccebc5"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['hum'], name="Õ´Õ¡Ö€Õ¤. Ö…Õ£Õ¶. Õ´Õ«Õ»Õ¸ÖÕ¶Õ¥Ö€", marker_color="#cccccc"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['comm'], name="Õ°Õ¡Õ´Õ¡ÕµÕ¶Ö„Õ¶Õ¥Ö€Õ« Õ´Õ«Õ»Õ¸ÖÕ¶Õ¥Ö€", marker_color="#ffcf20"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['state'], name="ÕºÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢ÕµÕ¸Ö‚Õ»Õ¥", marker_color="#4178c7"))
    
    f1.add_trace(go.Scatter(x=m_lbl, y=y24, name="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶-2024Õ©.", line=dict(color="#ff9900", width=3, dash="dot")))
    f1.add_trace(go.Scatter(x=m_lbl, y=y25, name="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶-2025Õ©.", line=dict(color="#0066cc", width=3), mode="lines+markers+text", text=y25, textposition="top center", marker=dict(color="#cc0000", size=7)))
    f1.update_layout(barmode="relative", title="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", legend=dict(orientation="v", y=1, x=0.01, font=dict(size=10)))
    c1.plotly_chart(S(f1, h=650), width="stretch")
    
    # Create an inner column structure in the right column for the two side charts
    rc1 = c2.container()
    
    df7_2 = load_data('p7_funding.csv')
    fin_lab = df7_2['source'].tolist()[::-1]
    fin_val = df7_2['val'].tolist()[::-1]
    f2 = go.Figure(go.Bar(x=fin_val, y=fin_lab, orientation="h", marker_color=["#0070c0"] * 4 + ["#cc0000"], text=fin_val, textposition="outside"))
    f2.update_layout(title="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶ Õ¨Õ½Õ¿ Ö†Õ«Õ¶Õ¡Õ¶Õ½Õ¡Õ¾Õ¸Ö€Õ´Õ¡Õ¶", xaxis=dict(range=["auto", "auto"]))
    rc1.plotly_chart(S(f2, h=300), width="stretch")
    
    df7_3 = load_data('p7_sectors.csv')
    sec_lab = df7_3['sector'].tolist()[::-1]
    sec_val = df7_3['val'].tolist()[::-1]
    f3 = go.Figure(go.Bar(x=sec_val, y=sec_lab, orientation="h", marker_color=["#0070c0", "#0070c0", "#0070c0", "#cc0000"], text=sec_val, textposition="outside"))
    f3.update_layout(title="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶ Õ¨Õ½Õ¿ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ«", xaxis=dict(range=["auto", "auto"]))
    rc1.plotly_chart(S(f3, h=300), width="stretch")

elif page == "Ô±Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ·Õ¸Ö‚Õ¯Õ¡Õ¶ Ö‡ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¡Õ¯Õ¡Õ¶ Õ©Õ¸Ö‚ÕµÕ¬Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ ÔµÖ€Ö‡Õ¡Õ¶Õ¸Ö‚Õ´":
    st.title(page)
    st.info("2025Õ©. Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ½Õ¥ÕºÕ¿Õ¥Õ´Õ¢Õ¥Ö€Õ«Õ¶ Õ€Õ€-Õ¸Ö‚Õ´ Õ¢Õ¶Õ¡Õ¯Õ¥Õ¬Õ« Õ¡Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ£Õ¶Õ¥Ö€Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 3.8%-Õ¸Õ¾` ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ°Õ¡Õ¿Õ¯Õ¡ÕºÕ¥Õ½ Õ€Õ€-Õ¸Ö‚Õ´ Õ¢Õ¶Õ¡Õ¯Õ¥Õ¬Õ« Õ¿Õ¶Õ¥Ö€Õ« Õ£Õ¶Õ¥Ö€Õ«, Õ«Õ¶Õ¹ÕºÕ¥Õ½ Õ¶Õ¡Ö‡ ÔµÖ€Ö‡Õ¡Õ¶Õ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¢Õ¶Õ¡Õ¯Õ¡Ö€Õ¡Õ¶Õ¶Õ¥Ö€Õ« Õ£Õ¶Õ¥Ö€Õ« Õ¡Õ³Õ¸Õ¾:\n\nÕ€Õ¡Õ¶Ö€Õ¡ÕºÕ¥Õ¿Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¿Õ¡Ö€Õ¡Õ®Ö„Õ¸Ö‚Õ´ Õ¢Õ¶Õ¡Õ¯Õ¥Õ¬Õ« Õ¡Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ¡Õ¼Õ¸Ö‚Õ¾Õ¡Õ³Õ¡Õ¼Ö„Õ« Õ£Õ¸Ö€Õ®Õ¡Ö€Ö„Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 23.4%-Õ¸Õ¾Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ«Õ¶Õ¹ÕºÕ¥Õ½ ÔµÖ€Ö‡Õ¡Õ¶Õ¸Ö‚Õ´, Õ¡ÕµÕ¶ÕºÕ¥Õ½ Õ§Õ¬ ÔµÖ€Ö‡Õ¡Õ¶Õ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¡Õ¼Õ¸Ö‚Õ¾Õ¡Õ³Õ¡Õ¼Ö„Õ« Õ£Õ¸Ö€Õ®Õ¡Ö€Ö„Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¡Õ³Õ¸Õ¾:\n\n2025Õ©. ÔµÖ€Ö‡Õ¡Õ¶Õ¸Ö‚Õ´ Õ¿Ö€Õ¾Õ¡Õ® Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¡Õ¯Õ¡Õ¶ Õ©Õ¸Ö‚ÕµÕ¬Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ Õ¦Õ£Õ¡Õ¬Õ«Õ¸Ö€Õ¥Õ¶ Õ¦Õ«Õ»Õ¸Ö‚Õ´ Õ§ Õ¾Õ¥Ö€Õ»Õ«Õ¶ 3 Õ¿Õ¡Ö€Õ«Õ¶Õ¥Ö€Õ« Õ¨Õ¶Õ©Õ¡ÖÖ„Õ¸Ö‚Õ´ Õ¿Ö€Õ¾Õ¡Õ® Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¡Õ¯Õ¡Õ¶ Õ©Õ¸Ö‚ÕµÕ¬Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Ö„Õ¡Õ¶Õ¡Õ¯Õ«Õ¶, Õ«Õ¶Õ¹ Õ°Õ¥Õ¿Õ¡Õ£Õ¡ Õ¼Õ«Õ½Õ¯Õ¥Ö€ Õ§ Õ½Õ¿Õ¥Õ²Õ®Õ¸Ö‚Õ´ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ³Õ« Õ¯Õ¡ÕµÕ¸Ö‚Õ¶Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¿Õ¥Õ½Õ¡Õ¶Õ¯ÕµÕ¸Ö‚Õ¶Õ«Ö:")
    c1, c2, c3 = st.columns(3)
    
    # Chart 1: Price Index (approximate visual)
    df8_1 = load_data('p8_real_estate.csv')
    q_lbl = [translate_p(x) for x in df8_1['quarter'].tolist()]
    
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=q_lbl, y=df8_1['trend_blue'], name="Õ€Õ€-Õ¸Ö‚Õ´", line=dict(color="#3182bd", width=3)))
    f1.add_trace(go.Scatter(x=q_lbl, y=df8_1['trend_red'], name="ÔµÖ€Ö‡Õ¡Õ¶Õ«Ö Õ¤Õ¸Ö‚Ö€Õ½", line=dict(color="#de2d26", width=3)))
    f1.update_layout(title="Ô²Õ¶Õ¡Õ¯Õ¥Õ¬Õ« Õ¡Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ« Õ£Õ¶Õ¥Ö€Õ¨<br>(2018Õ©.=100)", showlegend=False, xaxis=dict(tickangle=-90, tickfont=dict(size=9)))
    c1.plotly_chart(S(f1, h=500), width="stretch")
    
    # Chart 2: Transactions
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=q_lbl, y=df8_1['t_blue'], name="Õ€Õ€-Õ¸Ö‚Õ´", line=dict(color="#3182bd", width=3)))
    f2.add_trace(go.Scatter(x=q_lbl, y=df8_1['t_red'], name="ÔµÖ€Ö‡Õ¡Õ¶Õ«Ö Õ¤Õ¸Ö‚Ö€Õ½", line=dict(color="#de2d26", width=3)))
    f2.update_layout(title="Ô²Õ¶Õ¡Õ¯Õ¥Õ¬Õ« Õ¡Õ¶Õ·Õ¡Ö€Õª Õ£Õ¸Ö‚ÕµÖ„Õ«<br>Õ¡Õ¼Õ¸Ö‚Õ¾Õ¡Õ³Õ¡Õ¼Ö„Õ« Õ£Õ¸Ö€Õ®Õ¡Ö€Ö„Õ¶Õ¥Ö€Õ¨, Õ°Õ¡Õ¿", showlegend=False, xaxis=dict(tickangle=-90, tickfont=dict(size=9)))
    c2.plotly_chart(S(f2, h=500), width="stretch")
    
    # Chart 3: Construction Permits
    df8_2 = load_data('p8_permits.csv')
    qq = [translate_p(x) for x in df8_2['quarter'].tolist()]
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2022'], name="2022Õ©.", line=dict(color="#3182bd", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2023'], name="2023Õ©.", line=dict(color="#74c476", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2024'], name="2024Õ©.", line=dict(color="#fd8d3c", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2025'], name="2025Õ©.", line=dict(color="#cc0000", width=4), mode="lines+markers+text", text=df8_2['p2025'].astype(str), textposition="top center"))
    f3.update_layout(title="Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¡Õ¯Õ¡Õ¶ Õ©Õ¸Ö‚ÕµÕ¬Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ«<br>Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ ÔµÖ€Ö‡Õ¡Õ¶Õ¸Ö‚Õ´, Õ°Õ¡Õ¿", legend=dict(orientation="v", y=0, x=0.8, font=dict(size=10)))
    c3.plotly_chart(S(f3, h=500), width="stretch")

elif page == "Ô¶Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. Õ¥Ö€Ö€Õ¸Ö€Õ¤ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¸Ö‚Õ´ Õ¦Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§ 0.1%-Õ¸Õ¾ (Õ·Õ¸Ö‚Ö€Õ» 1400 Õ´Õ¡Ö€Õ¤Õ¸Õ¾)Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ¸Õ¹ Õ¾Õ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€Õ«* Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¶Õ¾Õ¡Õ¦Õ´Õ¡Õ´Õ¢: Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¸Ö‚Õ´ Õ¦Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯Õ¨ (52.1%) Õ¶Õ¡Õ­Õ¸Ö€Õ¤ Õ¿Õ¡Ö€Õ¾Õ¡ Õ¶Õ¸Ö‚ÕµÕ¶ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ« Õ¶Õ¯Õ¡Õ¿Õ´Õ¡Õ´Õ¢ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ´Õ« Õ¯Õ¸Õ²Õ´Õ«Ö Õ¦Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¶Õ¾Õ¡Õ¦Õ´Õ¡Õ´Õ¢, Õ´ÕµÕ¸Ö‚Õ½ Õ¯Õ¸Õ²Õ´Õ«Ö Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¼Õ¥Õ½Õ¸Ö‚Ö€Õ½Õ¶Õ¥Ö€Õ« Õ¡Õ¾Õ¥Õ¬Õ¡ÖÕ´Õ¡Õ´Õ¢:")
    c1, c2 = st.columns(2)
    df9_1 = load_data('p9_employment.csv')
    q = [translate_p(x) for x in df9_1['period'].tolist()]
    
    # Appending None as a gap to effectively disconnect the `ÕÕ¡Ö€Õ¥Õ¯Õ¡Õ¶` values from the monthly line curves.
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=["1,159.4", "1,190.7", "1,210.0", "", "", ""], textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f1.update_layout(title="Ô¶Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¶Õ¥Ö€ (Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤)")
    c1.plotly_chart(S(f1), width="stretch")
    
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=["50.1", "51.2", "52.1", "", "", ""], textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f2.update_layout(title="Ô¶Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯, %")
    c2.plotly_chart(S(f2), width="stretch")
    
    df9_2 = load_data('p9_employment_structure.csv')
    f3 = go.Figure(data=[
        go.Bar(name="ÕˆÕ¹ Õ¾Õ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€", x=df9_2['no_wage'], y=df9_2['type'], orientation="h", marker_color="#ffa657", text=df9_2['no_wage'], textposition="inside"),
        go.Bar(name="ÕŽÕ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€", x=df9_2['wage'], y=df9_2['type'], orientation="h", marker_color="#92d050", text=df9_2['wage'], textposition="inside")
    ])
    f3.update_layout(barmode="stack", title="Ô¶Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¶Õ¥Ö€Õ« Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¨ (III Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯, Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤)")
    st.plotly_chart(S(f3, h=300), width="stretch")

elif page == "Ô³Õ¸Ö€Õ®Õ¡Õ¦Ö€Õ¯Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.info("2025Õ©. Õ¥Ö€Ö€Õ¸Ö€Õ¤ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¸Ö‚Õ´ Õ£Õ¸Ö€Õ®Õ¡Õ¦Õ¸Ö‚Ö€Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ Õ¯Ö€Õ³Õ¡Õ¿Õ¾Õ¥Õ¬ Õ§ 13.1%-Õ¸Õ¾ (Õ·Õ¸Ö‚Ö€Õ» 24.3 Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤Õ¸Õ¾), Õ½Õ¡Õ¯Õ¡ÕµÕ¶ Õ¾Õ¥Ö€Õ»Õ«Õ¶Õ¶Õ¥Ö€Õ½ Õ¹Õ¥Õ¶ Õ°Õ¡Õ´Õ¡Õ¬Ö€Õ¥Õ¬ Õ¦Õ¢Õ¡Õ²Õ¾Õ¡Õ®Õ¶Õ¥Ö€Õ« Õ·Õ¡Ö€Ö„Õ¨, Õ¡ÕµÕ¬ Õ¶Õ¥Ö€Õ£Ö€Õ¡Õ¾Õ¾Õ¥Õ¬ Õ¥Õ¶ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¯Õ¡Õ¦Õ´Õ¸Ö‚Õ´, Õ«Õ¶Õ¹Õ« Õ°Õ¥Õ¿Ö‡Õ¡Õ¶Ö„Õ¸Õ¾ Õ¯Ö€Õ³Õ¡Õ¿Õ¾Õ¥Õ¬ Õ§ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ« Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯Õ¨: Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¸Ö‚Õ´ Õ£Õ¸Ö€Õ®Õ¡Õ¦Ö€Õ¯Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯Õ¨ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§ 1.5 Õ¿Õ¸Õ¯Õ¸Õ½Õ¡ÕµÕ«Õ¶ Õ¯Õ¥Õ¿Õ¸Õ¾Õ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 11.8%:")
    c1, c2 = st.columns(2)
    df10_1 = load_data('p10_unemployment.csv')
    q = [translate_p(x) for x in df10_1['period'].tolist()]
    
    f1 = go.Figure()
    text25_10 = [str(v) if not pd.isna(v) else "" for v in df10_1['unemp25']]
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text25_10, textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f1.update_layout(title="Ô³Õ¸Ö€Õ®Õ¡Õ¦Õ¸Ö‚Ö€Õ¯Õ¶Õ¥Ö€ (Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤)")
    c1.plotly_chart(S(f1), width="stretch")
    
    f2 = go.Figure()
    textlvl25_10 = [str(v) if not pd.isna(v) else "" for v in df10_1['lvl25']]
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=textlvl25_10, textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f2.update_layout(title="Ô³Õ¸Ö€Õ®Õ¡Õ¦Ö€Õ¯Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯, %")
    c2.plotly_chart(S(f2), width="stretch")
    
    c3, c4 = st.columns(2)
    df10_2 = load_data('p10_changes.csv')
    f3 = go.Figure(go.Bar(y=df10_2['category'][::-1], x=df10_2['val'][::-1], orientation="h", marker_color=["#1f6feb", "#1f6feb", "#1f6feb", "#c00000"], text=df10_2['val'][::-1], textposition="outside"))
    f3.update_layout(title="Ô²Õ¡ÖÕ¡Ö€Õ±Õ¡Õ¯ ÖƒÕ¸ÖƒÕ¸Õ­Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€ (III Õ¥Õ¼., Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤)", xaxis=dict(range=[-40, 50]))
    c3.plotly_chart(S(f3, h=400), width="stretch")
    
    df10_3 = load_data('p10_registered.csv')
    m = [translate_p(x) for x in df10_3['month'].tolist()]
    r25 = df10_3['r25'].tolist()
    r24 = df10_3['r24'].tolist()
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=m, y=r24, name="2024Õ©.", line=dict(color="#3182bd", width=2, dash="dash"), mode="lines+text", text=[str(v) if not pd.isna(v) else "" for v in r24], textposition="top center"))
    f4.add_trace(go.Scatter(x=m, y=r25, name="2025Õ©.", line=dict(color="#ffa657", width=3), mode="lines+markers+text", text=[str(v) if not pd.isna(v) else "" for v in r25], textposition="bottom center"))
    f4.add_hline(y=0, line_color="#ff7b72", line_dash="dot")
    f4.update_layout(title="ÕŠÕ¡Õ·Õ¿Õ¸Õ¶Õ¡ÕºÕ¥Õ½ Õ£Ö€Õ¡Õ¶ÖÕ¾Õ¡Õ® Õ£Õ¸Ö€Õ®Õ¡Õ¦Õ¸Ö‚Ö€Õ¯Õ¶Õ¥Ö€ (Ô±Õ³, %)", legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5))
    c4.plotly_chart(S(f4, h=400), width="stretch")

elif page == "Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¼Õ¥Õ½Õ¸Ö‚Ö€Õ½Õ¶Õ¥Ö€":
    st.title(page)
    st.info("2025Õ©. Õ¥Ö€Ö€Õ¸Ö€Õ¤ Õ¥Õ¼Õ¡Õ´Õ½ÕµÕ¡Õ¯Õ¸Ö‚Õ´ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¼Õ¥Õ½Õ¸Ö‚Ö€Õ½Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 0.1%-Õ¸Õ¾ (Õ·Õ¸Ö‚Ö€Õ» 2.9 Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤Õ¸Õ¾), Õ¸Ö€Õ¸Õ¶Ö„ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ°Õ¡Õ´Õ¡Õ¬Ö€Õ¥Õ¬ Õ¥Õ¶ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¯Õ¡Õ¦Õ´Õ¨: Õ„Õ«Õ¡ÕªÕ¡Õ´Õ¡Õ¶Õ¡Õ¯ Õ¿Õ¥Õ²Õ« Õ§ Õ¸Ö‚Õ¶Õ¥ÖÕ¥Õ¬ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ« Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯Õ« Õ¶Õ¾Õ¡Õ¦Õ¸Ö‚Õ´Õ 1.8% (25.8 Õ°Õ¡Õ¦. Õ´Õ¡Ö€Õ¤): Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Ö„Õ¸Ö‚Õ´ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ« Õ´Õ¡Õ½Õ¶Õ¡Õ¯ÖÕ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯Õ¨ Ö‡Õ½ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§Õ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 59.0%, Õ«Õ½Õ¯ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ´Õ¡Õ¯Õ¡Ö€Õ¤Õ¡Õ¯Õ¶Õ Õ¡Õ³Õ¥Õ¬Õ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 41.0%:")
    c1, c2, c3 = st.columns(3)
    df11 = load_data('p11_labor_resources.csv')
    q = [translate_p(x) for x in df11['period'].tolist()]
    
    f1 = go.Figure()
    # Explicitly casting float to int-like string for 2314 to match the original layout exactly, ignoring NaN
    text_res25 = ["2314.5", "2327.2", "2324.2", "", "", ""]
    pos_res25 = ["bottom right", "top center", "bottom right", "top center", "top center", "top center"]
    text_res24 = ["", "", "", "", "", "2295.9"]
    text_res23 = ["", "", "", "", "", "2223.2"]
    f1.add_trace(go.Scatter(x=q, y=df11['res25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_res25, textposition=pos_res25))
    f1.add_trace(go.Scatter(x=q, y=df11['res24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_res24, textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df11['res23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_res23, textposition="top center"))
    f1.update_layout(title="Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ¡ÕµÕ«Õ¶ Õ¼Õ¥Õ½Õ¸Ö‚Ö€Õ½Õ¶Õ¥Ö€ (Õ°Õ¡Õ¦.)")
    c1.plotly_chart(S(f1), width="stretch")
    
    f2 = go.Figure()
    text_sup25 = ["1347.6", "1357.1", "1371.3", "", "", ""]
    pos_sup25 = ["bottom right", "bottom right", "top center", "top center", "top center", "top center"]
    text_sup24 = ["", "", "", "", "", "1357.3"]
    text_sup23 = ["", "", "", "", "", "1341.2"]
    f2.add_trace(go.Scatter(x=q, y=df11['sup25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_sup25, textposition=pos_sup25))
    f2.add_trace(go.Scatter(x=q, y=df11['sup24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_sup24, textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df11['sup23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_sup23, textposition="top center"))
    f2.update_layout(title="Ô±Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ« Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯ (Õ°Õ¡Õ¦.)")
    c2.plotly_chart(S(f2), width="stretch")
    
    f3 = go.Figure()
    text_out25 = ["967", "970", "953", "", "", ""]
    pos_out25 = ["bottom right", "top center", "bottom right", "top center", "top center", "top center"]
    text_out24 = ["", "", "924", "", "", "939"]
    text_out23 = ["", "", "", "", "", "882"]
    f3.add_trace(go.Scatter(x=q, y=df11['out25'], name="2025Õ©.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_out25, textposition=pos_out25))
    f3.add_trace(go.Scatter(x=q, y=df11['out24'], name="2024Õ©.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_out24, textposition="top center"))
    f3.add_trace(go.Scatter(x=q, y=df11['out23'], name="2023Õ©.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_out23, textposition="top center"))
    f3.update_layout(title="Ô±Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ«Ö Õ¤Õ¸Ö‚Ö€Õ½ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (Õ°Õ¡Õ¦.)")
    c3.plotly_chart(S(f3), width="stretch")

elif page == "Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ± Ö‡ Õ¾Õ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€":
    st.title(page)
    st.info("2025Õ©. Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€Õ«Õ¶ ÕºÕ¡Õ·Õ¿Õ¸Õ¶Õ¡ÕºÕ¥Õ½ Õ£Ö€Õ¡Õ¶ÖÕ¾Õ¡Õ® Õ£Õ¸Ö€Õ®Õ¡Õ¦Õ¸Ö‚Ö€Õ¯Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§ 13.7%-Õ¸Õ¾ (Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 36,378 Õ´Õ¡Ö€Õ¤), Õ«Õ½Õ¯ Õ¾Õ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ¨ Õ§ Õ¡Õ³Õ¥Õ¬ 4.6%-Õ¸Õ¾  (Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 795,212 Õ´Õ¡Ö€Õ¤):\n\nÕ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€Õ«Õ¶ Õ´Õ«Õ»Õ«Õ¶ Õ¡Õ´Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¶Õ¾Õ¡Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ¶ Õ¡Õ³Õ¥Õ¬ Õ§ 5.6%-Õ¸Õ¾Õ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬Õ¸Õ¾ 303,140 Õ¤Ö€Õ¡Õ´ (ÕºÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ°Õ¡Õ¿Õ¾Õ¡Õ®Õ¸Ö‚Õ´Õ 239,369 Õ¤Ö€Õ¡Õ´, Õ¸Õ¹ ÕºÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´Õ 327,604 Õ¤Ö€Õ¡Õ´): Ô±Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ« Õ¡Õ³Õ¨ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ¥Õ²Õ¥Õ¬ Õ¡Õ¼Ö‡Õ¿Ö€Õ«, Õ¯Ö€Õ©Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Ö‡ Õ´Õ·Õ¡Õ¯Õ¸Õ² Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ¸Ö‚Õ´ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ¥Ö€Õ« Õ¡Õ³Õ¸Õ¾Ö‰ Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€Õ«Õ¶ 3.3% Õ£Õ¶Õ¡Õ³Õ« ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¶Õ¥Ö€Õ¸Ö‚Õ´ Õ´Õ«Õ»Õ«Õ¶ Õ¡Õ´Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ« Õ«Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³Õ¨ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ§ 2.2%:")
    
    df12 = load_data('p12_wages.csv')
    sec = df12['sector'].tolist()
    wg = df12['wg'].tolist()
    em = df12['em'].tolist()
    
    # Compute dynamic colors: light blue for positives, red for negatives, explicitly checking wg structure.
    wg_colors = ["#cc0000" if v < 0 else "#c9daf8" for v in wg]
    
    f = go.Figure()
    f.add_trace(go.Bar(x=sec, y=wg, name="Õ„Õ«Õ»Õ«Õ¶ Õ¡Õ´Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ« Õ¡Õ³, %", marker_color=wg_colors, text=[str(v) for v in wg], textposition="inside", insidetextanchor="start", textangle=0, textfont=dict(size=10)))
    f.add_trace(go.Scatter(x=sec, y=em, name="ÕŽÕ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€Õ« Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¡Õ³, %", line=dict(color="#e6550d", width=2), marker=dict(size=10, color="#e6550d"), mode="lines+markers+text", text=[str(v) for v in em], textposition="top center", textfont=dict(color="#e6550d", size=12)))
    
    # Adding the black bounding box explicitly to 'Ô¸Õ¶Õ¤Õ¡Õ´Õ¥Õ¶Õ¨' using shapes might be complex in standard layout, 
    # so we will ensure its visibility structurally.
    f.add_shape(type="rect", x0=-0.4, x1=0.4, y0=-2, y1=10, line=dict(color="#888888", width=2), fillcolor="rgba(0,0,0,0)", layer="below")
    
    f.update_layout(title="ÕŽÕ¡Ö€Õ±Õ¸Ö‚ Õ¡Õ·Õ­Õ¡Õ¿Õ¸Õ²Õ¶Õ¥Ö€Õ« Öƒ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¾Õ¡Ö€Õ±Õ¥Ö€Õ« Õ¡Õ³Õ¥Ö€Õ¨ Õ¨Õ½Õ¿ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ«, %<br>(Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ¤Õ¥Õ¯Õ¿Õ¥Õ´Õ¢Õ¥Ö€)", barmode="group", xaxis_tickangle=-45, legend=dict(orientation="h", yanchor="bottom", y=0.85, xanchor="left", x=0.01))
    
    # Adjust yaxis to make room for bottom labels easily without overlap
    f.update_layout(yaxis=dict(range=[-7, 18]))
    st.plotly_chart(S(f, h=650), width="stretch")

elif page == "Ô±Ö€Õ¿Õ¡Ö„Õ«Õ¶ Õ¡Õ¼Ö‡Õ¿Ö€Õ¡Õ·Ö€Õ»Õ¡Õ¶Õ¡Õ¼Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶":
    st.title(page)
    st.markdown("---")
    st.info("Õ€Õ€ Õ¡Ö€Õ¿Õ¡Ö„Õ«Õ¶ Õ¡Õ¼Ö‡Õ¿Ö€Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶ 2024-2025Õ©Õ©-Õ«Õ¶ Õ¢Õ¶Õ¸Ö‚Õ©Õ¡Õ£Ö€Õ¾Õ¸Ö‚Õ´ Õ§ Õ¡Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ´Õ¡Õ¶ Ö‡ Õ¶Õ¥Ö€Õ´Õ¸Ö‚Õ®Õ´Õ¡Õ¶ Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¡ÕµÕ«Õ¶ ÖƒÕ¸ÖƒÕ¸Õ­Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ¸Õ¾ Ö‡ Õ£Õ¸Ö€Õ®Õ¨Õ¶Õ¯Õ¥Ö€ Õ¥Ö€Õ¯Ö€Õ¶Õ¥Ö€Õ« Õ¤Õ«Õ¾Õ¥Ö€Õ½Õ«Ö†Õ«Õ¯Õ¡ÖÕ´Õ¡Õ¶ Õ´Õ«Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¸Õ¾Ö‰")
    
    df_trade = load_data('adv_trade.csv')
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Ô±Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ´Õ¡Õ¶ Ö‡ Õ¶Õ¥Ö€Õ´Õ¸Ö‚Õ®Õ´Õ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¶Õ¥Ö€")
        # Comparing 2024 Total with 2025 Jan
        f = go.Figure()
        f.add_trace(go.Bar(name='2024 (Ô±Õ´ÖƒÕ¸Öƒ, Õ´Õ¬Õ¶ USD)', x=df_trade['Ô¿Õ¡Õ¿Õ¥Õ£Õ¸Ö€Õ«Õ¡'], y=df_trade['2024_Ô¸Õ¶Õ¤Õ¡Õ´Õ¥Õ¶Õ¨'], marker_color="#1f6feb"))
        f.add_trace(go.Bar(name='2025 (Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€, Õ´Õ¬Õ¶ USD)', x=df_trade['Ô¿Õ¡Õ¿Õ¥Õ£Õ¸Ö€Õ«Õ¡'], y=df_trade['2025_Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€'], marker_color="#ff7b72"))
        f.update_layout(barmode='group')
        st.plotly_chart(S(f, h=450), width="stretch")
    
    with c2:
        st.subheader("Õ€Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶ Õ·Õ¥Õ·Õ¿Õ¡Õ¤Ö€Õ¸Ö‚Õ´Õ¶Õ¥Ö€")
        st.write(f"""
        - **2025Õ©.** Õ¡Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ¸Ö‚Õ´Õ¨ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ§ **{df_trade[df_trade['Ô¿Õ¡Õ¿Õ¥Õ£Õ¸Ö€Õ«Õ¡']=='Ô±Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ¸Ö‚Õ´']['2025_Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€'].values[0]} Õ´Õ¬Õ¶ USD**:
        - **Ô±Õ¼Ö‡Õ¿Ö€Õ¡ÕµÕ«Õ¶ Õ°Õ¡Õ·Õ¾Õ¥Õ¯Õ·Õ«Õ¼Õ¨** Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¸Ö‚Õ´ Õ§ Õ´Õ¶Õ¡Õ¬ Õ¢Õ¡ÖÕ¡Õ½Õ¡Õ¯Õ¡Õ¶, Õ½Õ¡Õ¯Õ¡ÕµÕ¶ Õ¡Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ´Õ¡Õ¶ Õ¡Õ³Õ« Õ¿Õ¥Õ´ÕºÕ¥Ö€Õ¨ Õ¸Ö€Õ¸Õ·Õ¡Õ¯Õ« ÕªÕ¡Õ´Õ¡Õ¶Õ¡Õ¯Õ¡Õ°Õ¡Õ¿Õ¾Õ¡Õ®Õ¶Õ¥Ö€Õ¸Ö‚Õ´ Õ£Õ¥Ö€Õ¡Õ¦Õ¡Õ¶ÖÕ¸Ö‚Õ´ Õ¥Õ¶ Õ¶Õ¥Ö€Õ´Õ¸Ö‚Õ®Õ´Õ¡Õ¶Õ¨Ö‰
        - **ÕŒÕ¸Ö‚Õ½Õ¡Õ½Õ¿Õ¡Õ¶Õ¨, Ô±Õ„Ô·-Õ¶ Ö‡ Õ‰Õ«Õ¶Õ¡Õ½Õ¿Õ¡Õ¶Õ¨** Õ´Õ¶Õ¸Ö‚Õ´ Õ¥Õ¶ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶ Õ£Õ¸Ö€Õ®Õ¨Õ¶Õ¯Õ¥Ö€Õ¶Õ¥Ö€Õ¨:
        - Ô±Õ¾Õ¥Õ¬Õ¡ÖÕ¥Õ¬ Õ§ Õ©Õ¡Õ¶Õ¯Õ¡Ö€ÕªÕ¥Ö„ Ö„Õ¡Ö€Õ¥Ö€Õ« Ö‡ Õ´Õ¥Õ¿Õ¡Õ²Õ¶Õ¥Ö€Õ« Õ´Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶Õ¨ Õ¡Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ´Õ¡Õ¶ Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¸Ö‚Õ´Ö‰
        """)

elif page == "Ô´Ö€Õ¡Õ´Õ¡Õ¾Õ¡Ö€Õ¯Õ¡ÕµÕ«Õ¶ Õ¯Õ¡ÕµÕ¸Ö‚Õ¶Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Ô³Õ¶Õ¡Õ³":
    st.title(page)
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Õ“Õ¸Õ­Õ¡Ö€ÕªÕ¥Ö„Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡ (AMD per USD/RUB)")
        df_fx = load_data('adv_fx.csv')
        df_fx['Ô±Õ´Õ½Õ¡Õ©Õ«Õ¾'] = pd.to_datetime(df_fx['Ô±Õ´Õ½Õ¡Õ©Õ«Õ¾'])
        fx_lbl = [f"{translate_p(d.month)} {d.year}Õ©." if d.day==1 else f"{d.day} {translate_p(d.month)} {d.year}Õ©." for d in df_fx['Ô±Õ´Õ½Õ¡Õ©Õ«Õ¾']]
        
        f = go.Figure()
        f.add_trace(go.Scatter(x=fx_lbl, y=df_fx['USD'], name="USD/AMD", line=dict(color="#58a6ff", width=4)))
        f.add_trace(go.Scatter(x=fx_lbl, y=df_fx['RUB'], name="RUB/AMD", yaxis="y2", line=dict(color="#ff9f43", width=4)))
        f.update_layout(yaxis=dict(title="USD/AMD"), yaxis2=dict(title="RUB/AMD", overlaying="y", side="right"))
        st.plotly_chart(S(f, h=450), width="stretch")
        st.caption("Ô±Õ²Õ¢ÕµÕ¸Ö‚Ö€Õ Õ€Õ€ Ô¿Õ¥Õ¶Õ¿Ö€Õ¸Õ¶Õ¡Õ¯Õ¡Õ¶ Ô²Õ¡Õ¶Õ¯")

    with c2:
        st.subheader("ðŸ“‰ ÕÕºÕ¡Õ¼Õ¸Õ²Õ¡Õ¯Õ¡Õ¶ Ô³Õ¶Õ¥Ö€Õ« Õ€Õ¡Õ´Õ¡Õ©Õ«Õ¾ (Ô³Õ¶Õ¡Õ³, %)")
        df_cpi = load_data('adv_cpi.csv')
        f_cpi = go.Figure()
        f_cpi.add_trace(go.Scatter(x=df_cpi['Ô±Õ´Õ«Õ½'], y=df_cpi['2025'], name="2025 (Õ¶Õ¡Õ­. Õ¡Õ´Õ½Õ¾Õ¡ Õ¶Õ¯.)", fill='tozeroy', line=dict(color="#1f6feb")))
        f_cpi.add_trace(go.Scatter(x=df_cpi['Ô±Õ´Õ«Õ½'], y=df_cpi['2024'], name="2024", line=dict(color="#adbac7", dash='dot')))
        f_cpi.update_layout(yaxis_title="Ô±Õ´Õ½Õ¡Õ¯Õ¡Õ¶ ÖƒÕ¸ÖƒÕ¸Õ­Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶, %")
        st.plotly_chart(S(f_cpi, h=450), width="stretch")

elif page == "Ô·Õ¶Õ¥Ö€Õ£Õ¥Õ¿Õ«Õ¯Õ¡ Ö‡ Õ„Õ¡Õ¯Ö€Õ¸-Õ¡Õ¼Õ¡Õ»Õ¡Õ¶ÖÕ«Õ¯ ÖÕ¸Ö‚ÖÕ«Õ¹":
    st.title(page)
    st.markdown("---")
    st.info("Ô·Õ¬Õ¥Õ¯Õ¿Ö€Õ¡Õ§Õ¶Õ¥Ö€Õ£Õ«Õ¡ÕµÕ« Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¨ Õ°Õ¡Õ¶Õ¤Õ«Õ½Õ¡Õ¶Õ¸Ö‚Õ´ Õ§ Õ€Õ†Ô±-Õ« Ö‡ Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Õ³Õ« Õ°Õ¸Ö‚Õ½Õ¡Õ¬Õ« Â«proxyÂ» ÖÕ¸Ö‚ÖÕ¡Õ¶Õ«Õ·Ö‰")
    
    df_e = load_data('adv_electricity.csv')
    f = go.Figure()
    f.add_trace(go.Bar(x=df_e['Ô±Õ´Õ«Õ½'], y=df_e['2024'], name="2024 (Õ´Õ¬Õ¶ Õ¤Ö€Õ¡Õ´)", marker_color="#adbac7"))
    f.add_trace(go.Bar(x=df_e['Ô±Õ´Õ«Õ½'], y=df_e['2025'], name="2025 (Õ´Õ¬Õ¶ Õ¤Ö€Õ¡Õ´)", marker_color="#1f6feb"))
    f.update_layout(title="Ô·Õ¬Õ¥Õ¯Õ¿Ö€Õ¡Õ§Õ¶Õ¥Ö€Õ£Õ«Õ¡ÕµÕ«, Õ£Õ¡Õ¦Õ«, Õ£Õ¸Õ¬Õ¸Ö€Õ·Õ¸Ö‚ Ö‡ Õ¬Õ¡Õ¾Õ¸Ö€Õ¡Õ¯ Ö…Õ¤Õ« Õ´Õ¡Õ¿Õ¡Õ¯Õ¡Ö€Õ¡Ö€Õ´Õ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¶Õ¥Ö€", barmode='group')
    st.plotly_chart(S(f, h=550), width="stretch")
    
    st.write("""
    **ÕŽÕ¥Ö€Õ¬Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶**: 2025Õ©. Õ°Õ¸Ö‚Õ¬Õ«Õ½-Ö…Õ£Õ¸Õ½Õ¿Õ¸Õ½ Õ¡Õ´Õ«Õ½Õ¶Õ¥Ö€Õ«Õ¶ Õ¶Õ¯Õ¡Õ¿Õ¾Õ¥Õ¬ Õ§ Õ§Õ¶Õ¥Ö€Õ£Õ«Õ¡ÕµÕ« Õ½ÕºÕ¡Õ¼Õ´Õ¡Õ¶ Õ¯Õ¿Ö€Õ¸Ö‚Õ¯ Õ¡Õ³Õ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ©Õ¥Õ› Õ¯Õ¬Õ«Õ´Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¶Õ¥Ö€Õ¸Õ¾, Õ©Õ¥Õ› Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ°Õ¦Õ¸Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ¡Õ¯Õ¿Õ«Õ¾Õ¡ÖÕ´Õ¡Õ´Õ¢Ö‰
    """)

elif page == "Õ€Õ¡Ö€Õ¯Õ¡Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ¿Õ¡ÕµÕ«Õ¶ ÖÕ¸Ö‚ÖÕ¡Õ¶Õ«Õ·Õ¶Õ¥Ö€":
    st.title(page)
    st.markdown("---")
    
    df_f = load_data('adv_fiscal.csv')
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("ÕŠÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ« Õ¡Õ³ (Õ´Õ¬Õ¶ Õ¤Ö€Õ¡Õ´)")
        f = go.Figure()
        f.add_trace(go.Bar(name='ÔµÕ¯Õ¡Õ´Õ¸Ö‚Õ¿Õ¶Õ¥Ö€', x=df_f['Year'], y=df_f['Revenue'], marker_color='#238636'))
        f.add_trace(go.Bar(name='Ô¾Õ¡Õ­Õ½Õ¥Ö€', x=df_f['Year'], y=df_f['Expenditure'], marker_color='#da3633'))
        st.plotly_chart(S(f, h=500), width="stretch")
        
    with c2:
        st.subheader("Õ€Õ¡Ö€Õ¯Õ¡Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ¿Õ¡ÕµÕ«Õ¶ Õ¡Õ´ÖƒÕ¸ÖƒÕ¡Õ£Ö€")
        st.success("2025Õ©. ÕºÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ« Õ¥Õ¯Õ¡Õ´Õ¸Ö‚Õ¿Õ¶Õ¥Ö€Õ¨ Õ¶Õ¡Õ­Õ¶Õ¡Õ¯Õ¡Õ¶ Õ°Õ¡Õ·Õ¾Õ¡Ö€Õ¯Õ¶Õ¥Ö€Õ¸Õ¾ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ¥Õ¶ **2.88 Õ¿Ö€Õ«Õ¬Õ«Õ¸Õ¶ Õ¤Ö€Õ¡Õ´** (+11.9%):")
        st.write("""
        - ÕŠÕ¥Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢ÕµÕ¸Ö‚Õ»Õ¥Õ« **Õ¤Õ¥Ö†Õ«ÖÕ«Õ¿Õ¨** ÕºÕ¡Õ°ÕºÕ¡Õ¶Õ¾Õ¸Ö‚Õ´ Õ§ Õ¯Õ¡Õ¼Õ¡Õ¾Õ¡Ö€Õ¥Õ¬Õ«Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ½Õ¡Õ°Õ´Õ¡Õ¶Õ¶Õ¥Ö€Õ¸Ö‚Õ´:
        - Õ€Õ¡Ö€Õ¯Õ¡ÕµÕ«Õ¶ Õ¥Õ¯Õ¡Õ´Õ¸Ö‚Õ¿Õ¶Õ¥Ö€Õ« Õ¡Õ³Õ¨ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ Õ¡ÕºÕ¡Õ°Õ¸Õ¾Õ¾Õ¥Õ¬ Õ§ **Ô±Ô±Õ€**-Õ« Ö‡ **ÔµÕ¯Õ¡Õ´Õ¿Õ¡ÕµÕ«Õ¶ Õ°Õ¡Ö€Õ¯Õ«** Õ°Õ¡Õ·Õ¾Õ«Õ¶:
        - Ô¿Õ¡ÕºÕ«Õ¿Õ¡Õ¬ Õ®Õ¡Õ­Õ½Õ¥Ö€Õ« Õ´Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶Õ¨ Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¸Ö‚Õ´ Õ§ Õ¡Õ³Õ¥Õ¬Õ Õ¸Ö‚Õ²Õ²Õ¾Õ¥Õ¬Õ¸Õ¾ Õ¥Õ¶Õ©Õ¡Õ¯Õ¡Õ¼Õ¸Ö‚ÖÕ¾Õ¡Õ®Ö„Õ¶Õ¥Ö€Õ« Õ¦Õ¡Ö€Õ£Õ¡ÖÕ´Õ¡Õ¶Õ¨Ö‰
        """)

elif page == "Ô²Õ¡Õ¶Õ¯Õ¡ÕµÕ«Õ¶ Õ°Õ¡Õ´Õ¡Õ¯Õ¡Ö€Õ£ Ö‡ ÕŽÕ¡Ö€Õ¯Õ¡Õ¾Õ¸Ö€Õ¸Ö‚Õ´":
    st.title(page)
    st.markdown("---")
    st.subheader("ÕŽÕ¡Ö€Õ¯Õ¡Õ¾Õ¸Ö€Õ¸Ö‚Õ´Õ¶ Õ¨Õ½Õ¿ Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ« (ÕÖ€Õ¡Õ´Õ¡Õ¤Ö€Õ¾Õ¡Õ® Õ¾Õ¡Ö€Õ¯Õ¥Ö€Õ« Õ´Õ¶Õ¡ÖÕ¸Ö€Õ¤)")
    
    # Representative data based on CBA trends
    banking_data = pd.DataFrame({
        "ÕˆÕ¬Õ¸Ö€Õ¿": ["ÕÕºÕ¡Õ¼Õ¸Õ²Õ¡Õ¯Õ¡Õ¶", "Õ€Õ«ÖƒÕ¸Õ©Õ¥Ö„", "Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô±Õ¼Ö‡Õ¿Õ¸Ö‚Ö€", "Õ‡Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô³ÕµÕ¸Ö‚Õ²Õ¡Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶"],
        "Õ„Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶, %": [22.4, 20.8, 14.5, 13.2, 11.4, 6.7]
    })
    f = px.pie(banking_data, values="Õ„Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶, %", names="ÕˆÕ¬Õ¸Ö€Õ¿", color_discrete_sequence=px.colors.qualitative.Prism)
    f.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(S(f, h=600), width="stretch")
    
    st.info("Õ€Õ«ÖƒÕ¸Õ©Õ¥Ö„Õ¡ÕµÕ«Õ¶ Õ¾Õ¡Ö€Õ¯Õ¥Ö€Õ« Õ¯Õ¿Ö€Õ¸Ö‚Õ¯ Õ¡Õ³Õ¨ (Õ¿Õ¡Ö€Õ¥Õ¯Õ¡Õ¶ ~25-30%) Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¸Ö‚Õ´ Õ§ Õ´Õ¶Õ¡Õ¬ Õ·Õ«Õ¶Õ¡Ö€Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¸Õ¬Õ¸Ö€Õ¿Õ« Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶ Õ·Õ¡Ö€ÕªÕ«Õ¹ Õ¸Ö‚ÕªÕ¥Ö€Õ«Ö Õ´Õ¥Õ¯Õ¨Ö‰")

elif page == "Õ„Õ¡Ö€Õ¦Õ¡ÕµÕ«Õ¶ Õ¿Õ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ ÕºÕ¡Õ¿Õ¯Õ¥Ö€":
    st.title(page)
    st.markdown("---")
    st.info("Õ€Õ€ Õ¿Õ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¯Õ¿Õ«Õ¾Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ·Õ­Õ¡Ö€Õ°Õ¡Õ£Ö€Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¡Õ·Õ­Õ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¨Õ Õ¨Õ½Õ¿ Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„Õ« Õ®Õ¡Õ¾Õ¡Õ¬Õ« (2024Õ©.)Ö‰")
    
    df_m = load_data('adv_marz.csv')
    df_m = df_m.sort_values('Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„_1000_Õ¤Ö€Õ¡Õ´', ascending=False)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Ô±Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„Õ¶ Õ¨Õ½Õ¿ Õ´Õ¡Ö€Õ¦Õ¥Ö€Õ«")
        f = px.bar(df_m, x="Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„_1000_Õ¤Ö€Õ¡Õ´", y="Õ„Õ¡Ö€Õ¦", orientation='h', color="Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„_1000_Õ¤Ö€Õ¡Õ´", color_continuous_scale="Blues")
        f.update_layout(xaxis_title="Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„ (Õ°Õ¡Õ¦. Õ¤Ö€Õ¡Õ´)", yaxis_title="")
        st.plotly_chart(S(f, h=550), width="stretch")
        
    with c2:
        st.subheader("Ô¿Õ¥Õ¶Õ¿Ö€Õ¸Õ¶Õ¡ÖÕ¾Õ¡Õ®Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¾Õ¥Ö€Õ¬Õ¸Ö‚Õ®Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶")
        yerevan_share = (df_m[df_m['Õ„Õ¡Ö€Õ¦'] == 'ÔµÖ€Ö‡Õ¡Õ¶']['Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„_1000_Õ¤Ö€Õ¡Õ´'].values[0] / df_m['Ô±Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„_1000_Õ¤Ö€Õ¡Õ´'].sum()) * 100
        st.metric("ÔµÖ€Ö‡Õ¡Õ¶Õ« Õ´Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶Õ¨", f"{yerevan_share:.1f}%")
        st.write("""
        - **ÔµÖ€Ö‡Õ¡Õ¶Õ¨** Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¸Ö‚Õ´ Õ§ Õ´Õ¶Õ¡Õ¬ Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ°Õ¦Õ¸Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¯Õ¥Õ¶Õ¿Ö€Õ¸Õ¶Õ¨:
        - **ÕÕµÕ¸Ö‚Õ¶Õ«Ö„Õ¨** Õ¥Ö€Õ¯Ö€Õ¸Ö€Õ¤Õ¶ Õ§Õ Õ·Õ¶Õ¸Ö€Õ°Õ«Õ¾ Õ°Õ¡Õ¶Ö„Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ°Õ½Õ¯Õ¡ÕµÕ¡Õ¯Õ¡Õ¶ Õ®Õ¡Õ¾Õ¡Õ¬Õ¶Õ¥Ö€Õ«:
        - **2025Õ©. Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€-Õ“Õ¥Õ¿Ö€Õ¾Õ¡Ö€**: Õ†Õ¡Õ­Õ¶Õ¡Õ¯Õ¡Õ¶ Õ¿Õ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ¸Õ¾ Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¡Õ¯Õ¡Õ¶ Õ¡Ö€Õ¿Õ¡Õ¤Ö€Õ¡Õ¶Ö„Õ¨ Õ¶Õ¾Õ¡Õ¦Õ¥Õ¬ Õ§ **19.4%**-Õ¸Õ¾, Õ«Õ¶Õ¹Õ¨ Õ°Õ«Õ´Õ¶Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ´ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ°Õ¡Õ¶Ö„Õ¡Ö€Õ¤ÕµÕ¸Ö‚Õ¶Õ¡Õ¢Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¸Õ¬Õ¸Ö€Õ¿Õ« ÕªÕ¡Õ´Õ¡Õ¶Õ¡Õ¯Õ¡Õ¾Õ¸Ö€ Õ¡Õ¶Õ¯Õ´Õ¡Õ´Õ¢:
        """)

elif page == "ÕÕ Ö‡ Ô²Õ¡Ö€Õ±Ö€ Õ¿Õ¥Õ­Õ¶Õ¸Õ¬Õ¸Õ£Õ«Õ¡Õ¶Õ¥Ö€":
    st.title(page)
    st.markdown("---")
    
    st.subheader("ÕÕ Õ¸Õ¬Õ¸Ö€Õ¿Õ« Õ¡Õ³Õ« Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶")
    df_sum = load_data('adv_summary_stats.csv')
    it_24 = df_sum[df_sum['Õ‘Õ¸Ö‚ÖÕ¡Õ¶Õ«Õ·'] == 'ÕÕ Õ¸Õ¬Õ¸Ö€Õ¿Õ« Õ¡Õ³ (%)']['2024'].values[0]
    it_25 = df_sum[df_sum['Õ‘Õ¸Ö‚ÖÕ¡Õ¶Õ«Õ·'] == 'ÕÕ Õ¸Õ¬Õ¸Ö€Õ¿Õ« Õ¡Õ³ (%)']['2025'].values[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("ÕÕ¥Õ²Õ¥Õ¯Õ¡Õ¿Õ¾Õ¡Õ¯Õ¡Õ¶ Õ¿Õ¥Õ­Õ¶Õ¸Õ¬Õ¸Õ£Õ«Õ¡Õ¶Õ¥Ö€Õ¨ (ÕÕ) Õ°Õ¡Õ¶Õ¤Õ«Õ½Õ¡Õ¶Õ¸Ö‚Õ´ Õ¥Õ¶ Õ€Õ€ Õ¿Õ¶Õ¿Õ¥Õ½Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¡Õ´Õ¥Õ¶Õ¡Õ¡Ö€Õ¡Õ£ Õ¡Õ³Õ¸Õ² Ö‡ Õ¡Ö€Õ¿Õ¡Õ°Õ¡Õ¶Õ¥Õ¬Õ« Õ°Õ¡Õ¿Õ¾Õ¡Õ®Õ¨Ö‰")
        st.metric("ÕÕ Õ¡Õ³ (2024)", f"+{it_24}%")
        st.metric("ÕÕ Õ¡Õ³ (2025 Õ€Õ¸Ö‚Õ¶Õ¾Õ¡Ö€)", f"+{it_25}%", delta="Ô±Õ¯Õ¿Õ«Õ¾Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ·Õ¡Ö€Õ¸Ö‚Õ¶Õ¡Õ¯Õ¡Õ¯Õ¡Õ¶Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶")
        st.write("""
        - ÕˆÕ¬Õ¸Ö€Õ¿Õ« Õ¡Õ³Õ¨ ÕºÕ¡ÕµÕ´Õ¡Õ¶Õ¡Õ¾Õ¸Ö€Õ¾Õ¡Õ® Õ§ Õ©Õ¥Õ› Õ¿Õ¥Õ²Õ¡Õ¯Õ¡Õ¶ Õ¨Õ¶Õ¯Õ¥Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ¦Õ¡Ö€Õ£Õ¡ÖÕ´Õ¡Õ´Õ¢, Õ©Õ¥Õ› Õ´Õ«Õ»Õ¡Õ¦Õ£Õ¡ÕµÕ«Õ¶ Õ¿Õ¥Õ­Õ¶Õ¸Õ¬Õ¸Õ£Õ«Õ¡Õ¯Õ¡Õ¶ Õ°Õ½Õ¯Õ¡Õ¶Õ¥Ö€Õ« Õ¶Õ¥Ö€Õ¯Õ¡ÕµÕ¸Ö‚Õ©ÕµÕ¡Õ´Õ¢Ö‰
        - 2025Õ©. Õ°Õ¸Ö‚Õ¶Õ¾Õ¡Ö€Õ«Õ¶ ÕÕ Õ¸Õ¬Õ¸Ö€Õ¿Õ« Õ´Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶Õ¨ Õ®Õ¡Õ¼Õ¡ÕµÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ¨Õ¶Õ¤Õ°Õ¡Õ¶Õ¸Ö‚Ö€ Õ®Õ¡Õ¾Õ¡Õ¬Õ¸Ö‚Õ´ Õ¯Õ¡Õ¦Õ´Õ¥Õ¬ Õ§ **20.9%**:
        """)
    
    with c2:
        # Representative data based on Armenian IT sector structure
        it_subsectors = pd.DataFrame({
            "ÕˆÖ‚Õ²Õ²Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶": ["Ô¾Ö€Õ¡Õ£Ö€Õ¡ÕµÕ«Õ¶ Õ¡ÕºÕ¡Õ°Õ¸Õ¾Õ¸Ö‚Õ´", "ÕÕ¾ÕµÕ¡Õ¬Õ¶Õ¥Ö€Õ« Õ´Õ·Õ¡Õ¯Õ¸Ö‚Õ´", "Ô½Õ¸Ö€Õ°Ö€Õ¤Õ¡Õ¿Õ¾Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", "Ô¿Õ¡ÕºÕ« Õ®Õ¡Õ¼Õ¡ÕµÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€", "Ô±ÕµÕ¬"],
            "Õ„Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶": [55, 15, 12, 10, 8]
        })
        f = px.pie(it_subsectors, values="Õ„Õ¡Õ½Õ¶Õ¡Õ¢Õ¡ÕªÕ«Õ¶", names="ÕˆÖ‚Õ²Õ²Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶", hole=.4, color_discrete_sequence=px.colors.sequential.deep)
        st.plotly_chart(S(f, h=400), width="stretch")

elif page == "ÔºÕ¸Õ²Õ¸Õ¾Ö€Õ¤Õ¡Õ£Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Ö‡ Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡":
    st.title(page)
    st.markdown("---")
    
    df_sum = load_data('adv_summary_stats.csv')
    pop_24_raw = float(df_sum[df_sum['Õ‘Õ¸Ö‚ÖÕ¡Õ¶Õ«Õ·'] == 'Õ„Õ·Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶']['2024'].values[0])
    pop_25_raw = float(df_sum[df_sum['Õ‘Õ¸Ö‚ÖÕ¡Õ¶Õ«Õ·'] == 'Õ„Õ·Õ¿Õ¡Õ¯Õ¡Õ¶ Õ¢Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶']['2025'].values[0])
    pop_24 = pop_24_raw / 1000000
    pop_25 = pop_25_raw / 1000000
    pop_abs_change = int(pop_25_raw - pop_24_raw)
    pop_growth_pct = ((pop_25_raw / pop_24_raw) - 1) * 100
    mig_24 = df_sum[df_sum['Õ‘Õ¸Ö‚ÖÕ¡Õ¶Õ«Õ·'] == 'Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡ÕµÕ« Õ´Õ¶Õ¡ÖÕ¸Ö€Õ¤ (Õ´Õ¡Ö€Õ¤)']['2024'].values[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Ô²Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (2024)", f"{pop_24:.2f} Õ´Õ¬Õ¶")
    c2.metric("Ô²Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ (2025)", f"{pop_25:.2f} Õ´Õ¬Õ¶", delta=f"+{pop_growth_pct:.1f}% | +{pop_abs_change:,} Õ´Õ¡Ö€Õ¤")
    c3.metric("Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡ (2024)", f"{mig_24}", "Õ¦Õ¸Ö‚Õ¿ Õ¡Õ³")
    
    st.subheader("Ô²Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ¤Õ«Õ¶Õ¡Õ´Õ«Õ¯Õ¡Õ¶ Ö‡ Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¸Õ¶ Õ°Õ¸Õ½Ö„Õ¥Ö€Õ¨")
    st.write("""
    2024-2025Õ©Õ©. ÕªÕ¸Õ²Õ¸Õ¾Ö€Õ¤Õ¡Õ£Ö€Õ¡Õ¯Õ¡Õ¶ ÕºÕ¡Õ¿Õ¯Õ¥Ö€Õ¨ Õ¢Õ¶Õ¸Ö‚Õ©Õ¡Õ£Ö€Õ¾Õ¸Ö‚Õ´ Õ§ **Õ¤Ö€Õ¡Õ¯Õ¡Õ¶ Õ´Õ«Õ£Ö€Õ¡ÖÕ«Õ¸Õ¶ Õ´Õ¶Õ¡ÖÕ¸Ö€Õ¤Õ¸Õ¾**, Õ«Õ¶Õ¹Õ¨ Õ§Õ¡Õ¯Õ¡Õ¶ Õ¡Õ¦Õ¤Õ¥ÖÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶ Õ§ Õ¸Ö‚Õ¶Õ¥Õ¶Õ¸Ö‚Õ´ Õ¶Õ¥Ö€Ö„Õ«Õ¶ Õ½ÕºÕ¡Õ¼Õ´Õ¡Õ¶ Ö‡ Õ¡Õ·Õ­Õ¡Õ¿Õ¡Õ¶Ö„Õ« Õ·Õ¸Ö‚Õ¯Õ¡ÕµÕ« Õ¾Ö€Õ¡Ö‰
    
    - **Ô±Õ·Õ­Õ¡Õ¿Õ¸Ö‚ÕªÕ« Õ¡Õ¼Õ¡Õ»Õ¡Ö€Õ¯**: Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¸Õ¶ Õ°Õ¸Õ½Ö„Õ¥Ö€Õ¨ Õ¶ÕºÕ¡Õ½Õ¿Õ¸Ö‚Õ´ Õ¥Õ¶ Õ¢Õ¡Ö€Õ±Ö€ Õ¸Ö€Õ¡Õ¯Õ¡Õ¾Õ¸Ö€Õ¸Ö‚Õ´ Õ¸Ö‚Õ¶Õ¥ÖÕ¸Õ² Õ´Õ¡Õ½Õ¶Õ¡Õ£Õ¥Õ¿Õ¶Õ¥Ö€Õ« Õ¶Õ¥Ö€Õ°Õ¸Õ½Ö„Õ«Õ¶ (Õ°Õ¡Õ¿Õ¯Õ¡ÕºÕ¥Õ½ ÕÕ Õ¸Õ¬Õ¸Ö€Õ¿Õ¸Ö‚Õ´)Ö‰
    - **ÕÕºÕ¡Õ¼Õ¸Õ²Õ¡Õ¯Õ¡Õ¶ ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¡Ö€Õ¯**: Ô²Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ©Õ¾Õ¡Ö„Õ¡Õ¶Õ¡Õ¯Õ« Õ¡Õ³Õ¨ Õ­Õ©Õ¡Õ¶Õ¸Ö‚Õ´ Õ§ Õ¡Õ¼Ö‡Õ¿Ö€Õ« Ö‡ Õ®Õ¡Õ¼Õ¡ÕµÕ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ¸Õ¬Õ¸Ö€Õ¿Õ¶Õ¥Ö€Õ¨Ö‰
    - **Õ„Õ¡Ö€Õ¿Õ¡Õ°Ö€Õ¡Õ¾Õ¥Ö€Õ¶Õ¥Ö€**: Ô²Õ¶Õ¡Õ¯Õ¹Õ¸Ö‚Õ©ÕµÕ¡Õ¶ Õ®Õ¥Ö€Õ¡ÖÕ´Õ¡Õ¶ Õ´Õ«Õ¿Õ¸Ö‚Õ´Õ¶Õ¥Ö€Õ¨ ÕºÕ¡Õ°Õ¡Õ¶Õ»Õ¸Ö‚Õ´ Õ¥Õ¶ Õ¥Ö€Õ¯Õ¡Ö€Õ¡ÕªÕ¡Õ´Õ¯Õ¥Õ¿ Õ½Õ¸ÖÕ«Õ¡Õ¬Õ¡Õ¯Õ¡Õ¶ Ö‡ Õ¿Õ¶Õ¿Õ¥Õ½Õ¡Õ¯Õ¡Õ¶ Õ¼Õ¡Õ¦Õ´Õ¡Õ¾Õ¡Ö€Õ¸Ö‚Õ©ÕµÕ¸Ö‚Õ¶Õ¶Õ¥Ö€Õ« Õ´Õ·Õ¡Õ¯Õ¸Ö‚Õ´Ö‰
    """)
    
    # Simple migration trend visualization
    mig_data = pd.DataFrame({
        "ÕÕ¡Ö€Õ«": ["2021", "2022", "2023", "2024"],
        "Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡ÕµÕ« Õ´Õ¶Õ¡ÖÕ¸Ö€Õ¤ (Õ´Õ¡Ö€Õ¤)": [4500, 15200, 62000, 76900]
    })
    f = px.line(mig_data, x="ÕÕ¡Ö€Õ«", y="Õ„Õ«Õ£Ö€Õ¡ÖÕ«Õ¡ÕµÕ« Õ´Õ¶Õ¡ÖÕ¸Ö€Õ¤ (Õ´Õ¡Ö€Õ¤)", markers=True, line_shape="spline")
    f.update_traces(line_color="#238636", fill='tozeroy')
    st.plotly_chart(S(f, h=400), width="stretch")

