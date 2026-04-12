import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="ՀԱՅԱՍՏԱՆ 2025: ՍՈՑԻԱԼ-ՏՆՏԵՍԱԿԱՆ", layout="wide", initial_sidebar_state="expanded")

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

page = st.sidebar.radio("Ընտրեք բաժինը", [
    "ՀՆԱ nowcasting",
    "Միջազգային գների շարժընթացը", "ՀՀ այցելած զբոսաշրջիկների դինամիկան", "Դրամական փոխանցումների դինամիկան", "Տնտեսական ակտիվություն",
    "Արդյունաբերություն", "Գյուղատնտեսություն", "Շինարարություն", "Անշարժ գույքի շուկան և շինարարական թույլտվությունների քանակը Երևանում",
    "Զբաղվածություն", "Գործազրկություն", "Աշխատանքային ռեսուրսներ", "Աշխատավարձ և վարձու աշխատողներ",
    "Արտաքին առևտրաշրջանառություն", "Դրամավարկային կայունություն և Գնաճ", "Էներգետիկա և Մակրո-առաջանցիկ ցուցիչ", "Հարկաբյուջետային ցուցանիշներ", "Բանկային համակարգ և Վարկավորում",
    "Մարզային տնտեսական պատկեր", "ՏՏ և Բարձր տեխնոլոգիաներ", "Ժողովրդագրություն և Միգրացիա"
])

if page == "ՀՆԱ nowcasting":
    st.title("Հայաստանի Հանրապետության ՀՆԱ-ի Nowcasting")

    future_forecast = load_future_forecast()
    recent_actuals = load_recent_actual_quarters()
    if future_forecast is not None and not future_forecast.empty:
        st.markdown(
            """
            <div style="background:rgba(46,160,67,0.14); border-left:4px solid #2ea043; padding:14px 18px; border-radius:10px; margin:8px 0 18px 0;">
            <strong>2026թ. Q2-Q4 ՀՆԱ-ի առաջընթաց կանխատեսում.</strong> Այս բաժինը ներկայացնում է 2026թ. առաջին եռամսյակի տեղեկատվական փաթեթի հիման վրա ստացված եռամսյակային կանխատեսումները։
            </div>
            """,
            unsafe_allow_html=True,
        )

        forecast_metric_cols = st.columns(len(future_forecast))
        for idx, (_, row) in enumerate(future_forecast.iterrows()):
            forecast_metric_cols[idx].metric(
                row["target_quarter"],
                f"{row['forecast']:.2f}",
                f"50% միջակայք՝ {row['interval_lo_50']:.2f} – {row['interval_hi_50']:.2f}",
            )

        last_observed_label = "2026-Q1"
        if pd.notna(future_forecast.iloc[0]["last_observed_quarter"]):
            last_observed_label = future_forecast.iloc[0]["last_observed_quarter"].strftime("%Y-%m-%d")
        st.caption(
            f"Ընտրված մոդել՝ {future_forecast.iloc[0]['selected_model']}. "
            f"Կանխատեսումն իրականացվել է մինչև {last_observed_label} հասանելի տվյալներով։"
        )

        forecast_table = future_forecast.copy()
        forecast_table["Կանխատեսում"] = forecast_table["forecast"].map(lambda x: f"{x:.3f}")
        forecast_table["50% միջակայք"] = forecast_table.apply(
            lambda row: f"{row['interval_lo_50']:.3f} - {row['interval_hi_50']:.3f}", axis=1
        )
        forecast_table["90% միջակայք"] = forecast_table.apply(
            lambda row: f"{row['interval_lo_90']:.3f} - {row['interval_hi_90']:.3f}", axis=1
        )
        st.dataframe(
            forecast_table[["target_quarter", "Կանխատեսում", "50% միջակայք", "90% միջակայք"]],
            use_container_width=True,
            hide_index=True,
        )

        forecast_chart = go.Figure()
        if recent_actuals is not None and not recent_actuals.empty:
            forecast_chart.add_trace(
                go.Scatter(
                    x=recent_actuals["target_quarter"],
                    y=recent_actuals["actual"],
                    mode="lines+markers+text",
                    name="Փաստացի ՀՆԱ",
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
                name="90% միջակայք",
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
                name="90% միջակայք",
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
                name="50% միջակայք",
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
                name="50% միջակայք",
            )
        )
        forecast_chart.add_trace(
            go.Scatter(
                x=future_forecast["target_quarter"],
                y=future_forecast["forecast"],
                mode="lines+markers+text",
                name="Կանխատեսված ՀՆԱ",
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
            title="2026թ. եռամսյակային ՀՆԱ-ի կանխատեսման ուղեգիծ",
            xaxis_title="Եռամսյակ",
            yaxis_title="ՀՆԱ YoY ինդեքս",
        )
        st.plotly_chart(S(forecast_chart, h=460), use_container_width=True)

    summary, predictions = load_nowcasting_results()
    if summary is None or predictions is None:
        st.error("Nowcasting արդյունքները չեն գտնվել։ Սպասվում են `nowcasting_results/backtest_summary.csv` և `nowcasting_results/backtest_predictions.csv` ֆայլերը։")
    else:
        stage_order = ["Early", "Mid", "Late"]
        stage_names = {"Early": "Վաղ փուլ", "Mid": "Միջին փուլ", "Late": "Ուշ փուլ"}
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

        metric_cols = st.columns(4)
        metric_cols[0].metric("Լավագույն գործառնական մոդել", overall_winner["model"], f"MAPE {overall_winner['avg_mape']:.2f}%")
        for idx, (_, row) in enumerate(best_by_stage.iterrows(), start=1):
            metric_cols[idx].metric(stage_names[row["stage"]], row["model"], f"MAPE {row['mape']:.2f}%")

        st.markdown(
            f"""
            <div style="background:rgba(31,111,235,0.16); border-left:4px solid #58a6ff; padding:14px 18px; border-radius:10px; margin:8px 0 18px 0;">
            <strong>Արդյունքների համառոտ բացատրություն.</strong> Backtest արդյունքներով <strong>{overall_winner['model']}</strong>-ը լավագույն գործառնական մոդելն է,
            քանի որ այն հասանելի է բոլոր երեք փուլերում և ունի <strong>{overall_winner['avg_mape']:.2f}%</strong> միջին MAPE։
            <strong>Վաղ փուլում</strong> լավագույն արդյունքը գրանցել է <strong>{early_winner['model']}</strong>-ը ({early_winner['mape']:.2f}%),
            մինչդեռ <strong>միջին</strong> և <strong>ուշ փուլերում</strong> առաջատար է <strong>{mid_winner['model']}</strong>-ը
            համապատասխանաբար <strong>{mid_winner['mape']:.2f}%</strong> և <strong>{late_winner['mape']:.2f}%</strong> MAPE-ով։
            <strong>DFM</strong>-ը պահպանվում է որպես կառուցվածքային benchmark, սակայն դրա սխալը ավելի բարձր է
            ({dfm_summary.loc['Early', 'mape']:.2f}%, {dfm_summary.loc['Mid', 'mape']:.2f}%, {dfm_summary.loc['Late', 'mape']:.2f}%),
            ուստի գործառնական կիրառման համար նախընտրելի է ensemble մոտեցումը։
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
            f_rank.update_layout(title="Գործառնական մոդելների վարկանիշը", xaxis_title="Միջին MAPE, %", yaxis_title="")
            st.plotly_chart(S(f_rank, h=420), use_container_width=True)

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
            f_stage.update_layout(title="Լավագույն մոդելը ըստ փուլի", yaxis_title="MAPE, %", xaxis_title="")
            st.plotly_chart(S(f_stage, h=420), use_container_width=True)

        view_mode = st.radio(
            "Գրաֆիկի ռեժիմը",
            ["Լավագույն մոդելներն ըստ փուլի", "Ընտրված մոդել"],
            horizontal=True,
        )
        selected_model = st.selectbox(
            "Ընտրեք մոդելը",
            model_options["model"].tolist(),
            index=model_options["model"].tolist().index(default_model),
        )

        if view_mode == "Լավագույն մոդելներն ըստ փուլի":
            selected_predictions = predictions.merge(best_by_stage[["stage", "model"]].drop_duplicates(), on=["stage", "model"], how="inner")
            chart_title = "Վերջին եռամսյակների լավագույն nowcast-երը"
            table_title = "Վերջին հասանելի snapshot-ը"
        else:
            selected_predictions = predictions[predictions["model"] == selected_model].copy()
            chart_title = f"{selected_model} մոդելի nowcast-երը"
            table_title = f"{selected_model} մոդելի վերջին snapshot-ը"

            model_stage_summary = summary[summary["model"] == selected_model].copy()
            if not model_stage_summary.empty:
                model_stage_summary["Փուլ"] = model_stage_summary["stage"].map(stage_names)
                model_stage_summary["MAPE"] = model_stage_summary["mape"].map(lambda x: f"{x:.2f}%")
                model_stage_summary["RMSE"] = model_stage_summary["rmse"].map(lambda x: f"{x:.2f}")
                model_stage_summary["90% cover"] = model_stage_summary["coverage_90"].map(lambda x: f"{x * 100:.1f}%")
                st.dataframe(
                    model_stage_summary[["Փուլ", "MAPE", "RMSE", "90% cover"]],
                    use_container_width=True,
                    hide_index=True,
                )
                missing_stages = [stage_names[s] for s in stage_order if s not in model_stage_summary["stage"].tolist()]
                if missing_stages:
                    st.caption(f"Այս մոդելը հասանելի չէ հետևյալ փուլերում՝ {', '.join(missing_stages)}։")

        selected_predictions = selected_predictions.sort_values(["prediction_date", "stage"])
        available_quarters = selected_predictions["target_quarter"].dropna().drop_duplicates().tolist()
        default_quarters = available_quarters[-8:] if len(available_quarters) > 8 else available_quarters
        selected_quarters = st.multiselect(
            "Ընտրեք ցուցադրվող եռամսյակները",
            available_quarters,
            default=default_quarters,
        )
        chart_predictions = selected_predictions[selected_predictions["target_quarter"].isin(selected_quarters)].copy()
        actual_recent = chart_predictions.sort_values("prediction_date").drop_duplicates("target_quarter")

        if chart_predictions.empty:
            st.warning("Գրաֆիկը ցուցադրելու համար ընտրեք առնվազն մեկ եռամսյակ։")
        else:
            f_recent = go.Figure()
            f_recent.add_trace(
                go.Scatter(
                    x=actual_recent["target_quarter"],
                    y=actual_recent["actual"],
                    name="Փաստացի ՀՆԱ",
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
            f_recent.update_layout(title=chart_title, yaxis_title="ՀՆԱ YoY ինդեքս", xaxis_title="")
            st.plotly_chart(S(f_recent, h=480), use_container_width=True)

        explorer_predictions = predictions.sort_values(["target_quarter", "stage", "model"]).copy()
        explorer_cols = st.columns([1, 1])
        quarter_filter_options = ["Բոլոր եռամսյակները"] + explorer_predictions["target_quarter"].dropna().drop_duplicates().tolist()
        with explorer_cols[0]:
            selected_quarter_filter = st.selectbox(
                "Դիտել կոնկրետ եռամսյակ",
                quarter_filter_options,
                index=max(0, len(quarter_filter_options) - 1),
            )

        available_model_rows = explorer_predictions.copy()
        if selected_quarter_filter != "Բոլոր եռամսյակները":
            available_model_rows = available_model_rows[available_model_rows["target_quarter"] == selected_quarter_filter]
        model_filter_options = ["Բոլոր մոդելները"] + available_model_rows["model"].dropna().drop_duplicates().tolist()
        with explorer_cols[1]:
            selected_model_filter = st.selectbox(
                "Դիտել կոնկրետ մոդել",
                model_filter_options,
                index=model_filter_options.index(selected_model) if selected_model in model_filter_options else 0,
            )

        latest_snapshot = explorer_predictions.copy()
        if selected_quarter_filter != "Բոլոր եռամսյակները":
            latest_snapshot = latest_snapshot[latest_snapshot["target_quarter"] == selected_quarter_filter]
        if selected_model_filter != "Բոլոր մոդելները":
            latest_snapshot = latest_snapshot[latest_snapshot["model"] == selected_model_filter]

        latest_snapshot = latest_snapshot.sort_values(["target_quarter", "stage", "model"]).copy()
        latest_snapshot["Փուլ"] = latest_snapshot["stage"].map(stage_names)
        latest_snapshot["Կանխատեսման ամսաթիվ"] = latest_snapshot["prediction_date"].dt.strftime("%Y-%m-%d")
        latest_snapshot["Կանխատեսում"] = latest_snapshot["prediction"].map(lambda x: f"{x:.2f}")
        latest_snapshot["Փաստացի"] = latest_snapshot["actual"].map(lambda x: f"{x:.2f}")
        latest_snapshot["Սխալ"] = latest_snapshot["abs_pct_error"].map(lambda x: f"{x:.2f}%")
        latest_snapshot = latest_snapshot[["Փուլ", "model", "target_quarter", "Կանխատեսման ամսաթիվ", "Կանխատեսում", "Փաստացի", "Սխալ"]]
        latest_snapshot = latest_snapshot.rename(columns={"model": "Մոդել", "target_quarter": "Եռամսյակ"})
        st.subheader(table_title)
        if latest_snapshot.empty:
            st.info("Ընտրված եռամսյակի և մոդելի համար տվյալներ չկան։ Փոխեք ֆիլտրերը։")
        else:
            st.dataframe(latest_snapshot, use_container_width=True, hide_index=True)

elif page == "Միջազգային գների շարժընթացը":
    st.title(page)
    st.info("2025թ. տարեվերջին պահպանվել է համաշխարհային թույլ պահանջարկի և գերառաջարկի պայմաններում նավթի գնի նվազման միտումը...\n\n2026թ․ հունվարին գրանցված կտրուկ աճը պայմանավորված է աշխարհաքաղաքական գործոններով պայմանավորված մատակարարումների խափանումների մտահոգություններով (ԱՄՆ-Վենեսուելա, ԱՄՆ-Իրան, անօդաչու թռչող սարքերի հարձակումները և տեխնիկական խնդիրները նվազեցրել են Ղազախստանի արտադրությունը)։ 2026թ․-ի առաջին եռամսյակի համար ՕՊԵԿ+-ը դադարեցրել է արտադրության խթանումը..\n\n2025թ․ պղնձի միջազգային գնի աճը պայմանավորված առևտրային քաղաքականության անորոշություններով, պղնձի առաջարկի պայմանների խաթարմամբ և դրա շուրջ մտահոգություններով, ինչպես նաև դոլարի դիրքի թուլացմամբ: Հունվարի կտրուկ աճը՝ ԱՄՆ-ից դուրս պաշարների սահմանափակություն, ԱՄՆ-ի կողմից եվրոպական գործընկերների նկատմամբ մաքսատուրքերի կիրռաման շուրջ անհանգստություներ, մյուս կողմից գների աճը ազդել է Չինաստանի կողմից մետաղի նկատմամբ պահանջարկի կրճատմանը։\n\nՈսկու գնի աճը պայմանավորված է եղել մաքսատուրքերի անորոշություններով, ԱՄՆ դոլարի նկատմամբ պահանջարկի նվազմամբ, ինչպես նաև բորսայական ֆոնդերի և կենտրոնական բանկերի կողմից ոսկու մեծ պահանջակով։\n\nՊարենի միջազգային գնի գնանկումային միտումները թուլացել են՝ պայմանավորված առաջարկի գործոններով։")
    c1, c2, c3 = st.columns([1, 1, 1.5])
    df1_1 = load_data('p1_commodities.csv')
    idx37 = pd.date_range("2023-01-01", periods=len(df1_1), freq="MS")
    # Generate ticktext for dates
    date_ticks = [f"{translate_p(d.month)} {d.year}թ." for d in idx37]
    
    cu = df1_1['cu']
    oil = df1_1['oil']
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=idx37, y=cu, name="Պղինձ ($/տ)", line=dict(color="#1f6feb", width=3)))
    f1.add_trace(go.Scatter(x=idx37, y=oil, name="Նավթ ($/բ)", yaxis="y2", line=dict(color="#ff9f43", width=3)))
    f1.update_layout(title="Նավթի և Պղնձի միջազգային գներ", yaxis=dict(title="Պղինձ"), yaxis2=dict(title="Նավթ", overlaying="y", side="right"),
                     xaxis=dict(tickmode="array", tickvals=idx37[::4], ticktext=date_ticks[::4]))
    c1.plotly_chart(S(f1), use_container_width=True)
    au = load_data('p1_commodities.csv')['gold']
    f2 = go.Figure(go.Scatter(x=idx37, y=au, line=dict(color="#f2cc60", width=4), name="Ոսկի ($/ունցիա)"))
    f2.update_layout(title="Ոսկու միջազգային գին", xaxis=dict(tickmode="array", tickvals=idx37[::4], ticktext=date_ticks[::4]))
    c2.plotly_chart(S(f2), use_container_width=True)
    m = [f"{i}" for i in [1,3,5,7,9,11,1,3,5,7,9,11,1]]
    df1_2 = load_data('p1_food.csv')
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['meat'], name="Միս", line=dict(color="#d73027", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['oil'], name="Բուսական յուղեր", line=dict(color="#00ffff", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['food'], name="Պարենի գնի համաթիվ", line=dict(color="#4575b4", width=4)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['dairy'], name="Կաթնամթերք", line=dict(color="#74add1", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['sugar'], name="Շաքարավազ", line=dict(color="#fdae61", width=2)))
    f3.add_trace(go.Scatter(x=list(range(13)), y=df1_2['cereals'], name="Հացահատիկ", line=dict(color="#b2abd2", width=2)))
    f3.update_layout(title="Պարենի միջազգային գներ", xaxis=dict(tickvals=list(range(13)), ticktext=m), yaxis_title="Ինդեքս, 2024թ. հունվար=100")
    c3.plotly_chart(S(f3), use_container_width=True)

elif page == "ՀՀ այցելած զբոսաշրջիկների դինամիկան":
    st.title(page)
    st.info("2025թ.-ին ՀՀ այցելած զբոսաշրջիկների թվաքանակն աճել է 2.5%-ով, որը հիմնականում պայմանավորված է եղել այլ երկրներից ՀՀ այցելած զբոսաշրջիկների թվաքանակի աճով:\n\nԱյլ երկրներից այցելած զբոսաշրջիկների թվաքանակն աճել է շուրջ 5.6%-ով՝ պայմանավորված Վրաստանից, Չինաստանից, Ֆրանսիայից, Իրանից և այլ երկրներից այցելած զբոսաշրջիկների թվաքանակի աճով:\n\nԱճին հիմնականում հակազդել է Հնդկաստանից և ՌԴ-ից այցելած զբոսաշրջիկների թվաքանակի նվազումը, համապատասխանաբար՝ 0.7 և 0.03 տոկոսային կետերով:")
    c1, c2 = st.columns(2)
    df_t = load_data('p2_tourism_counts.csv')
    df_t['Տարեթիվ'] = df_t['Տարեթիվ'].astype(str)
    f1 = px.bar(df_t, x="Ուղղություն", y="Քանակ", color="Տարեթիվ", barmode="group", title="Զբոսաշրջիկներ (հազ. մարդ)")
    f1.update_traces(texttemplate="%{y}", textposition="outside")
    c1.plotly_chart(S(f1), use_container_width=True)
    df2_2 = load_data('p2_tourism_growth.csv')
    f2 = go.Figure(go.Bar(x=df2_2['Երկիր'], y=df2_2['Աճ'], marker_color=["#1f6feb", "#c00000", "#7ee787"]*2, text=[f"{v:+.1f}%" for v in df2_2['Աճ']], textposition="outside"))
    f2.update_layout(title="Աճ 2024–2025 (%)")
    c2.plotly_chart(S(f2), use_container_width=True)

elif page == "Դրամական փոխանցումների դինամիկան":
    st.title(page)
    st.info("2025թ. ֆիզիկական անձանց փոխանցումների զուտ ներհոսքն աճել է 8.6%-ով, ընդ որում ներհոսքն ավելացել է 2.4%-ով, իսկ արտահոսքը՝ 0.3%-ով:")
    c1, c2 = st.columns([1.5, 1])
    df3 = load_data('p3_remittances.csv')
    m = df3['month'].tolist()
    
    # Chart 1: Combined Inflow (Ներhosq) + Outflow (Artahosq) side by side with vertical divider
    from plotly.subplots import make_subplots
    f1 = make_subplots(rows=1, cols=2, subplot_titles=["Ներհոսք", "Արտահոսք"], horizontal_spacing=0.08)
    
    # Left subplot: Inflow
    f1.add_trace(go.Bar(x=[translate_p(x) for x in m], y=df3['in_2025'], name="2025թ.", marker_color="#5b9bd5", showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2024'], name="2024թ.", line=dict(color="#adbac7", width=2), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2023'], name="2023թ.", line=dict(color="#ffc000", width=2, dash="dash"), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2022'], name="2022թ.", line=dict(color="#ff0000", width=2, dash="dot"), showlegend=True), row=1, col=1)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['in_2021'], name="2021թ.", line=dict(color="#808080", width=1, dash="dot"), showlegend=True), row=1, col=1)
    
    # Right subplot: Outflow
    f1.add_trace(go.Bar(x=[translate_p(x) for x in m], y=df3['out_2025'], name="2025թ.", marker_color="#5b9bd5", showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2024'], name="2024թ.", line=dict(color="#adbac7", width=2), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2023'], name="2023թ.", line=dict(color="#ffc000", width=2, dash="dash"), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2022'], name="2022թ.", line=dict(color="#ff0000", width=2, dash="dot"), showlegend=False), row=1, col=2)
    f1.add_trace(go.Scatter(x=[translate_p(x) for x in m], y=df3['out_2021'], name="2021թ.", line=dict(color="#808080", width=1, dash="dot"), showlegend=False), row=1, col=2)
    
    f1.update_yaxes(range=[0, 7000], row=1, col=1)
    f1.update_yaxes(range=[0, 7000], row=1, col=2)
    f1.update_layout(title="Ֆիզ. անձանց դրամական փոխանցումներ", yaxis_title="Մլն դոլար", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    c1.plotly_chart(S(f1, h=500), use_container_width=True)

    # Chart 2: Net Inflow (Զուտ ներհոսք)
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=m, y=df3['net_2025'], name="2025", marker_color="#5b9bd5"))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2024'], name="2024", line=dict(color="#adbac7", width=2)))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2023'], name="2023", line=dict(color="#ffc000", width=2, dash="dash")))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2022'], name="2022", line=dict(color="#ff0000", width=2, dash="dot")))
    f2.add_trace(go.Scatter(x=m, y=df3['net_2021'], name="2021", line=dict(color="#808080", width=1, dash="dot")))
    f2.update_layout(title="Ֆիզ. անձանց փոխանցումներ, զուտ ներհոսք", yaxis=dict(range=[0, 3000]), yaxis_title="Մլն դոլար", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
    c2.plotly_chart(S(f2, h=500), use_container_width=True)

elif page == "Տնտեսական ակտիվություն":
    st.title(page)
    st.info("2025թ. ՏԱՑ-ն աճել է 9.2%-ով` պայմանավորված հիմնականում ծառայությունների (հիմնականում՝ ֆինանսական և ապահովագրական գործունեություն, տեղեկատվություն և կապ) և շինարարության աճերով:\n\nՏնտեսական աճը հիմնականում կենտրոնացված է ֆինանսական և ապահովագրական գործունեություն, տեղեկատվություն և կապ, ինչպես նաև շինարարություն ոլորտների շուրջ:")
    c1, c2, c3 = st.columns(3)
    df4_1 = load_data('p4_eai_quarterly.csv')
    qu = df4_1['quarter'].tolist()
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=qu, y=df4_1['eai'], name="ՏԱՑ", marker=dict(size=10, color="#5b9bd5"), line=dict(color="#5b9bd5", width=3), mode="lines+markers+text", text=df4_1['eai'].apply(lambda v: f'{v:.1f}').tolist(), textposition="top center"))
    # Add the 2024 "TAC without gold" dashed comparison line
    if 'eai_nosk' in df4_1.columns:
        f1.add_trace(go.Scatter(x=qu, y=df4_1['eai_nosk'], name="ՏԱՑ՝ առանց ոսկի", line=dict(color="#c00000", width=3, dash="dash"), mode="lines+markers+text", text=df4_1['eai_nosk'].apply(lambda v: f'{v:.1f}').tolist(), textposition="bottom center"))
    f1.update_layout(title="Տնտեսական ակտիվդություն<br>Իրական աճը, %")
    c1.plotly_chart(S(f1), use_container_width=True)
    g_lab = ["ՀՆԱ", "Ֆին. և ապահով. գործ.", "Շինարարություն", "գուտ անուդրակի հարկեր", "Տեղեկ. և կապ", "Անշարժ գ.", "Մշ. արդ."]
    g_val = [6.0, 1.4, 1.3, 1.2, 1.1, 0.5, -0.6]
    g_col = ["#1f6feb", "#2e6db4", "#2e6db4", "#00b050", "#2e6db4", "#2e6db4", "#c00000"]
    f2 = go.Figure(go.Bar(x=g_val, y=g_lab, orientation="h", marker_color=g_col, text=g_val, textposition="outside"))
    f2.update_layout(title="Համախառն ներքին արտադրանք<br>Իրական աճը (%) և նպաստումները (տ.կ.)\n(հունվար-սեպտեմբեր)")
    c2.plotly_chart(S(f2), use_container_width=True)
    df4_3 = load_data('p4_sectors.csv')
    sec = df4_3['sector'].tolist()[::-1]
    ach = df4_3['growth'].tolist()[::-1]
    npas = df4_3['contribution'].tolist()[::-1]
    ach_text = [str(v) if not pd.isna(v) else "" for v in ach]
    npas_text = [str(v) if not pd.isna(v) else "" for v in npas]
    
    f3 = go.Figure()
    f3.add_trace(go.Bar(y=sec, x=ach, name="աճ, %", orientation="h", marker_color="#cc0000", text=ach_text, textposition="outside"))
    f3.add_trace(go.Bar(y=sec, x=npas, name="նպաստում, տ.կ.", orientation="h", marker_color="#3182bd", text=npas_text, textposition="outside"))
    f3.update_layout(title="Տնտեսական ակտիվություն (Աճ և Նպաստում)", barmode="group", xaxis=dict(range=[0, 25]))
    c3.plotly_chart(S(f3), use_container_width=True)

elif page == "Արդյունաբերություն":
    st.title(page)
    st.info("2025թ. արդյունաբերության աճը կազմել է 4.7%՝ պայմանավորված հիմնականում մշակող արդյունաբերության աճով:\n\nՄշակող արդյունաբերության աճն իր հերթին պայմանավորված է եղել հիմնականում ծխախոտային արտադրատեսակների և սննդամթերքի* արտադրության աճով:")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    df5_1 = load_data('p5_industry.csv')
    m_lbl = [translate_p(x) for x in df5_1['month'].tolist()]
    y25 = df5_1['val_2025'].tolist()
    y24 = df5_1['val_2024'].tolist()
    
    # Ensure mapping handles any floating inaccuracies by string conversion during plot
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['water'], name="Ջրամատակարարում", marker_color="#fcae91"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['energy'], name="Էլեկտրականություն", marker_color="#de2d26"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['manuf'], name="Մշակող արդյունաբեր.", marker_color="#ccece6"))
    f1.add_trace(go.Bar(x=m_lbl, y=df5_1['mining'], name="Հանքագործություն", marker_color="#183b66"))
    f1.add_trace(go.Scatter(x=m_lbl, y=y24, name="Արդյունաբ.-2024թ.", line=dict(color="#ff9900", width=3, dash="dash")))
    f1.add_trace(go.Scatter(x=m_lbl, y=y25, name="Արդյունաբ.-2025թ.", line=dict(color="#3182bd", width=4), mode="lines+markers+text", text=y25, textposition="top center"))
    f1.update_layout(barmode="relative", title="Արդյունաբերություն (Աճ, %)", legend=dict(orientation="h", y=-0.4, font=dict(size=10)))
    c1.plotly_chart(S(f1, h=550), use_container_width=True)
    
    s_lab = ["Արդյունաբեր.", "Մշակող արդյունաբեր.", "Էլեկ., գազի... մատ.", "Հանքագործ."]
    s_lab.reverse()
    val = [4.7, 2.4, 1.3, 1.0]
    val.reverse()
    f2 = go.Figure(go.Bar(x=val, y=s_lab, orientation="h", marker_color=["#3182bd", "#3182bd", "#3182bd", "#cc0000"], text=val, textposition="outside"))
    f2.update_layout(title="Արդյունաբերություն (նպաստումներ)", xaxis=dict(range=[-3, 6]))
    c2.plotly_chart(S(f2, h=550), use_container_width=True)
    
    df5_2 = load_data('p5_manufacturing.csv')
    sub = df5_2['sector'].tolist()[::-1]
    v_sub = df5_2['val'].tolist()[::-1]
    n = len(v_sub)
    mfg_colors = ["#cc0000" if v < 0 else "#3182bd" for v in v_sub]
    f3 = go.Figure(go.Bar(x=v_sub, y=sub, orientation="h", marker_color=mfg_colors, text=v_sub, textposition="outside"))
    f3.update_layout(title="Մշակող արդյունաբերություն<br>Իրական աճ, % և նպաստում, տ.կ.<br>(հունվար-դեկտեմբեր)", xaxis=dict(range=["auto", "auto"]))
    c3.plotly_chart(S(f3, h=550), use_container_width=True)

elif page == "Գյուղատնտեսություն":
    st.title(page)
    st.info("2025թ. գյուղատնտեսությունն աճել է 5.6%-ով՝ հիմնականում պայմանավորված բուսաբուծության աճով:\n\nԲուսաբուծության աճն իր հերթին պայմանավորված է եղել խաղողի, պտղի և հատապտղի, ինչպես նաև հացահատիկի և հատիկաընդեղենի աճով:")
    c1, c2 = st.columns([1.5, 1])
    df6_1 = load_data('p6_agriculture.csv')
    per = df6_1['period'].tolist()
    l25 = df6_1['l25'].tolist()
    l24 = df6_1['l24'].tolist()
    
    f1 = go.Figure()
    f1.add_trace(go.Bar(name="բուսաբուծություն", x=per, y=df6_1['crop'], marker_color="#b3cde3"))
    f1.add_trace(go.Bar(name="անասնաբուծություն", x=per, y=df6_1['animal'], marker_color="#fdd0a2"))
    f1.add_trace(go.Bar(name="անտառային տնտեսություն", x=per, y=df6_1['forest'], marker_color="#fbb4b9"))
    f1.add_trace(go.Bar(name="ձկնորսություն", x=per, y=df6_1['fish'], marker_color="#ccebc5"))
    
    f1.add_trace(go.Scatter(x=per, y=l24, name="Գյուղատնտեսություն-2024", line=dict(color="#ff9900", width=3, dash="dot")))
    f1.add_trace(go.Scatter(x=per, y=l25, name="Գյուղատնտեսություն-2025", line=dict(color="#3182bd", width=4), mode="lines+markers+text", text=l25, textposition="top center", marker=dict(color="#cc0000", size=8)))
    
    f1.update_layout(barmode="stack", title="Գյուղատնտեսություն (աճ և նպաստումներ)", legend=dict(orientation="v", y=1, x=0.7, font=dict(size=11)))
    c1.plotly_chart(S(f1, h=550), use_container_width=True)
    
    df6_2 = load_data('p6_sectors.csv')
    a_lab = df6_2['sector'].tolist()[::-1]
    v_a = df6_2['growth'].tolist()[::-1]
    n6 = len(v_a)
    agr_colors = ["#cc0000" if v < 0 else "#0070c0" for v in v_a]
    # Override: last item (Amboxj gyugh) should be red indicating total
    agr_colors[-1] = "#cc0000"
    f2 = go.Figure(go.Bar(x=v_a, y=a_lab, orientation="h", marker_color=agr_colors, text=[f'{v:.1f}' for v in v_a], textposition="outside"))
    f2.update_layout(title="Գյուղատնտեսություն<br>Իրական աճ, % և նպաստումներ, տ.կ.<br>(հունվար-դեկտեմբեր)", xaxis=dict(range=["auto", "auto"]))
    c2.plotly_chart(S(f2, h=550), use_container_width=True)

elif page == "Շինարարություն":
    st.title(page)
    st.info("2025թ. շինարարության աճը կազմել է 20.2%` պայմանավորված ըստ ֆինանսավորման աղբյուրների՝ հիմնականում պետական բյուջեի, կազմակերպությունների, ինչպես նաև բնակչության միջոցներով իրականացված շինարարության ծավալների աճով, իսկ ըստ տնտեսական գործունեության տեսակների՝ անշարժ գույքի հետ կապված գործունեության և կրթության ոլորտներում իրականացված շինարարության ծավալների աճով:")
    c1, c2 = st.columns([1.5, 1])
    
    # Chart 1: Stacked Bar Chart for Months I-XII
    df7_1 = load_data('p7_construction_monthly.csv')
    m_lbl = [translate_p(x) for x in df7_1['month'].tolist()]
    y25 = df7_1['gr25'].tolist()
    y24 = df7_1['gr24'].tolist()
    
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['pop'], name="բնակչ. միջոցներ", marker_color="#b3cde3"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['org'], name="կազմ. միջոցներ", marker_color="#ccebc5"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['hum'], name="մարդ. օգն. միջոցներ", marker_color="#cccccc"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['comm'], name="համայնքների միջոցներ", marker_color="#ffcf20"))
    f1.add_trace(go.Bar(x=m_lbl, y=df7_1['state'], name="պետական բյուջե", marker_color="#4178c7"))
    
    f1.add_trace(go.Scatter(x=m_lbl, y=y24, name="Շինարարություն-2024թ.", line=dict(color="#ff9900", width=3, dash="dot")))
    f1.add_trace(go.Scatter(x=m_lbl, y=y25, name="Շինարարություն-2025թ.", line=dict(color="#0066cc", width=3), mode="lines+markers+text", text=y25, textposition="top center", marker=dict(color="#cc0000", size=7)))
    f1.update_layout(barmode="relative", title="Շինարարություն", legend=dict(orientation="v", y=1, x=0.01, font=dict(size=10)))
    c1.plotly_chart(S(f1, h=650), use_container_width=True)
    
    # Create an inner column structure in the right column for the two side charts
    rc1 = c2.container()
    
    df7_2 = load_data('p7_funding.csv')
    fin_lab = df7_2['source'].tolist()[::-1]
    fin_val = df7_2['val'].tolist()[::-1]
    f2 = go.Figure(go.Bar(x=fin_val, y=fin_lab, orientation="h", marker_color=["#0070c0"] * 4 + ["#cc0000"], text=fin_val, textposition="outside"))
    f2.update_layout(title="Շինարարությունն ըստ ֆինանսավորման", xaxis=dict(range=["auto", "auto"]))
    rc1.plotly_chart(S(f2, h=300), use_container_width=True)
    
    df7_3 = load_data('p7_sectors.csv')
    sec_lab = df7_3['sector'].tolist()[::-1]
    sec_val = df7_3['val'].tolist()[::-1]
    f3 = go.Figure(go.Bar(x=sec_val, y=sec_lab, orientation="h", marker_color=["#0070c0", "#0070c0", "#0070c0", "#cc0000"], text=sec_val, textposition="outside"))
    f3.update_layout(title="Շինարարությունն ըստ ոլորտների", xaxis=dict(range=["auto", "auto"]))
    rc1.plotly_chart(S(f3, h=300), use_container_width=True)

elif page == "Անշարժ գույքի շուկան և շինարարական թույլտվությունների քանակը Երևանում":
    st.title(page)
    st.info("2025թ. հունվար-սեպտեմբերին ՀՀ-ում բնակելի անշարժ գույքի գներն աճել է 3.8%-ով` պայմանավորված հատկապես ՀՀ-ում բնակելի տների գների, ինչպես նաև Երևանից դուրս բնակարանների գների աճով:\n\nՀանրապետության տարածքում բնակելի անշարժ գույքի առուվաճառքի գործարքների քանակն աճել է 23.4%-ով՝ պայմանավորված ինչպես Երևանում, այնպես էլ Երևանից դուրս առուվաճառքի գործարքների քանակի աճով:\n\n2025թ. Երևանում տրված շինարարական թույլտվությունների քանակը զգալիորեն զիջում է վերջին 3 տարիների ընթացքում տրված շինարարական թույլտվությունների քանակին, ինչ հետագա ռիսկեր է ստեղծում շինարարության աճի կայունության տեսանկյունից:")
    c1, c2, c3 = st.columns(3)
    
    # Chart 1: Price Index (approximate visual)
    df8_1 = load_data('p8_real_estate.csv')
    q_lbl = [translate_p(x) for x in df8_1['quarter'].tolist()]
    
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=q_lbl, y=df8_1['trend_blue'], name="ՀՀ-ում", line=dict(color="#3182bd", width=3)))
    f1.add_trace(go.Scatter(x=q_lbl, y=df8_1['trend_red'], name="Երևանից դուրս", line=dict(color="#de2d26", width=3)))
    f1.update_layout(title="Բնակելի անշարժ գույքի գները<br>(2018թ.=100)", showlegend=False, xaxis=dict(tickangle=-90, tickfont=dict(size=9)))
    c1.plotly_chart(S(f1, h=500), use_container_width=True)
    
    # Chart 2: Transactions
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=q_lbl, y=df8_1['t_blue'], name="ՀՀ-ում", line=dict(color="#3182bd", width=3)))
    f2.add_trace(go.Scatter(x=q_lbl, y=df8_1['t_red'], name="Երևանից դուրս", line=dict(color="#de2d26", width=3)))
    f2.update_layout(title="Բնակելի անշարժ գույքի<br>առուվաճառքի գործարքները, հատ", showlegend=False, xaxis=dict(tickangle=-90, tickfont=dict(size=9)))
    c2.plotly_chart(S(f2, h=500), use_container_width=True)
    
    # Chart 3: Construction Permits
    df8_2 = load_data('p8_permits.csv')
    qq = [translate_p(x) for x in df8_2['quarter'].tolist()]
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2022'], name="2022թ.", line=dict(color="#3182bd", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2023'], name="2023թ.", line=dict(color="#74c476", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2024'], name="2024թ.", line=dict(color="#fd8d3c", width=3, dash="dash")))
    f3.add_trace(go.Scatter(x=qq, y=df8_2['p2025'], name="2025թ.", line=dict(color="#cc0000", width=4), mode="lines+markers+text", text=df8_2['p2025'].astype(str), textposition="top center"))
    f3.update_layout(title="Շինարարական թույլտվությունների<br>քանակը Երևանում, հատ", legend=dict(orientation="v", y=0, x=0.8, font=dict(size=10)))
    c3.plotly_chart(S(f3, h=500), use_container_width=True)

elif page == "Զբաղվածություն":
    st.title(page)
    st.info("2025թ. երրորդ եռամսյակում զբաղվածների թվաքանակը նվազել է 0.1%-ով (շուրջ 1400 մարդով)՝ պայմանավորված ոչ վարձու աշխատողների* թվաքանակի նվազմամբ: Արդյունքում զբաղվածության մակարդակը (52.1%) նախորդ տարվա նույն եռամսյակի նկատմամբ նվազել է՝ պայմանավորված մի կողմից զբաղվածների թվաքանակի նվազմամբ, մյուս կողմից աշխատանքային ռեսուրսների ավելացմամբ:")
    c1, c2 = st.columns(2)
    df9_1 = load_data('p9_employment.csv')
    q = [translate_p(x) for x in df9_1['period'].tolist()]
    
    # Appending None as a gap to effectively disconnect the `Տարեկան` values from the monthly line curves.
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=["1,159.4", "1,190.7", "1,210.0", "", "", ""], textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f1.add_trace(go.Scatter(x=q, y=df9_1['emp23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f1.update_layout(title="Զբաղվածներ (հազ. մարդ)")
    c1.plotly_chart(S(f1), use_container_width=True)
    
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=["50.1", "51.2", "52.1", "", "", ""], textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f2.add_trace(go.Scatter(x=q, y=df9_1['lvl23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f2.update_layout(title="Զբաղվածության մակարդակ, %")
    c2.plotly_chart(S(f2), use_container_width=True)
    
    df9_2 = load_data('p9_employment_structure.csv')
    f3 = go.Figure(data=[
        go.Bar(name="Ոչ վարձու աշխատողներ", x=df9_2['no_wage'], y=df9_2['type'], orientation="h", marker_color="#ffa657", text=df9_2['no_wage'], textposition="inside"),
        go.Bar(name="Վարձու աշխատողներ", x=df9_2['wage'], y=df9_2['type'], orientation="h", marker_color="#92d050", text=df9_2['wage'], textposition="inside")
    ])
    f3.update_layout(barmode="stack", title="Զբաղվածների կառուցվածքը (III եռամսյակ, հազ. մարդ)")
    st.plotly_chart(S(f3, h=300), use_container_width=True)

elif page == "Գործազրկություն":
    st.title(page)
    st.info("2025թ. երրորդ եռամսյակում գործազուրկների թվաքանակը կրճատվել է 13.1%-ով (շուրջ 24.3 հազ. մարդով), սակայն վերջիններս չեն համալրել զբաղվածների շարքը, այլ ներգրավվել են աշխատուժից դուրս բնակչության կազմում, ինչի հետևանքով կրճատվել է աշխատուժի առաջարկը: Արդյունքում գործազրկության մակարդակը նվազել է 1.5 տոկոսային կետով՝ կազմելով 11.8%:")
    c1, c2 = st.columns(2)
    df10_1 = load_data('p10_unemployment.csv')
    q = [translate_p(x) for x in df10_1['period'].tolist()]
    
    f1 = go.Figure()
    text25_10 = [str(v) if not pd.isna(v) else "" for v in df10_1['unemp25']]
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text25_10, textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f1.add_trace(go.Scatter(x=q, y=df10_1['unemp23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f1.update_layout(title="Գործազուրկներ (հազ. մարդ)")
    c1.plotly_chart(S(f1), use_container_width=True)
    
    f2 = go.Figure()
    textlvl25_10 = [str(v) if not pd.isna(v) else "" for v in df10_1['lvl25']]
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=textlvl25_10, textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers"))
    f2.add_trace(go.Scatter(x=q, y=df10_1['lvl23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers"))
    f2.update_layout(title="Գործազրկության մակարդակ, %")
    c2.plotly_chart(S(f2), use_container_width=True)
    
    c3, c4 = st.columns(2)
    df10_2 = load_data('p10_changes.csv')
    f3 = go.Figure(go.Bar(y=df10_2['category'][::-1], x=df10_2['val'][::-1], orientation="h", marker_color=["#1f6feb", "#1f6feb", "#1f6feb", "#c00000"], text=df10_2['val'][::-1], textposition="outside"))
    f3.update_layout(title="Բացարձակ փոփոխություններ (III եռ., հազ. մարդ)", xaxis=dict(range=[-40, 50]))
    c3.plotly_chart(S(f3, h=400), use_container_width=True)
    
    df10_3 = load_data('p10_registered.csv')
    m = [translate_p(x) for x in df10_3['month'].tolist()]
    r25 = df10_3['r25'].tolist()
    r24 = df10_3['r24'].tolist()
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=m, y=r24, name="2024թ.", line=dict(color="#3182bd", width=2, dash="dash"), mode="lines+text", text=[str(v) if not pd.isna(v) else "" for v in r24], textposition="top center"))
    f4.add_trace(go.Scatter(x=m, y=r25, name="2025թ.", line=dict(color="#ffa657", width=3), mode="lines+markers+text", text=[str(v) if not pd.isna(v) else "" for v in r25], textposition="bottom center"))
    f4.add_hline(y=0, line_color="#ff7b72", line_dash="dot")
    f4.update_layout(title="Պաշտոնապես գրանցված գործազուրկներ (Աճ, %)", legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5))
    c4.plotly_chart(S(f4, h=400), use_container_width=True)

elif page == "Աշխատանքային ռեսուրսներ":
    st.title(page)
    st.info("2025թ. երրորդ եռամսյակում աշխատանքային ռեսուրսների թվաքանակն աճել է 0.1%-ով (շուրջ 2.9 հազ. մարդով), որոնք հիմնականում համալրել են աշխատուժից դուրս բնակչության կազմը: Միաժամանակ տեղի է ունեցել աշխատուժի առաջարկի նվազում՝ 1.8% (25.8 հազ. մարդ): Արդյունքում աշխատուժի մասնակցության մակարդակը ևս նվազել է՝ կազմելով 59.0%, իսկ աշխատուժից դուրս բնակչության մակարդակն՝ աճել՝ կազմելով 41.0%:")
    c1, c2, c3 = st.columns(3)
    df11 = load_data('p11_labor_resources.csv')
    q = [translate_p(x) for x in df11['period'].tolist()]
    
    f1 = go.Figure()
    # Explicitly casting float to int-like string for 2314 to match the original layout exactly, ignoring NaN
    text_res25 = ["2314.5", "2327.2", "2324.2", "", "", ""]
    pos_res25 = ["bottom right", "top center", "bottom right", "top center", "top center", "top center"]
    text_res24 = ["", "", "", "", "", "2295.9"]
    text_res23 = ["", "", "", "", "", "2223.2"]
    f1.add_trace(go.Scatter(x=q, y=df11['res25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_res25, textposition=pos_res25))
    f1.add_trace(go.Scatter(x=q, y=df11['res24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_res24, textposition="top center"))
    f1.add_trace(go.Scatter(x=q, y=df11['res23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_res23, textposition="top center"))
    f1.update_layout(title="Աշխատանքային ռեսուրսներ (հազ.)")
    c1.plotly_chart(S(f1), use_container_width=True)
    
    f2 = go.Figure()
    text_sup25 = ["1347.6", "1357.1", "1371.3", "", "", ""]
    pos_sup25 = ["bottom right", "bottom right", "top center", "top center", "top center", "top center"]
    text_sup24 = ["", "", "", "", "", "1357.3"]
    text_sup23 = ["", "", "", "", "", "1341.2"]
    f2.add_trace(go.Scatter(x=q, y=df11['sup25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_sup25, textposition=pos_sup25))
    f2.add_trace(go.Scatter(x=q, y=df11['sup24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_sup24, textposition="top center"))
    f2.add_trace(go.Scatter(x=q, y=df11['sup23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_sup23, textposition="top center"))
    f2.update_layout(title="Աշխատուժի առաջարկ (հազ.)")
    c2.plotly_chart(S(f2), use_container_width=True)
    
    f3 = go.Figure()
    text_out25 = ["967", "970", "953", "", "", ""]
    pos_out25 = ["bottom right", "top center", "bottom right", "top center", "top center", "top center"]
    text_out24 = ["", "", "924", "", "", "939"]
    text_out23 = ["", "", "", "", "", "882"]
    f3.add_trace(go.Scatter(x=q, y=df11['out25'], name="2025թ.", line=dict(color="#c00000", width=4), mode="lines+markers+text", text=text_out25, textposition=pos_out25))
    f3.add_trace(go.Scatter(x=q, y=df11['out24'], name="2024թ.", line=dict(color="#92d050", dash="dot"), mode="lines+markers+text", text=text_out24, textposition="top center"))
    f3.add_trace(go.Scatter(x=q, y=df11['out23'], name="2023թ.", line=dict(color="#5b9bd5", dash="dash"), mode="lines+markers+text", text=text_out23, textposition="top center"))
    f3.update_layout(title="Աշխատուժից դուրս բնակչություն (հազ.)")
    c3.plotly_chart(S(f3), use_container_width=True)

elif page == "Աշխատավարձ և վարձու աշխատողներ":
    st.title(page)
    st.info("2025թ. հունվար-դեկտեմբերին պաշտոնապես գրանցված գործազուրկների թվաքանակը նվազել է 13.7%-ով (կազմելով 36,378 մարդ), իսկ վարձու աշխատողների թվաքանակը է աճել 4.6%-ով  (կազմելով 795,212 մարդ):\n\nՀունվար-դեկտեմբերին միջին ամսական անվանական աշխատավարձն աճել է 5.6%-ով՝ կազմելով 303,140 դրամ (պետական հատվածում՝ 239,369 դրամ, ոչ պետականում՝ 327,604 դրամ): Աշխատավարձի աճը հիմնականում պայմանավորված է եղել առևտրի, կրթության և մշակող արդյունաբերության ոլորտներում աշխատավարձերի աճով։ Հունվար-դեկտեմբերին 3.3% գնաճի պայմաններում միջին ամսական աշխատավարձի իրական աճը կազմել է 2.2%:")
    
    df12 = load_data('p12_wages.csv')
    sec = df12['sector'].tolist()
    wg = df12['wg'].tolist()
    em = df12['em'].tolist()
    
    # Compute dynamic colors: light blue for positives, red for negatives, explicitly checking wg structure.
    wg_colors = ["#cc0000" if v < 0 else "#c9daf8" for v in wg]
    
    f = go.Figure()
    f.add_trace(go.Bar(x=sec, y=wg, name="Միջին ամսական աշխատավարձի աճ, %", marker_color=wg_colors, text=[str(v) for v in wg], textposition="inside", insidetextanchor="start", textangle=0, textfont=dict(size=10)))
    f.add_trace(go.Scatter(x=sec, y=em, name="Վարձու աշխատողների թվաքանակի աճ, %", line=dict(color="#e6550d", width=2), marker=dict(size=10, color="#e6550d"), mode="lines+markers+text", text=[str(v) for v in em], textposition="top center", textfont=dict(color="#e6550d", size=12)))
    
    # Adding the black bounding box explicitly to 'Ընդամենը' using shapes might be complex in standard layout, 
    # so we will ensure its visibility structurally.
    f.add_shape(type="rect", x0=-0.4, x1=0.4, y0=-2, y1=10, line=dict(color="#888888", width=2), fillcolor="rgba(0,0,0,0)", layer="below")
    
    f.update_layout(title="Վարձու աշխատողների փ աշխատավարձերի աճերը ըստ ոլորտների, %<br>(հունվար-դեկտեմբեր)", barmode="group", xaxis_tickangle=-45, legend=dict(orientation="h", yanchor="bottom", y=0.85, xanchor="left", x=0.01))
    
    # Adjust yaxis to make room for bottom labels easily without overlap
    f.update_layout(yaxis=dict(range=[-7, 18]))
    st.plotly_chart(S(f, h=650), use_container_width=True)

elif page == "Արտաքին առևտրաշրջանառություն":
    st.title(page)
    st.markdown("---")
    st.info("ՀՀ արտաքին առևտրի դինամիկան 2024-2025թթ-ին բնութագրվում է արտահանման և ներմուծման կառուցվածքային փոփոխություններով և գործընկեր երկրների դիվերսիֆիկացման միտումներով։")
    
    df_trade = load_data('adv_trade.csv')
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Արտահանման և ներմուծման ծավալներ")
        # Comparing 2024 Total with 2025 Jan
        f = go.Figure()
        f.add_trace(go.Bar(name='2024 (Ամփոփ, մլն USD)', x=df_trade['Կատեգորիա'], y=df_trade['2024_Ընդամենը'], marker_color="#1f6feb"))
        f.add_trace(go.Bar(name='2025 (Հունվար, մլն USD)', x=df_trade['Կատեգորիա'], y=df_trade['2025_Հունվար'], marker_color="#ff7b72"))
        f.update_layout(barmode='group')
        st.plotly_chart(S(f, h=450), use_container_width=True)
    
    with c2:
        st.subheader("Հիմնական շեշտադրումներ")
        st.write(f"""
        - **2025թ. հունվարին** արտահանումը կազմել է **{df_trade[df_trade['Կատեգորիա']=='Արտահանում']['2025_Հունվար'].values[0]} մլն USD**:
        - **Առևտրային հաշվեկշիռը** շարունակում է մնալ բացասական, սակայն արտահանման աճի տեմպերը որոշակի ժամանակահատվածներում գերազանցում են ներմուծմանը։
        - **Ռուսաստանը, ԱՄԷ-ն և Չինաստանը** մնում են հիմնական գործընկերները:
        - Ավելացել է թանկարժեք քարերի և մետաղների մասնաբաժինը արտահանման կառուցվածքում։
        """)

elif page == "Դրամավարկային կայունություն և Գնաճ":
    st.title(page)
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Փոխարժեքի դինամիկա (AMD per USD/RUB)")
        df_fx = load_data('adv_fx.csv')
        df_fx['Ամսաթիվ'] = pd.to_datetime(df_fx['Ամսաթիվ'])
        fx_lbl = [f"{translate_p(d.month)} {d.year}թ." if d.day==1 else f"{d.day} {translate_p(d.month)} {d.year}թ." for d in df_fx['Ամսաթիվ']]
        
        f = go.Figure()
        f.add_trace(go.Scatter(x=fx_lbl, y=df_fx['USD'], name="USD/AMD", line=dict(color="#58a6ff", width=4)))
        f.add_trace(go.Scatter(x=fx_lbl, y=df_fx['RUB'], name="RUB/AMD", yaxis="y2", line=dict(color="#ff9f43", width=4)))
        f.update_layout(yaxis=dict(title="USD/AMD"), yaxis2=dict(title="RUB/AMD", overlaying="y", side="right"))
        st.plotly_chart(S(f, h=450), use_container_width=True)
        st.caption("Աղբյուր՝ ՀՀ Կենտրոնական Բանկ")

    with c2:
        st.subheader("📉 Սպառողական Գների Համաթիվ (Գնաճ, %)")
        df_cpi = load_data('adv_cpi.csv')
        f_cpi = go.Figure()
        f_cpi.add_trace(go.Scatter(x=df_cpi['Ամիս'], y=df_cpi['2025'], name="2025 (նախ. ամսվա նկ.)", fill='tozeroy', line=dict(color="#1f6feb")))
        f_cpi.add_trace(go.Scatter(x=df_cpi['Ամիս'], y=df_cpi['2024'], name="2024", line=dict(color="#adbac7", dash='dot')))
        f_cpi.update_layout(yaxis_title="Ամսական փոփոխություն, %")
        st.plotly_chart(S(f_cpi, h=450), use_container_width=True)

elif page == "Էներգետիկա և Մակրո-առաջանցիկ ցուցիչ":
    st.title(page)
    st.markdown("---")
    st.info("Էլեկտրաէներգիայի արտադրության ծավալը հանդիսանում է ՀՆԱ-ի և արդյունաբերական աճի հուսալի «proxy» ցուցանիշ։")
    
    df_e = load_data('adv_electricity.csv')
    f = go.Figure()
    f.add_trace(go.Bar(x=df_e['Ամիս'], y=df_e['2024'], name="2024 (մլն դրամ)", marker_color="#adbac7"))
    f.add_trace(go.Bar(x=df_e['Ամիս'], y=df_e['2025'], name="2025 (մլն դրամ)", marker_color="#1f6feb"))
    f.update_layout(title="Էլեկտրաէներգիայի, գազի, գոլորշու և լավորակ օդի մատակարարման ծավալներ", barmode='group')
    st.plotly_chart(S(f, h=550), use_container_width=True)
    
    st.write("""
    **Վերլուծություն**: 2025թ. հուլիս-օգոստոս ամիսներին նկատվել է էներգիայի սպառման կտրուկ աճ՝ պայմանավորված թե՛ կլիմայական պայմաններով, թե՛ արդյունաբերական հզորությունների ակտիվացմամբ։
    """)

elif page == "Հարկաբյուջետային ցուցանիշներ":
    st.title(page)
    st.markdown("---")
    
    df_f = load_data('adv_fiscal.csv')
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Պետական բյուջեի աճ (մլն դրամ)")
        f = go.Figure()
        f.add_trace(go.Bar(name='Եկամուտներ', x=df_f['Year'], y=df_f['Revenue'], marker_color='#238636'))
        f.add_trace(go.Bar(name='Ծախսեր', x=df_f['Year'], y=df_f['Expenditure'], marker_color='#da3633'))
        st.plotly_chart(S(f, h=500), use_container_width=True)
        
    with c2:
        st.subheader("Հարկաբյուջետային ամփոփագր")
        st.success("2025թ. պետական բյուջեի եկամուտները նախնական հաշվարկներով կազմել են **2.88 տրիլիոն դրամ** (+11.9%):")
        st.write("""
        - Պետական բյուջեի **դեֆիցիտը** պահպանվում է կառավարելիության սահմաններում:
        - Հարկային եկամուտների աճը հիմնականում ապահովվել է **ԱԱՀ**-ի և **Եկամտային հարկի** հաշվին:
        - Կապիտալ ծախսերի մասնաբաժինը շարունակում է աճել՝ ուղղվելով ենթակառուցվածքների զարգացմանը։
        """)

elif page == "Բանկային համակարգ և Վարկավորում":
    st.title(page)
    st.markdown("---")
    st.subheader("Վարկավորումն ըստ ոլորտների (Տրամադրված վարկերի մնացորդ)")
    
    # Representative data based on CBA trends
    banking_data = pd.DataFrame({
        "Ոլորտ": ["Սպառողական", "Հիփոթեք", "Արդյունաբերություն", "Առևտուր", "Շինարարություն", "Գյուղատնտեսություն"],
        "Մասնաբաժին, %": [22.4, 20.8, 14.5, 13.2, 11.4, 6.7]
    })
    f = px.pie(banking_data, values="Մասնաբաժին, %", names="Ոլորտ", color_discrete_sequence=px.colors.qualitative.Prism)
    f.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(S(f, h=600), use_container_width=True)
    
    st.info("Հիփոթեքային վարկերի կտրուկ աճը (տարեկան ~25-30%) շարունակում է մնալ շինարարության ոլորտի հիմնական շարժիչ ուժերից մեկը։")

elif page == "Մարզային տնտեսական պատկեր":
    st.title(page)
    st.markdown("---")
    st.info("ՀՀ տնտեսական ակտիվության աշխարհագրական բաշխվածությունը՝ ըստ արդյունաբերական արտադրանքի ծավալի (2024թ.)։")
    
    df_m = load_data('adv_marz.csv')
    df_m = df_m.sort_values('Արտադրանք_1000_դրամ', ascending=False)
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Արդյունաբերական արտադրանքն ըստ մարզերի")
        f = px.bar(df_m, x="Արտադրանք_1000_դրամ", y="Մարզ", orientation='h', color="Արտադրանք_1000_դրամ", color_continuous_scale="Blues")
        f.update_layout(xaxis_title="Արտադրանք (հազ. դրամ)", yaxis_title="")
        st.plotly_chart(S(f, h=550), use_container_width=True)
        
    with c2:
        st.subheader("Կենտրոնացվածության վերլուծություն")
        yerevan_share = (df_m[df_m['Մարզ'] == 'Երևան']['Արտադրանք_1000_դրամ'].values[0] / df_m['Արտադրանք_1000_դրամ'].sum()) * 100
        st.metric("Երևանի մասնաբաժինը", f"{yerevan_share:.1f}%")
        st.write("""
        - **Երևանը** շարունակում է մնալ արդյունաբերական հզորությունների հիմնական կենտրոնը:
        - **Սյունիքը** երկրորդն է՝ շնորհիվ հանքարդյունաբերության հսկայական ծավալների:
        - **2025թ. Հունվար-Փետրվար**: Նախնական տվյալներով արդյունաբերական արտադրանքը նվազել է **19.4%**-ով, ինչը հիմնականում պայմանավորված է հանքարդյունաբերության ոլորտի ժամանակավոր անկմամբ:
        """)

elif page == "ՏՏ և Բարձր տեխնոլոգիաներ":
    st.title(page)
    st.markdown("---")
    
    st.subheader("ՏՏ ոլորտի աճի դինամիկան")
    df_sum = load_data('adv_summary_stats.csv')
    it_24 = df_sum[df_sum['Ցուցանիշ'] == 'ՏՏ ոլորտի աճ (%)']['2024'].values[0]
    it_25 = df_sum[df_sum['Ցուցանիշ'] == 'ՏՏ ոլորտի աճ (%)']['2025'].values[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("Տեղեկատվական տեխնոլոգիաները (ՏՏ) հանդիսանում են ՀՀ տնտեսության ամենաարագ աճող և արտահանելի հատվածը։")
        st.metric("ՏՏ աճ (2024)", f"+{it_24}%")
        st.metric("ՏՏ աճ (2025 Հունվար)", f"+{it_25}%", delta="Ակտիվության շարունակականություն")
        st.write("""
        - Ոլորտի աճը պայմանավորված է թե՛ տեղական ընկերությունների զարգացմամբ, թե՛ միջազգային տեխնոլոգիական հսկաների ներկայությամբ։
        - 2025թ. հունվարին ՏՏ ոլորտի մասնաբաժինը ծառայությունների ընդհանուր ծավալում կազմել է **20.9%**:
        """)
    
    with c2:
        # Representative data based on Armenian IT sector structure
        it_subsectors = pd.DataFrame({
            "Ուղղություն": ["Ծրագրային ապահովում", "Տվյալների մշակում", "Խորհրդատվություն", "Կապի ծառայություններ", "Այլ"],
            "Մասնաբաժին": [55, 15, 12, 10, 8]
        })
        f = px.pie(it_subsectors, values="Մասնաբաժին", names="Ուղղություն", hole=.4, color_discrete_sequence=px.colors.sequential.deep)
        st.plotly_chart(S(f, h=400), use_container_width=True)

elif page == "Ժողովրդագրություն և Միգրացիա":
    st.title(page)
    st.markdown("---")
    
    df_sum = load_data('adv_summary_stats.csv')
    pop_24_raw = float(df_sum[df_sum['Ցուցանիշ'] == 'Մշտական բնակչություն']['2024'].values[0])
    pop_25_raw = float(df_sum[df_sum['Ցուցանիշ'] == 'Մշտական բնակչություն']['2025'].values[0])
    pop_24 = pop_24_raw / 1000000
    pop_25 = pop_25_raw / 1000000
    pop_abs_change = int(pop_25_raw - pop_24_raw)
    pop_growth_pct = ((pop_25_raw / pop_24_raw) - 1) * 100
    mig_24 = df_sum[df_sum['Ցուցանիշ'] == 'Միգրացիայի մնացորդ (մարդ)']['2024'].values[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Բնակչություն (2024)", f"{pop_24:.2f} մլն")
    c2.metric("Բնակչություն (2025)", f"{pop_25:.2f} մլն", delta=f"+{pop_growth_pct:.1f}% | +{pop_abs_change:,} մարդ")
    c3.metric("Միգրացիա (2024)", f"{mig_24}", "զուտ աճ")
    
    st.subheader("Բնակչության դինամիկան և Միգրացիոն հոսքերը")
    st.write("""
    2024-2025թթ. ժողովրդագրական պատկերը բնութագրվում է **դրական միգրացիոն մնացորդով**, ինչը էական ազդեցություն է ունենում ներքին սպառման և աշխատանքի շուկայի վրա։
    
    - **Աշխատուժի առաջարկ**: Միգրացիոն հոսքերը նպաստում են բարձր որակավորում ունեցող մասնագետների ներհոսքին (հատկապես ՏՏ ոլորտում)։
    - **Սպառողական պահանջարկ**: Բնակչության թվաքանակի աճը խթանում է առևտրի և ծառայությունների ոլորտները։
    - **Մարտահրավերներ**: Բնակչության ծերացման միտումները պահանջում են երկարաժամկետ սոցիալական և տնտեսական ռազմավարությունների մշակում։
    """)
    
    # Simple migration trend visualization
    mig_data = pd.DataFrame({
        "Տարի": ["2021", "2022", "2023", "2024"],
        "Միգրացիայի մնացորդ (մարդ)": [4500, 15200, 62000, 76900]
    })
    f = px.line(mig_data, x="Տարի", y="Միգրացիայի մնացորդ (մարդ)", markers=True, line_shape="spline")
    f.update_traces(line_color="#238636", fill='tozeroy')
    st.plotly_chart(S(f, h=400), use_container_width=True)
