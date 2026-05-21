import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from views.utils import (
    BASE_DIR,
    load_nowcasting_results,
    load_future_forecast,
    load_google_ablation,
    load_recent_actual_quarters,
    translate_p,
    S,
)

def show_nowcast_page():
    st.title("Հայաստանի Հանրապետության ՀՆԱ-ի Nowcasting")
    nowcast_section = st.radio(
        "Ընտրեք nowcasting բաժնի էջը",
        ["Գլխավոր էջ", "Մոդելների վերլուծություն", "Գծապատկերներ"],
        horizontal=True,
        key="nowcast_section",
    )

    if nowcast_section == "Գլխավոր էջ":
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
            st.plotly_chart(S(forecast_chart, h=460), width="stretch")

        st.stop()

    elif nowcast_section == "Գծապատկերներ":
        import os as _os
        _FIG_DIR = _os.path.join(BASE_DIR, "results", "figures")
        _FALLBACK = _os.path.join(BASE_DIR, "nowcasting_results", "figures")
        _fig_dir = _FIG_DIR if _os.path.isdir(_FIG_DIR) else (_FALLBACK if _os.path.isdir(_FALLBACK) else None)
        st.markdown(
            "<div style='background:rgba(31,111,235,0.13);border-left:4px solid #58a6ff;"
            "padding:12px 18px;border-radius:10px;margin:0 0 18px 0;'>"
            "<strong>Nowcast Գծապատկերների Արխիվ.</strong> "
            "Բոլոր 15 գծապատկերները ավտոմատ կերպով ստեղծվել են վերջին walk-forward backtest-ի (StackingNowcast) արդյունքում:</div>",
            unsafe_allow_html=True,
        )
        _figures = [
            ("backtest_stage_mape.png",               "Մոդելների ճշգրտությունն ըստ փուլի (MAPE %)"),
            ("stage_winner_small_multiples.png",       "Լավագույն մոդելն ըստ փուլի — Փաստացի vs Կանխատեսված"),
            ("backtest_best_late_model.png",           "Ուշ փուլի լավագույն մոդել — StackingNowcast"),
            ("backtest_dfm_shock_adjusted.png",        "DFMShockAdjusted ճգնաժամային benchmark"),
            ("adaptive_ensemble_spaghetti.png",        "Ensemble spaghetti — բոլոր մոդելների կանխատեսման ուղեգծերը"),
            ("forecast_uncertainty_bands.png",         "Կանխատեսման անորոշության միջակայքեր"),
            ("focus_2020q2_models.png",                "2020-Q2 (COVID) ճնշում — մոդելների համեմատություն"),
            ("focus_2020q2_information_set.png",       "2020-Q2 — Տեղեկատվական փաթեթի ընդլայնում ըստ փուլի"),
            ("release_calendar_information_flow.png",  "Տեղեկատվական հոսքի ժամանակացույց ըստ փուլի"),
            ("model_ranking_heatmap.png",              "Մոդելների վարկանիշային heatmap ըստ փուլի"),
            ("model_family_comparison.png",            "Մոդելների ընտանիքների համեմատություն — ML vs Structural vs Combination"),
            ("family_shock_nonshock.png",              "Ճնշումային vs սովորական եռամսյակ — ճշգրտություն ըստ ընտանիքի"),
            ("selected_model_bias_profile.png",        "Ընտրված մոդելների bias profile ըստ փուլի"),
            ("google_trends_marginal_value.png",       "Google Trends-ի լրացուցիչ արժեքը (ablation)"),
            ("future_gdp_forecast_dark.png",           "2026թ. ՀՆԱ-ի կանխատեսումն ըստ ռեկուրսիվ մոդելի"),
        ]
        if _fig_dir is None:
            st.error("Figure directory not found.")
        else:
            for _i in range(0, len(_figures), 2):
                _cols = st.columns(2)
                for _j, _col in enumerate(_cols):
                    if _i + _j >= len(_figures):
                        break
                    _fname, _caption = _figures[_i + _j]
                    _fpath = _os.path.join(_fig_dir, _fname)
                    if _os.path.exists(_fpath):
                        _col.image(_fpath, caption=_caption, use_container_width=True)
                    else:
                        _col.warning(f"{_fname} not found")
        st.stop()

    summary, predictions = load_nowcasting_results()
    ablation_summary, ablation_dm = load_google_ablation()
    if summary is None or predictions is None:
        st.error("Nowcasting արդյունքները չեն գտնվել։ Սպասվում են `results/backtests/backtest_summary.csv` կամ `nowcasting_results/backtest_summary.csv` ֆայլերը։")
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
        if mid_winner["model"] == late_winner["model"]:
            mid_late_summary = (
                f"<strong>միջին</strong> և <strong>ուշ փուլերում</strong> առաջատար է "
                f"<strong>{mid_winner['model']}</strong>-ը համապատասխանաբար "
                f"<strong>{mid_winner['mape']:.2f}%</strong> և <strong>{late_winner['mape']:.2f}%</strong> MAPE-ով։"
            )
        else:
            mid_late_summary = (
                f"<strong>միջին փուլում</strong> առաջատար է <strong>{mid_winner['model']}</strong>-ը "
                f"(<strong>{mid_winner['mape']:.2f}%</strong> MAPE), իսկ "
                f"<strong>ուշ փուլում</strong>՝ <strong>{late_winner['model']}</strong>-ը "
                f"(<strong>{late_winner['mape']:.2f}%</strong> MAPE)։"
            )

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
            իսկ {mid_late_summary}
            <strong>DFM</strong>-ը պահպանվում է որպես կառուցվածքային benchmark, սակայն դրա սխալը ավելի բարձր է
            ({dfm_summary.loc['Early', 'mape']:.2f}%, {dfm_summary.loc['Mid', 'mape']:.2f}%, {dfm_summary.loc['Late', 'mape']:.2f}%),
            ուստի գործառնական կիրառման համար նախընտրելի է ensemble մոտեցումը։
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

            st.markdown("### Այլընտրանքային տվյալների աբլացիոն թեստ")
            ablation_table = early_ablation.copy()
            ablation_table["Տեղեկատվական բլոկ"] = ablation_table["model"].replace(
                {
                    "Base": "Բազային մոդել",
                    "Base+Google": "Բազային + Google կոմպոզիտներ",
                    "Base+Market": "Բազային + շուկայական արագ փոփոխականներ",
                    "Base+Market+Google": "Բազային + շուկայական + Google կոմպոզիտներ",
                }
            )
            ablation_table["MAPE"] = ablation_table["mape"].map(lambda x: f"{x:.3f}%")
            ablation_table["RMSE"] = ablation_table["rmse"].map(lambda x: f"{x:.3f}")
            st.dataframe(
                ablation_table[["Տեղեկատվական բլոկ", "MAPE", "RMSE"]],
                width="stretch",
                hide_index=True,
            )

            dm_text = "n/a" if dm_p_value is None else f"{dm_p_value:.3f}"
            st.markdown(
                f"""
                <div style="background:rgba(46,160,67,0.10); border-left:4px solid #2ea043; padding:14px 18px; border-radius:10px; margin:4px 0 18px 0;">
                <strong>Թարմացված մեկնաբանություն.</strong> Թարմացված <em>Early</em> աբլացիոն թեստում
                շուկայական արագ բլոկը զգալիորեն բարելավում է բազային կառուցվածքային մոդելը,
                իսկ Google կոմպոզիտների ավելացումը դրա վրա տալիս է միայն փոքր լրացուցիչ շահույթ`
                <strong>{google_gain:.3f}</strong> MAPE տոկոսային կետ։
                Այդ ազդեցությունը դեռ վիճակագրորեն վճռական չէ, քանի որ Diebold-Mariano թեստի
                <strong>p = {dm_text}</strong>։
                Այսինքն` այլընտրանքային տվյալների պատմությունը դարձել է մի փոքր ավելի ուժեղ,
                բայց այն շարունակում է մնալ շերտավորված լրացում, ոչ թե որոշիչ հիմնական ազդակ։
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
            f_stage.update_layout(title="Լավագույն մոդելը ըստ փուլի", yaxis_title="MAPE, %", xaxis_title="")
            st.plotly_chart(S(f_stage, h=420), width="stretch")

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
                    width="stretch",
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
            st.plotly_chart(S(f_recent, h=480), width="stretch")

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
            st.dataframe(latest_snapshot, width="stretch", hide_index=True)
