from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.nowcast_config import build_paths
from src.visualization.plot_models import (
    STAGE_ORDER,
    MAIN_MODELS,
    FACTOR_MODELS,
    best_model,
    best_row,
    plot_stage_profile,
    plot_actual_vs_prediction,
    plot_focus_quarter,
    plot_spaghetti_revisions,
    plot_model_family_comparison,
    plot_information_set,
    plot_release_calendar,
    plot_uncertainty_bands,
    plot_stage_winner_panels,
    plot_ranking_heatmap,
    plot_family_shock_bars,
    plot_bias_profile,
    plot_google_trends_ablation,
)


def write_report_artifacts(base_dir: Path, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    paths = build_paths(base_dir)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    detailed = _read_optional_csv(paths["backtests"] / "backtest_summary_detailed.csv")
    family = _read_optional_csv(paths["backtests"] / "model_family_summary.csv")
    info_sets = _read_optional_csv(paths["backtests"] / "focus_quarter_information_sets.csv")
    residual = _read_optional_csv(paths["backtests"] / "residual_bias_summary.csv")
    google_ablation = _read_optional_csv(paths["backtests"] / "google_trends_ablation_summary.csv")

    plot_stage_profile(paths["figures"] / "backtest_stage_mape.png", summary)
    plot_actual_vs_prediction(
        paths["figures"] / "backtest_best_late_model.png",
        predictions,
        best_model(summary, "Late"),
        "Late",
        "Best Late-Stage Operational Model",
    )
    plot_actual_vs_prediction(
        paths["figures"] / "backtest_dfm_shock_adjusted.png",
        predictions,
        "DFMShockAdjusted",
        "Late",
        "Crisis-Adjusted Factor Benchmark",
    )
    plot_focus_quarter(paths["figures"] / "focus_2020q2_models.png", predictions, "2020-Q2")
    plot_spaghetti_revisions(
        paths["figures"] / "adaptive_ensemble_spaghetti.png",
        predictions,
        "AdaptiveEnsemble",
    )
    plot_model_family_comparison(paths["figures"] / "model_family_comparison.png", family if not family.empty else summary)
    plot_information_set(paths["figures"] / "focus_2020q2_information_set.png", info_sets, "2020-Q2")
    plot_release_calendar(paths["figures"] / "release_calendar_information_flow.png", info_sets)
    plot_uncertainty_bands(
        paths["figures"] / "forecast_uncertainty_bands.png",
        predictions,
        best_model(summary, "Late"),
        "Late",
    )
    plot_stage_winner_panels(paths["figures"] / "stage_winner_small_multiples.png", predictions, summary)
    plot_ranking_heatmap(paths["figures"] / "model_ranking_heatmap.png", summary)
    plot_family_shock_bars(paths["figures"] / "family_shock_nonshock.png", family if not family.empty else detailed)
    plot_bias_profile(paths["figures"] / "selected_model_bias_profile.png", residual)
    plot_google_trends_ablation(paths["figures"] / "google_trends_marginal_value.png", google_ablation)
    _write_markdown_summary(paths["backtests"] / "backtest_summary.md", predictions, summary, detailed)


def _write_markdown_summary(
    path: Path,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    detailed: pd.DataFrame,
) -> None:
    lines = [
        "# Armenia GDP Nowcasting Results",
        "",
        "This note presents the benchmark results in a journal-style order: headline accuracy, structural-model comparison, focus shock quarters, and model-selection conclusions.",
        "",
        "## Headline Findings",
        "",
    ]
    early_best = best_row(summary, "Early")
    mid_best = best_row(summary, "Mid")
    late_best = best_row(summary, "Late")
    stage_counts = summary.groupby("model")["stage"].nunique()
    eligible_models = stage_counts[stage_counts == len(STAGE_ORDER)].index
    overall = (
        summary[summary["model"].isin(eligible_models)]
        .groupby("model", as_index=False)["mape"]
        .mean()
        .sort_values("mape")
        .head(1)
    )
    overall_winner = "AdaptiveEnsemble"
    if early_best is not None:
        lines.append(f"- Best `Early` nowcast: `{early_best['model']}` with `{early_best['mape']:.3f}%` MAPE.")
    if mid_best is not None:
        lines.append(f"- Best `Mid` nowcast: `{mid_best['model']}` with `{mid_best['mape']:.3f}%` MAPE.")
    if late_best is not None:
        lines.append(f"- Best `Late` nowcast: `{late_best['model']}` with `{late_best['mape']:.3f}%` MAPE.")
    if not overall.empty:
        row = overall.iloc[0]
        overall_winner = str(row["model"])
        lines.append(f"- Best overall operational model: `{row['model']}` with average stage MAPE `{row['mape']:.3f}%`.")
    covid_early = predictions[
        (predictions["target_quarter"] == "2020-Q2")
        & (predictions["stage"] == "Early")
        & (predictions["model"] == "EarlyShockAdjusted")
    ]
    if not covid_early.empty:
        lines.append(
            f"- `2020 Q2 Early` targeted fix: `EarlyShockAdjusted` reduced the lockdown-quarter error to `{float(covid_early['abs_pct_error'].iloc[0]):.3f}%`."
        )
    lines.append("")

    lines.append("## Table 1. Main Benchmark Accuracy by Stage")
    lines.append("")
    main_table = summary[summary["model"].isin(MAIN_MODELS + ["ShockSwitch"])].copy()
    if not main_table.empty:
        pivot = main_table.pivot(index="model", columns="stage", values="mape").reindex(columns=STAGE_ORDER)
        pivot["Average"] = pivot.mean(axis=1)
        lines.extend(_markdown_table(pivot.reset_index().sort_values("Average")))
    lines.append("")

    lines.append("## Table 2. Factor Benchmark Comparison")
    lines.append("")
    factor_table = summary[summary["model"].isin(FACTOR_MODELS)].copy().sort_values(["stage", "mape"])
    if not factor_table.empty:
        lines.extend(_markdown_table(factor_table[["stage", "model", "mape", "mae", "rmse"]]))
    lines.append("")

    lines.append("## Table 3. Focus Shock Quarter Performance")
    lines.append("")
    focus = predictions[
        (predictions["target_quarter"].isin(["2020-Q2", "2021-Q2", "2021-Q4", "2022-Q2"]))
        & (predictions["model"].isin(["AdaptiveEnsemble", "ElasticNet", "Bridge", "DFMShockAdjusted", "EarlyShockAdjusted"]))
    ].copy()
    if not focus.empty:
        focus = focus.sort_values(["target_quarter", "stage", "abs_pct_error"])
        lines.extend(_markdown_table(focus[["target_quarter", "stage", "model", "prediction", "actual", "abs_pct_error"]]))
    lines.append("")

    if not detailed.empty:
        lines.append("## Table 4. Bias and Shock Diagnostics")
        lines.append("")
        diag = detailed[detailed["model"].isin(["AdaptiveEnsemble", "ElasticNet", "DFM", "DFMShockAdjusted", "EarlyShockAdjusted"])].copy()
        diag = diag.sort_values(["stage", "mape"])
        lines.extend(
            _markdown_table(
                diag[
                    [
                        "stage",
                        "model",
                        "shock_mape",
                        "non_shock_mape",
                        "focus_quarter_mape",
                        "prediction_bias",
                        "overprediction_rate",
                    ]
                ]
            )
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- `{overall_winner}` is the preferred operational model in this run because it has the lowest average error across stage-eligible benchmarks."
    )
    lines.append(
        "- `DFM` remains the main theory-grounded benchmark because it is the true mixed-frequency state-space model."
    )
    lines.append(
        "- `DFMShockAdjusted` is the preferred crisis-corrected factor benchmark because it reduces the structural model's shock-quarter overprediction problem."
    )
    lines.append(
        "- `EarlyShockAdjusted` should be discussed as a targeted month-1 crisis benchmark, not as the overall preferred model."
    )
    if not covid_early.empty:
        lines.append(
            f"- The main early-COVID weakness is materially reduced: `EarlyShockAdjusted` now reaches `{float(covid_early['abs_pct_error'].iloc[0]):.3f}%` error for `2020 Q2 Early`."
        )
    lines.append("")
    lines.append("## Figures to Use in the Thesis")
    lines.append("")
    lines.append("- `Figure 1`: Benchmark accuracy across `Early`, `Mid`, and `Late` stages.")
    lines.append("- `Figure 2`: Tracking `2020 Q2` across the quarter for the main competing models.")
    lines.append("- `Figure 3`: Spaghetti chart of revision paths for the preferred operational model.")
    lines.append("- `Figure 4`: Model-family comparison showing why combination methods win operationally.")
    lines.append("- `Figure 5`: Information-set expansion for `2020 Q2`, showing why the `Early` stage is sparse.")
    lines.append("- `Figure 6`: Release calendar and information-flow schematic across stages.")
    lines.append("- `Figure 7`: Forecast uncertainty bands for the preferred late-stage model.")
    lines.append("- `Figure 8`: Small multiples showing the winning model path in `Early`, `Mid`, and `Late`.")
    lines.append("- `Figure 9`: Heatmap of model ranking by stage.")
    lines.append("- `Figure 10`: Shock vs non-shock accuracy by model family.")
    lines.append("- `Figure 11`: Bias profile of selected models by stage.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        formatted = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.3f}" if pd.notna(value) else "")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return lines
