from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


STAGE_ORDER = ["Early", "Mid", "Late"]
MAIN_MODELS = [
    "StackingNowcast",
    "AdaptiveEnsemble",
    "ElasticNet",
    "Bridge",
    "DFMShockAdjusted",
    "DFM",
    "EarlyShockAdjusted",
]
FACTOR_MODELS = ["MIDAS", "DFM", "DFMShockAdjusted"]
FOCUS_MODELS = ["AdaptiveEnsemble", "StackingNowcast", "ElasticNet", "Bridge", "DFMShockAdjusted", "EarlyShockAdjusted"]
FOCUS_SHOCK_QUARTERS = {"2020-Q2", "2021-Q2", "2021-Q4", "2022-Q2"}
SPAGHETTI_MODELS = [
    "AR",
    "AdaptiveEnsemble",
    "StackingNowcast",
    "Bridge",
    "DFM",
    "DFMShockAdjusted",
    "EarlyShockAdjusted",
    "EarlyShockBridge",
    "ElasticNet",
    "GradientBoosting",
    "LightGBM",
    "RandomForest",
    "Shadow",
    "ShockSwitch",
    "SimpleEnsemble",
    "AdaptiveEnsemble",
]
MODEL_COLORS = {
    "AdaptiveEnsemble": "#ff6b6b",
    "StackingNowcast": "#ff9a3c",
    "ElasticNet": "#00d4ff",
    "Bridge": "#ffb703",
    "DFMShockAdjusted": "#7bdff2",
    "DFM": "#c77dff",
    "MIDAS": "#b8c0c2",
    "LightGBM": "#38ef7d",
    "EarlyShockAdjusted": "#80ed99",
    "Actual": "#f7f7f7",
}
FIG_BG = "#050816"
AX_BG = "#0b1020"
TEXT_MAIN = "#f5f7ff"
TEXT_MUTED = "#b6c2e2"
GRID_COLOR = "#6f7ea8"
SPINE_COLOR = "#8fa0c7"


def apply_dark_theme(fig, axes) -> None:
    fig.patch.set_facecolor(FIG_BG)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        if ax is None:
            continue
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_MUTED)
        ax.xaxis.label.set_color(TEXT_MAIN)
        ax.yaxis.label.set_color(TEXT_MAIN)
        ax.title.set_color(TEXT_MAIN)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_alpha(0.35)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(AX_BG)
            legend.get_frame().set_alpha(0.18)
            legend.get_frame().set_edgecolor(SPINE_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_MUTED)


def finalize_dark_figure(fig, path: Path, axes) -> None:
    apply_dark_theme(fig, axes)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def best_row(summary: pd.DataFrame, stage: str) -> pd.Series | None:
    stage_rows = summary[summary["stage"] == stage].sort_values("mape")
    if stage_rows.empty:
        return None
    return stage_rows.iloc[0]


def best_model(summary: pd.DataFrame, stage: str) -> str:
    row = best_row(summary, stage)
    return str(row["model"]) if row is not None else "AdaptiveEnsemble"


def shade_shocks(ax, data: pd.DataFrame) -> None:
    shock = data[data["shock_flag"] == True].sort_values("prediction_date")
    for date in shock["prediction_date"]:
        ax.axvspan(
            pd.Timestamp(date) - pd.Timedelta(days=45),
            pd.Timestamp(date) + pd.Timedelta(days=45),
            color="#7b2cbf",
            alpha=0.14,
        )


def plot_stage_profile(path: Path, summary: pd.DataFrame) -> None:
    data = summary[summary["model"].isin(MAIN_MODELS)].copy()
    if data.empty:
        return
    stage_pos = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for model in MAIN_MODELS:
        subset = data[data["model"] == model].copy()
        if subset.empty:
            continue
        subset["stage_pos"] = subset["stage"].map(stage_pos)
        subset = subset.sort_values("stage_pos")
        ax.plot(
            subset["stage_pos"],
            subset["mape"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=MODEL_COLORS.get(model, "#444444"),
            label=model,
        )
    ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Benchmark Accuracy Across Information Stages")
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    finalize_dark_figure(fig, path, ax)


def plot_actual_vs_prediction(
    path: Path,
    predictions: pd.DataFrame,
    model_name: str,
    stage: str,
    title: str,
) -> None:
    data = predictions[(predictions["stage"] == stage) & (predictions["model"] == model_name)].sort_values("prediction_date")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    shade_shocks(ax, data)
    ax.plot(data["prediction_date"], data["actual"], color=MODEL_COLORS["Actual"], linewidth=1.8, label="Actual GDP")
    ax.plot(
        data["prediction_date"],
        data["prediction"],
        color=MODEL_COLORS.get(model_name, "#555555"),
        linewidth=1.8,
        label=model_name,
    )
    ax.set_title(f"{title}: {model_name} ({stage})")
    ax.set_ylabel("GDP YoY")
    ax.grid(True, alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    finalize_dark_figure(fig, path, ax)


def plot_focus_quarter(path: Path, predictions: pd.DataFrame, target_quarter: str) -> None:
    data = predictions[(predictions["target_quarter"] == target_quarter) & (predictions["model"].isin(FOCUS_MODELS))].copy()
    if data.empty:
        return
    positions = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    actual = float(data["actual"].iloc[0])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhline(actual, color=MODEL_COLORS["Actual"], linewidth=1.7, linestyle="--", label=f"Actual ({actual:.1f})")
    for model in FOCUS_MODELS:
        subset = data[data["model"] == model].copy()
        if subset.empty:
            continue
        subset["stage_pos"] = subset["stage"].map(positions)
        subset = subset.sort_values("stage_pos")
        ax.plot(
            subset["stage_pos"],
            subset["prediction"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=MODEL_COLORS.get(model, "#444444"),
            label=model,
        )
    ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_ylabel("GDP YoY")
    ax.set_title(f"Tracking {target_quarter} Across the Quarter")
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    finalize_dark_figure(fig, path, ax)


def plot_model_family_comparison(path: Path, data: pd.DataFrame) -> None:
    if data.empty:
        return
    if "family" in data.columns:
        grouped = data.groupby(["family", "stage"], as_index=False)["mape"].mean()
    else:
        family_map = {
            "AR": "Structural",
            "Bridge": "Structural",
            "MIDAS": "Structural",
            "DFM": "Structural",
            "DFMShockAdjusted": "Structural",
            "ElasticNet": "ML",
            "RandomForest": "ML",
            "GradientBoosting": "ML",
            "LightGBM": "ML",
            "Huber": "ML",
            "EarlyShockBridge": "ML",
            "EarlyShockAdjusted": "ML",
            "SimpleEnsemble": "Combination",
            "AdaptiveEnsemble": "Combination",
            "ShockSwitch": "Combination",
            "StackingNowcast": "Combination",
        }
        tmp = data.copy()
        tmp["family"] = tmp["model"].map(family_map)
        grouped = tmp.groupby(["family", "stage"], as_index=False)["mape"].mean()
    stage_pos = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for family, marker, color in (
        ("Structural", "o", "#1d3557"),
        ("ML", "s", "#005f73"),
        ("Combination", "D", "#8c1d18"),
    ):
        subset = grouped[grouped["family"] == family].copy()
        if subset.empty:
            continue
        subset["stage_pos"] = subset["stage"].map(stage_pos)
        subset = subset.sort_values("stage_pos")
        ax.plot(
            subset["stage_pos"],
            subset["mape"],
            marker=marker,
            markersize=6,
            linewidth=2.0,
            color=color,
            label=family,
        )
    ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_ylabel("Average MAPE (%)")
    ax.set_title("Accuracy by Model Family")
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    finalize_dark_figure(fig, path, ax)


def plot_spaghetti_revisions(path: Path, predictions: pd.DataFrame, model_name: str) -> None:
    data = predictions[predictions["stage"].isin(STAGE_ORDER)].copy()
    if data.empty:
        return

    date_order = (
        data[["prediction_date", "target_quarter"]]
        .drop_duplicates()
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )
    if date_order.empty:
        return

    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    member_color = "#9aa7c7"
    stage_colors = {
        "Early": "#ff9f1c",
        "Mid": "#00f5d4",
        "Late": "#4ea8de",
    }
    mean_color = "#ff4d6d"
    clim_color = "#9ef01a"

    for quarter in FOCUS_SHOCK_QUARTERS:
        quarter_row = date_order[date_order["target_quarter"] == quarter]
        if quarter_row.empty:
            continue
        shock_date = pd.Timestamp(quarter_row["prediction_date"].iloc[0])
        ax.axvspan(
            shock_date - pd.Timedelta(days=40),
            shock_date + pd.Timedelta(days=40),
            color="#7b2cbf",
            alpha=0.16,
            zorder=0,
        )

    spaghetti_data = data[data["model"].isin(SPAGHETTI_MODELS)].copy()
    spaghetti = (
        spaghetti_data.pivot_table(
            index="prediction_date",
            columns=["model", "stage"],
            values="prediction",
            aggfunc="first",
        )
        .sort_index()
    )
    for col in spaghetti.columns:
        series = spaghetti[col].dropna()
        if series.empty:
            continue
        ax.plot(series.index, series.values, color=member_color, linewidth=0.8, alpha=0.32, zorder=1)

    ensemble_mean = data.groupby("prediction_date", as_index=False)["prediction"].mean().sort_values("prediction_date")
    ax.plot(
        ensemble_mean["prediction_date"],
        ensemble_mean["prediction"],
        color=mean_color,
        linewidth=2.8,
        zorder=4,
        label="Ensemble mean",
    )

    for stage in STAGE_ORDER:
        subset = (
            data[(data["model"] == model_name) & (data["stage"] == stage)]
            .sort_values("prediction_date")
            .dropna(subset=["prediction"])
        )
        if subset.empty:
            continue
        ax.plot(
            subset["prediction_date"],
            subset["prediction"],
            color=stage_colors[stage],
            linewidth=2.4,
            zorder=5,
            label=f"{model_name} {stage}",
        )

    actual = (
        data.groupby(["prediction_date", "target_quarter"], as_index=False)["actual"]
        .first()
        .sort_values("prediction_date")
    )
    ax.plot(
        actual["prediction_date"],
        actual["actual"],
        color=MODEL_COLORS["Actual"],
        linewidth=2.6,
        zorder=6,
        label="Actual GDP",
    )

    climatology = actual["actual"].mean()
    ax.axhline(
        climatology,
        color=clim_color,
        linewidth=2.0,
        linestyle="-.",
        zorder=2,
        label="Sample mean GDP",
    )

    for quarter in sorted(FOCUS_SHOCK_QUARTERS):
        point = actual[actual["target_quarter"] == quarter]
        if point.empty:
            continue
        x_val = pd.Timestamp(point["prediction_date"].iloc[0])
        y_val = float(point["actual"].iloc[0])
        ax.annotate(
            quarter,
            xy=(x_val, y_val),
            xytext=(0, -16 if quarter == "2020-Q2" else 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=mean_color,
            fontweight="bold",
        )

    ax.set_ylabel("GDP YoY")
    ax.set_xlabel("Target quarter")
    ax.set_title("Armenia GDP Nowcast Ensemble Spaghetti")
    ax.grid(True, linestyle=(0, (1.5, 4)), linewidth=0.9, alpha=0.4, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.01,
        0.98,
        "Thin gray lines are individual model-stage nowcasts; highlighted lines show the ensemble mean, the preferred model by stage, the realized GDP path, and the sample mean.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TEXT_MUTED,
    )
    legend_handles = [
        Line2D([0], [0], color=member_color, linewidth=1.0, alpha=0.5, label="Individual nowcasts"),
        Line2D([0], [0], color=mean_color, linewidth=2.8, label="Ensemble mean"),
        Line2D([0], [0], color=stage_colors["Early"], linewidth=2.4, label=f"{model_name} Early"),
        Line2D([0], [0], color=stage_colors["Mid"], linewidth=2.4, label=f"{model_name} Mid"),
        Line2D([0], [0], color=stage_colors["Late"], linewidth=2.4, label=f"{model_name} Late"),
        Line2D([0], [0], color=MODEL_COLORS["Actual"], linewidth=2.6, label="Actual GDP"),
        Line2D([0], [0], color=clim_color, linewidth=2.0, linestyle="-.", label="Sample mean GDP"),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.12))
    finalize_dark_figure(fig, path, ax)


def plot_information_set(path: Path, info_sets: pd.DataFrame, target_quarter: str) -> None:
    if info_sets.empty:
        return
    data = info_sets[info_sets["target_quarter"] == target_quarter].copy()
    if data.empty:
        return
    data["stage_pos"] = data["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)})
    data = data.sort_values("stage_pos")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    width = 0.34
    xpos = np.arange(len(data))
    ax.bar(xpos - width / 2, data["available_official_count"], width=width, color="#264653", label="Official monthly series")
    ax.bar(xpos + width / 2, data["available_fast_count"], width=width, color="#e76f51", label="Fast series")
    ax.set_xticks(xpos, data["stage"])
    ax.set_ylabel("Available series count")
    ax.set_title(f"Information Set for {target_quarter}")
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    finalize_dark_figure(fig, path, ax)


def plot_release_calendar(path: Path, info_sets: pd.DataFrame) -> None:
    stage_stats = []
    for stage in STAGE_ORDER:
        row = None
        if not info_sets.empty and "stage" in info_sets.columns:
            subset = info_sets[info_sets["stage"] == stage]
            if not subset.empty:
                row = subset.iloc[0]
        official_count = int(row["available_official_count"]) if row is not None else 0
        fast_count = int(row["available_fast_count"]) if row is not None else 0
        stage_stats.append((stage, official_count, fast_count))

    descriptions = {
        "Early": "Fast variables only: FX, commodities, Google shock composites, and lagged macro information.",
        "Mid": "First monthly official releases enter: activity, prices, banking, remittances, and ArmStat monthly updates.",
        "Late": "Full within-quarter information set: most monthly official indicators are available before GDP release.",
    }

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.05, 0.365, 0.68]
    box_w = 0.255
    box_h = 0.54
    colors = ["#e76f51", "#f4a261", "#264653"]

    for (stage, official_count, fast_count), x, color in zip(stage_stats, x_positions, colors):
        patch = FancyBboxPatch(
            (x, 0.28),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.3,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        ax.add_patch(patch)
        wrapped_description = textwrap.fill(descriptions[stage], width=31)
        ax.text(x + 0.02, 0.73, stage, fontsize=16, fontweight="bold", color=color, va="top")
        ax.text(
            x + 0.02,
            0.58,
            wrapped_description,
            fontsize=10,
            va="top",
            color=TEXT_MUTED,
            linespacing=1.25,
        )
        ax.text(
            x + 0.02,
            0.40,
            f"Official monthly series: {official_count}\nFast series: {fast_count}",
            fontsize=11,
            va="top",
            color=TEXT_MAIN,
        )

    arrow_style = dict(arrowstyle="->", lw=1.5, color=TEXT_MUTED)
    ax.annotate("", xy=(0.37, 0.52), xytext=(0.30, 0.52), arrowprops=arrow_style)
    ax.annotate("", xy=(0.68, 0.52), xytext=(0.61, 0.52), arrowprops=arrow_style)
    ax.text(
        0.5,
        0.92,
        "Release Calendar and Information Flow Across the Quarter",
        ha="center",
        fontsize=14,
        color=TEXT_MAIN,
    )
    ax.text(
        0.5,
        0.09,
        textwrap.fill(
            "The information set broadens from fast indicators in Early to fuller monthly official releases in Late, "
            "which is why crisis nowcasts improve as the quarter progresses.",
            width=110,
        ),
        ha="center",
        fontsize=10,
        color=TEXT_MUTED,
    )
    finalize_dark_figure(fig, path, ax)


def plot_uncertainty_bands(path: Path, predictions: pd.DataFrame, model_name: str, stage: str) -> None:
    data = predictions[(predictions["stage"] == stage) & (predictions["model"] == model_name)].sort_values("prediction_date").copy()
    if data.empty:
        return
    valid = data.dropna(subset=["interval_lo_90", "interval_hi_90", "interval_lo_50", "interval_hi_50"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    shade_shocks(ax, valid)
    ax.fill_between(
        valid["prediction_date"],
        valid["interval_lo_90"],
        valid["interval_hi_90"],
        color="#bfc7cf",
        alpha=0.6,
        label="90% interval",
    )
    ax.fill_between(
        valid["prediction_date"],
        valid["interval_lo_50"],
        valid["interval_hi_50"],
        color="#7d8597",
        alpha=0.7,
        label="50% interval",
    )
    ax.plot(valid["prediction_date"], valid["actual"], color=MODEL_COLORS["Actual"], linewidth=1.8, label="Actual GDP")
    ax.plot(
        valid["prediction_date"],
        valid["prediction"],
        color=MODEL_COLORS.get(model_name, "#444444"),
        linewidth=1.8,
        label=f"{model_name} point nowcast",
    )
    ax.set_title(f"Forecast Uncertainty Bands: {model_name} ({stage})")
    ax.set_ylabel("GDP YoY")
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    finalize_dark_figure(fig, path, ax)


def plot_stage_winner_panels(path: Path, predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True)
    plotted = False
    for ax, stage in zip(axes, STAGE_ORDER):
        model_name = best_model(summary, stage)
        subset = predictions[(predictions["stage"] == stage) & (predictions["model"] == model_name)].sort_values("prediction_date")
        if subset.empty:
            ax.set_visible(False)
            continue
        plotted = True
        shade_shocks(ax, subset)
        ax.plot(subset["prediction_date"], subset["actual"], color=MODEL_COLORS["Actual"], linewidth=2.0, label="Actual GDP")
        ax.plot(
            subset["prediction_date"],
            subset["prediction"],
            color=MODEL_COLORS.get(model_name, "#444444"),
            linewidth=2.0,
            label=f"{model_name} prediction",
        )
        ax.set_ylabel("GDP YoY")
        ax.set_title(f"{stage}: {model_name}")
        ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
    if not plotted:
        plt.close(fig)
        return
    axes[-1].set_xlabel("Target quarter")
    fig.suptitle("Best-Performing Model by Information Stage", fontsize=15, y=0.99, color=TEXT_MAIN)
    finalize_dark_figure(fig, path, axes)


def plot_ranking_heatmap(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    avg_rank = summary.groupby("model", as_index=False)["mape"].mean().sort_values("mape")
    selected_models = avg_rank["model"].head(10).tolist()
    if "EarlyShockAdjusted" in summary["model"].values and "EarlyShockAdjusted" not in selected_models:
        selected_models = ["EarlyShockAdjusted"] + selected_models[:-1]
    heatmap = (
        summary[summary["model"].isin(selected_models)]
        .pivot(index="model", columns="stage", values="mape")
        .reindex(index=selected_models, columns=STAGE_ORDER)
    )
    if heatmap.empty:
        return
    matrix = heatmap.values.astype(float)
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    im = ax.imshow(masked, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(len(STAGE_ORDER)), STAGE_ORDER)
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_title("Model Ranking Heatmap by Stage")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "--" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color=TEXT_MAIN, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("MAPE (%)")
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_MUTED)
    cbar.ax.yaxis.label.set_color(TEXT_MAIN)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    finalize_dark_figure(fig, path, ax)


def plot_family_shock_bars(path: Path, data: pd.DataFrame) -> None:
    if data.empty:
        return
    if "family" in data.columns:
        grouped = data.groupby(["family", "stage"], as_index=False)[["shock_mape", "non_shock_mape"]].mean()
    else:
        family_map = {
            "AR": "Structural",
            "Bridge": "Structural",
            "MIDAS": "Structural",
            "DFM": "Structural",
            "DFMShockAdjusted": "Structural",
            "ElasticNet": "ML",
            "RandomForest": "ML",
            "GradientBoosting": "ML",
            "LightGBM": "ML",
            "Huber": "ML",
            "EarlyShockBridge": "ML",
            "EarlyShockAdjusted": "ML",
            "SimpleEnsemble": "Combination",
            "AdaptiveEnsemble": "Combination",
            "ShockSwitch": "Combination",
            "StackingNowcast": "Combination",
        }
        tmp = data.copy()
        tmp["family"] = tmp["model"].map(family_map)
        grouped = tmp.groupby(["family", "stage"], as_index=False)[["shock_mape", "non_shock_mape"]].mean()
    families = ["Combination", "ML", "Structural"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    for ax, stage in zip(axes, STAGE_ORDER):
        subset = grouped[grouped["stage"] == stage].set_index("family").reindex(families)
        xpos = np.arange(len(families))
        width = 0.34
        ax.bar(xpos - width / 2, subset["non_shock_mape"], width=width, color="#7fb3d5", label="Non-shock")
        ax.bar(xpos + width / 2, subset["shock_mape"], width=width, color="#d98880", label="Shock")
        ax.set_xticks(xpos, families)
        ax.set_title(stage)
        ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Average MAPE (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Shock vs Non-Shock Accuracy by Model Family", fontsize=15, y=1.02, color=TEXT_MAIN)
    finalize_dark_figure(fig, path, axes)


def plot_bias_profile(path: Path, residual: pd.DataFrame) -> None:
    if residual.empty:
        return
    selected_models = ["AdaptiveEnsemble", "ElasticNet", "Bridge", "DFM", "DFMShockAdjusted", "EarlyShockAdjusted", "AR", "ShockSwitch"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.4), sharex=True)
    for ax, stage in zip(axes, STAGE_ORDER):
        subset = residual[(residual["stage"] == stage) & (residual["model"].isin(selected_models))].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        subset = subset.sort_values("mean_residual")
        colors = ["#c44e52" if val < 0 else "#4c72b0" for val in subset["mean_residual"]]
        ypos = np.arange(len(subset))
        ax.barh(ypos, subset["mean_residual"], color=colors, alpha=0.85)
        ax.axvline(0, color=TEXT_MAIN, linewidth=1.0)
        ax.set_yticks(ypos, subset["model"])
        ax.set_title(stage)
        ax.grid(True, axis="x", alpha=0.25, color=GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_xlabel("Mean residual (actual - prediction)")
    fig.suptitle("Bias Profile of Selected Models by Stage", fontsize=15, y=1.02, color=TEXT_MAIN)
    finalize_dark_figure(fig, path, axes)


def plot_google_trends_ablation(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    spec_order = ["Base", "Base+Market", "Base+Google", "Base+Market+Google"]
    colors = {
        "Base": "#94a3b8",
        "Base+Market": "#38bdf8",
        "Base+Google": "#f97316",
        "Base+Market+Google": "#22c55e",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    stage_pos = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    for spec in spec_order:
        subset = summary[summary["model"] == spec].copy()
        if subset.empty:
            continue
        subset["stage_pos"] = subset["stage"].map(stage_pos)
        subset = subset.sort_values("stage_pos")
        ax.plot(
            subset["stage_pos"],
            subset["mape"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=colors[spec],
            label=spec,
        )
    stages = summary["stage"].drop_duplicates().tolist()
    axis_stages = [stage for stage in STAGE_ORDER if stage in stages]
    ax.set_xticks(range(len(axis_stages)), axis_stages)
    ax.set_ylabel("MAPE (%)")
    title = "Marginal Value of Google Trends Composites by Stage"
    if axis_stages == ["Early"]:
        title = "Marginal Value of Google Trends Composites in the Early Stage"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    finalize_dark_figure(fig, path, ax)
