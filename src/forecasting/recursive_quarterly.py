from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.nowcast_dataset import load_source_data
from src.nowcast_config import BacktestConfig, build_paths


@dataclass(frozen=True)
class QuarterlyForecastConfig:
    target_column: str = "Real_GDP_Armenia_YoY"
    min_train_quarters: int = 40
    forecast_horizons: int = 3


FEATURE_COLUMNS = (
    "Real_GDP_Russia_YoY",
    "CPI_YoY",
    "Exchange_Rate_AMD_USD_YoY",
    "REER_YoY",
    "Brent_Oil_Price_USD_bbl",
    "Copper_Price_USD_mt",
    "Employment_YoY",
    "Unemployment_Rate_Pct",
    "Primary_Income_Labor_Mln_USD",
    "Secondary_Income_Transfers_Mln_USD",
    "Exchange_Rate_AMD_RUB_Abs",
)
FIG_BG = "#030712"
AX_BG = "#0b1224"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#b7c4de"
GRID_COLOR = "#7c8fb8"


def run_future_quarterly_forecast(
    base_dir: Path,
    config: QuarterlyForecastConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or QuarterlyForecastConfig()
    paths = build_paths(base_dir)
    forecast_dir = paths["results"] / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)

    source = load_source_data(base_dir, BacktestConfig())
    quarterly = source["quarterly"].sort_index().copy()
    quarterly.index = pd.to_datetime(quarterly.index)
    quarterly = quarterly.loc[quarterly[config.target_column].notna()].copy()

    scores = _evaluate_candidate_models(quarterly, config)
    selected_model_name = str(scores.iloc[0]["model"])
    model = _candidate_models()[selected_model_name]
    train = _build_training_frame(quarterly, config)
    model.fit(train.drop(columns=[config.target_column]), train[config.target_column])

    residuals = _collect_recursive_errors(quarterly, model, config)
    forecasts = _forecast_future_quarters(quarterly, model, residuals, config)
    forecasts["selected_model"] = selected_model_name
    forecasts["last_observed_quarter"] = quarterly.index.max()

    forecasts.to_csv(forecast_dir / "future_gdp_forecast.csv", index=False)
    scores.to_csv(forecast_dir / "future_gdp_model_scores.csv", index=False)
    (forecast_dir / "future_gdp_forecast.md").write_text(
        _build_forecast_markdown(forecasts, scores, config),
        encoding="utf-8",
    )
    forecast_figure_path = paths["figures"] / "future_gdp_forecast_dark.png"
    _plot_future_forecast(
        forecast_figure_path,
        quarterly,
        forecasts,
    )
    return forecasts, scores


def _candidate_models() -> dict[str, Pipeline]:
    cv = TimeSeriesSplit(n_splits=3)
    return {
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9],
                        alphas=np.logspace(-3, 1, 20),
                        cv=cv,
                        max_iter=20000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "StackingForecast": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    StackingRegressor(
                        estimators=[
                            ("rf", RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
                            ("lgb", lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
                            ("en", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=cv, max_iter=10000, random_state=42))
                        ],
                        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)
                    )
                )
            ]
        ),
    }


def _build_training_frame(quarterly: pd.DataFrame, config: QuarterlyForecastConfig) -> pd.DataFrame:
    features = _build_feature_frame(quarterly, config.target_column)
    return pd.concat([features, quarterly[[config.target_column]]], axis=1).dropna().copy()


def _build_feature_frame(quarterly: pd.DataFrame, target_column: str) -> pd.DataFrame:
    out = pd.DataFrame(index=quarterly.index)
    target = quarterly[target_column]
    for lag in (1, 2, 4, 8):
        out[f"AR_LAG_{lag}"] = target.shift(lag)
    out["AR_MA4"] = target.shift(1).rolling(4).mean()
    out["AR_DIFF_1_4"] = target.shift(1) - target.shift(4)

    for col in FEATURE_COLUMNS:
        if col not in quarterly.columns:
            continue
        out[f"{col}_LAG1"] = quarterly[col].shift(1)
        out[f"{col}_LAG4"] = quarterly[col].shift(4)
    return out


def _evaluate_candidate_models(quarterly: pd.DataFrame, config: QuarterlyForecastConfig) -> pd.DataFrame:
    training = _build_training_frame(quarterly, config)
    candidate_models = _candidate_models()
    records: list[dict[str, object]] = []
    for model_name, model in candidate_models.items():
        errors = _collect_recursive_errors(quarterly, model, config)
        if errors.empty:
            continue
        for horizon, group in errors.groupby("horizon"):
            records.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "n_obs": int(len(group)),
                    "mae": float(group["abs_error"].mean()),
                    "mape": float(group["abs_pct_error"].mean()),
                }
            )
    scores = pd.DataFrame(records)
    avg = (
        scores.groupby("model", as_index=False)[["mae", "mape"]]
        .mean()
        .sort_values(["mape", "mae", "model"])
        .reset_index(drop=True)
    )
    avg.insert(1, "horizon", "avg_1_to_3")
    avg.insert(2, "n_obs", scores.groupby("model")["n_obs"].sum().reindex(avg["model"]).to_numpy())
    detailed = pd.concat([avg, scores], ignore_index=True)
    return detailed.reset_index(drop=True)


def _collect_recursive_errors(
    quarterly: pd.DataFrame,
    model: Pipeline,
    config: QuarterlyForecastConfig,
) -> pd.DataFrame:
    training = _build_training_frame(quarterly, config)
    if len(training) <= config.min_train_quarters + config.forecast_horizons:
        return pd.DataFrame(columns=["origin_date", "forecast_date", "horizon", "error", "abs_error", "abs_pct_error"])

    records: list[dict[str, object]] = []
    for origin_pos in range(config.min_train_quarters, len(training) - config.forecast_horizons + 1):
        train_end = training.index[origin_pos - 1]
        future_index = list(training.index[origin_pos : origin_pos + config.forecast_horizons])

        train_quarterly = quarterly.loc[:train_end].copy()
        train_frame = _build_training_frame(train_quarterly, config)
        model.fit(train_frame.drop(columns=[config.target_column]), train_frame[config.target_column])

        predicted = _recursive_predict(model, train_quarterly, future_index, config.target_column)
        for horizon, (forecast_date, pred_value) in enumerate(zip(future_index, predicted, strict=True), start=1):
            actual = float(quarterly.loc[forecast_date, config.target_column])
            error = actual - pred_value
            records.append(
                {
                    "origin_date": train_end,
                    "forecast_date": forecast_date,
                    "horizon": horizon,
                    "error": error,
                    "abs_error": abs(error),
                    "abs_pct_error": (abs(error) / abs(actual)) * 100 if actual != 0 else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def _recursive_predict(
    model: Pipeline,
    quarterly: pd.DataFrame,
    forecast_index: list[pd.Timestamp],
    target_column: str,
) -> list[float]:
    temp = quarterly.copy()
    for date in forecast_index:
        if date not in temp.index:
            temp.loc[date, :] = np.nan
    temp = temp.sort_index()

    predictions: list[float] = []
    for date in forecast_index:
        row = _build_feature_frame(temp, target_column).loc[[date]]
        prediction = float(model.predict(row)[0])
        temp.loc[date, target_column] = prediction
        predictions.append(prediction)
    return predictions


def _forecast_future_quarters(
    quarterly: pd.DataFrame,
    model: Pipeline,
    residuals: pd.DataFrame,
    config: QuarterlyForecastConfig,
) -> pd.DataFrame:
    latest_quarter = quarterly.index.max()
    future_index = list(pd.date_range(latest_quarter + pd.offsets.QuarterBegin(), periods=config.forecast_horizons, freq="QS"))
    predicted = _recursive_predict(model, quarterly, future_index, config.target_column)

    records: list[dict[str, object]] = []
    for horizon, (date, prediction) in enumerate(zip(future_index, predicted, strict=True), start=1):
        horizon_errors = residuals[residuals["horizon"] == horizon]["abs_error"]
        q50 = float(horizon_errors.quantile(0.50)) if not horizon_errors.empty else np.nan
        q90 = float(horizon_errors.quantile(0.90)) if not horizon_errors.empty else np.nan
        records.append(
            {
                "forecast_date": date,
                "target_quarter": f"{date.year}-Q{date.quarter}",
                "horizon": horizon,
                "forecast": prediction,
                "interval_lo_50": prediction - q50 if pd.notna(q50) else np.nan,
                "interval_hi_50": prediction + q50 if pd.notna(q50) else np.nan,
                "interval_lo_90": prediction - q90 if pd.notna(q90) else np.nan,
                "interval_hi_90": prediction + q90 if pd.notna(q90) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_forecast_markdown(
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    config: QuarterlyForecastConfig,
) -> str:
    avg_scores = scores[scores["horizon"] == "avg_1_to_3"].copy()
    selected_model = str(forecasts["selected_model"].iloc[0])
    last_observed = pd.Timestamp(forecasts["last_observed_quarter"].iloc[0]).date().isoformat()
    lines = [
        "# Armenia GDP Forecast Through 2026 Q4",
        "",
        f"Last observed quarter used for training: `{last_observed}`.",
        f"Selected recursive quarterly model: `{selected_model}`.",
        "",
        "This is a forward quarterly forecast, not a same-quarter nowcast. For `2026 Q2-Q4`, the model uses the historical quarterly panel and rolls predictions forward recursively.",
        "",
        "## Model Selection",
        "",
    ]
    lines.extend(_markdown_table(avg_scores[["model", "n_obs", "mae", "mape"]]))
    lines.append("")
    lines.append("## Forecasts")
    lines.append("")
    lines.extend(
        _markdown_table(
            forecasts[
                [
                    "target_quarter",
                    "horizon",
                    "forecast",
                    "interval_lo_50",
                    "interval_hi_50",
                    "interval_lo_90",
                    "interval_hi_90",
                ]
            ]
        )
    )
    lines.append("")
    lines.append(
        f"The intervals are empirical error bands derived from historical recursive {config.forecast_horizons}-quarter-ahead forecast errors for the selected model."
    )
    lines.append("")
    return "\n".join(lines)


def _plot_future_forecast(path: Path, quarterly: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    recent = quarterly[["Real_GDP_Armenia_YoY"]].tail(12).copy()
    recent["series"] = "Actual"

    future = forecasts.copy()
    future = future.rename(columns={"forecast_date": "Date"})
    future["Date"] = pd.to_datetime(future["Date"])

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    ax.plot(
        recent.index,
        recent["Real_GDP_Armenia_YoY"],
        color="#f8fafc",
        linewidth=3.2,
        marker="o",
        markersize=6,
        label="Observed GDP YoY",
        zorder=5,
    )

    ax.fill_between(
        future["Date"],
        future["interval_lo_90"],
        future["interval_hi_90"],
        color="#1d4ed8",
        alpha=0.16,
        label="90% empirical band",
        zorder=1,
    )
    ax.fill_between(
        future["Date"],
        future["interval_lo_50"],
        future["interval_hi_50"],
        color="#22d3ee",
        alpha=0.26,
        label="50% empirical band",
        zorder=2,
    )
    ax.plot(
        future["Date"],
        future["forecast"],
        color="#38bdf8",
        linewidth=3.0,
        marker="D",
        markersize=7,
        label="Recursive forecast",
        zorder=6,
    )
    ax.scatter(
        future["Date"],
        future["forecast"],
        s=220,
        color="#38bdf8",
        alpha=0.12,
        zorder=4,
    )

    transition_date = recent.index.max()
    ax.axvline(transition_date, color="#f59e0b", linewidth=1.6, linestyle="--", alpha=0.9, zorder=3)
    ax.text(
        transition_date,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else recent["Real_GDP_Armenia_YoY"].max(),
        " last observed",
        color="#f59e0b",
        va="bottom",
        ha="left",
        fontsize=10,
    )

    quarter_colors = ["#0ea5e9", "#06b6d4", "#14b8a6"]
    for (_, row), color in zip(future.iterrows(), quarter_colors, strict=True):
        start = pd.Timestamp(row["Date"]) - pd.Timedelta(days=35)
        end = pd.Timestamp(row["Date"]) + pd.Timedelta(days=35)
        ax.axvspan(start, end, color=color, alpha=0.05, zorder=0)
        ax.annotate(
            f"{row['target_quarter']}\n{row['forecast']:.2f}",
            xy=(pd.Timestamp(row["Date"]), float(row["forecast"])),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            color=TEXT_MAIN,
            fontsize=10,
            fontweight="bold",
        )

    baseline = float(recent["Real_GDP_Armenia_YoY"].mean())
    ax.axhline(baseline, color="#94a3b8", linestyle=":", linewidth=1.4, alpha=0.85)
    ax.text(
        recent.index.min(),
        baseline + 0.15,
        f"recent 12-quarter mean {baseline:.2f}",
        color=TEXT_MUTED,
        fontsize=9,
        ha="left",
    )

    ax.set_title("Armenia GDP Through 2026 Q4: Forecast Beam", color=TEXT_MAIN, fontsize=16, pad=14)
    ax.set_ylabel("GDP YoY Index", color=TEXT_MAIN)
    ax.set_xlabel("Quarter", color=TEXT_MAIN)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.24)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color("#93a4c9")
        spine.set_alpha(0.32)

    legend = ax.legend(loc="upper left", frameon=True)
    legend.get_frame().set_facecolor(AX_BG)
    legend.get_frame().set_edgecolor("#93a4c9")
    legend.get_frame().set_alpha(0.2)
    for text in legend.get_texts():
        text.set_color(TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        formatted: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.3f}" if pd.notna(value) else "")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return lines
