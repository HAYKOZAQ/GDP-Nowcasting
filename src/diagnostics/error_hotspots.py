from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.nowcast_features import StageDataset


FOCUS_QUARTERS = ("2020-Q2", "2021-Q2", "2021-Q4", "2022-Q2")
FOCUS_MODELS = ("Bridge", "ElasticNet", "DFM", "DFMShockAdjusted", "AdaptiveEnsemble", "ShockSwitch")


def write_error_hotspots(
    backtest_dir: Path,
    predictions: pd.DataFrame,
    stage_datasets: dict[str, StageDataset],
) -> pd.DataFrame:
    enriched = _enrich_predictions(predictions, stage_datasets)
    enriched.to_csv(backtest_dir / "error_hotspots.csv", index=False)
    _write_hotspot_markdown(backtest_dir / "error_hotspots.md", enriched)
    return enriched


def _enrich_predictions(predictions: pd.DataFrame, stage_datasets: dict[str, StageDataset]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        stage_frame = stage_datasets[row.stage].frame
        prediction_date = pd.Timestamp(row.prediction_date)
        stage_row = stage_frame.loc[prediction_date]
        records.append(
            {
                **row._asdict(),
                "monthly_activity_signal": _mean_signal(
                    stage_row,
                    [
                        "OFF_Economic_Activity_Index_Discrete_YoY_LAST",
                        "OFF_Industry_Real_Growth_YoY_LAST",
                        "OFF_Construction_Real_Growth_YoY_LAST",
                        "OFF_Services_Real_Growth_YoY_LAST",
                    ],
                ),
                "monthly_remittance_signal": _mean_signal(
                    stage_row,
                    [
                        "OFF_Remittance_Net_Mln_AMD_YoY_LAST",
                        "OFF_Remittance_Inflow_Mln_AMD_YoY_LAST",
                        "OFF_Remittance_Outflow_Mln_AMD_YoY_LAST",
                    ],
                ),
                "monthly_shock_signal": _mean_signal(
                    stage_row,
                    [
                        "FAST_SHOCK_COMPOSITE_LAST",
                        "FAST_SHOCK_RELOCATION_LAST",
                        "FAST_SHOCK_BANKING_LAST",
                    ],
                ),
                "shock_dummy_active": int(_shock_dummy_active(stage_row)),
            }
        )
    return pd.DataFrame.from_records(records)


def _write_hotspot_markdown(path: Path, enriched: pd.DataFrame) -> None:
    lines = [
        "# Error Hotspots",
        "",
        "This report highlights the largest absolute percentage errors in the pseudo real-time backtest and attaches the key shock diagnostics used for interpretation.",
        "",
    ]

    valid = enriched[enriched["abs_pct_error"].notna()].copy()
    for stage in ("Early", "Mid", "Late"):
        stage_top = valid[valid["stage"] == stage].sort_values("abs_pct_error", ascending=False).head(10)
        lines.append(f"## Worst {stage} Quarters")
        lines.append("")
        lines.extend(
            _markdown_table(
                stage_top[
                    [
                        "prediction_date",
                        "target_quarter",
                        "model",
                        "actual",
                        "prediction",
                        "abs_pct_error",
                        "monthly_activity_signal",
                        "monthly_remittance_signal",
                        "monthly_shock_signal",
                        "shock_dummy_active",
                    ]
                ]
            )
        )
        lines.append("")

    for quarter in FOCUS_QUARTERS:
        subset = enriched[
            (enriched["target_quarter"] == quarter)
            & (enriched["model"].isin(FOCUS_MODELS))
        ].sort_values(["stage", "model"])
        lines.append(f"## Focus Quarter: {quarter}")
        lines.append("")
        lines.extend(
            _markdown_table(
                subset[
                    [
                        "stage",
                        "model",
                        "actual",
                        "prediction",
                        "abs_pct_error",
                        "monthly_activity_signal",
                        "monthly_remittance_signal",
                        "monthly_shock_signal",
                        "shock_dummy_active",
                    ]
                ]
            )
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _mean_signal(stage_row: pd.Series, columns: list[str]) -> float:
    vals = [float(stage_row[col]) for col in columns if col in stage_row.index and pd.notna(stage_row[col])]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _shock_dummy_active(stage_row: pd.Series) -> bool:
    cols = [
        col
        for col in stage_row.index
        if col.startswith("CURR_Dummy_")
        and not col.endswith(("Q1", "Q2", "Q3"))
    ]
    vals = [float(stage_row[col]) for col in cols if pd.notna(stage_row[col])]
    return bool(vals and max(vals) > 0)


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
            if isinstance(value, float):
                formatted.append(f"{value:.3f}" if pd.notna(value) else "")
            elif isinstance(value, pd.Timestamp):
                formatted.append(value.date().isoformat())
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return lines
