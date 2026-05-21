from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.walkforward import run_backtest
from src.forecasting.recursive_quarterly import run_future_quarterly_forecast
from src.nowcast_config import BacktestConfig, build_paths
from src.reporting.backtest_report import write_report_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Armenia GDP nowcasting backtest and generate thesis-ready artifacts."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing data/, src/, and results/ directories.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run the backtest only and do not regenerate report figures/markdown.",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Regenerate report artifacts from existing backtest CSV outputs without rerunning the backtest.",
    )
    parser.add_argument(
        "--future-forecast",
        action="store_true",
        help="Generate a recursive quarterly forecast from the latest observed quarter through the next three quarters.",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Generate the forward quarterly forecast without rerunning the backtest or rebuilding report figures.",
    )
    return parser


def run_pipeline(base_dir: Path, skip_report: bool = False) -> tuple[object, object]:
    predictions, summary = run_backtest(base_dir, BacktestConfig())
    if not skip_report:
        write_report_artifacts(base_dir, predictions, summary)
    return predictions, summary


def regenerate_reports(base_dir: Path) -> None:
    paths = build_paths(base_dir)
    predictions_path = paths["backtests"] / "backtest_predictions.csv"
    summary_path = paths["backtests"] / "backtest_summary.csv"
    if not predictions_path.exists() or not summary_path.exists():
        missing = [str(path) for path in (predictions_path, summary_path) if not path.exists()]
        raise FileNotFoundError(
            "Cannot regenerate reports because required backtest files are missing: "
            + ", ".join(missing)
        )

    import pandas as pd

    predictions = pd.read_csv(predictions_path, parse_dates=["prediction_date"])
    summary = pd.read_csv(summary_path)
    write_report_artifacts(base_dir, predictions, summary)


def main() -> None:
    args = build_parser().parse_args()
    base_dir = args.base_dir.resolve()

    if args.forecast_only:
        run_future_quarterly_forecast(base_dir)
        return

    if args.reports_only:
        regenerate_reports(base_dir)
        if args.future_forecast:
            run_future_quarterly_forecast(base_dir)
        return

    run_pipeline(base_dir, skip_report=args.skip_report)
    if args.future_forecast:
        run_future_quarterly_forecast(base_dir)


if __name__ == "__main__":
    main()
