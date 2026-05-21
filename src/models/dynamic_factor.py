from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.data.mixed_frequency_panel import DFMWindowData, MixedFrequencyPanel, get_stage_panel
from src.models.benchmark_models import PredictionResult
from src.nowcast_config import BacktestConfig


class DynamicFactorBenchmark:
    name = "DFM"

    def __init__(self, panel: MixedFrequencyPanel) -> None:
        self.panel = panel

    def predict_window(self, train_df, test_df, feature_cols, target_col, config: BacktestConfig) -> PredictionResult:
        stage = str(test_df["stage"].iloc[0])
        target_quarter = pd.Timestamp(test_df.index[0])
        train_end = pd.Timestamp(train_df.index[-1])
        window = get_stage_panel(self.panel, target_quarter, stage, train_end, config)

        if window.monthly_endog.shape[1] == 0 or window.quarterly_endog.empty:
            return PredictionResult(
                prediction=np.nan,
                feature_count=window.monthly_endog.shape[1],
                metadata={
                    "converged": False,
                    "n_monthly_series": window.monthly_endog.shape[1],
                    "n_factors": config.dfm_factors,
                    "cutoff_month": window.cutoff_month.isoformat(),
                    "fallback_used": "insufficient_panel",
                },
            )

        # Deterministic fallback order:
        # 1) full AR(1), 2) full white-noise idiosyncratic, 3) reduced white-noise panel.
        attempts: list[tuple[str, bool, int, int]] = [
            ("full_ar1", True, config.dfm_max_monthly_series, config.dfm_factors),
            ("full_white_noise", False, config.dfm_max_monthly_series, config.dfm_factors),
            ("reduced_white_noise", False, config.dfm_reduced_monthly_series, 1),
        ]

        last_metadata: dict[str, object] | None = None
        for attempt_name, idiosyncratic_ar1, series_cap, n_factors in attempts:
            subset = _subset_monthly_panel(window, series_cap)
            if subset.shape[1] == 0:
                continue
            if _fit_attempt_supports_n_factors(self._fit_attempt):
                fit_result = self._fit_attempt(
                    subset,
                    window,
                    config,
                    idiosyncratic_ar1,
                    n_factors=n_factors,
                )
            else:
                # Backward compatibility for monkeypatched test doubles that still
                # expose the old 4-argument signature.
                fit_result = self._fit_attempt(subset, window, config, idiosyncratic_ar1)
            metadata = {
                "converged": fit_result["converged"],
                "n_monthly_series": subset.shape[1],
                "n_factors": n_factors,
                "cutoff_month": window.cutoff_month.isoformat(),
                "fallback_used": attempt_name,
            }
            last_metadata = metadata
            result_obj = fit_result.get("result")
            if result_obj is None:
                continue
            prediction = _extract_target_prediction(result_obj, window)
            if not np.isfinite(prediction):
                continue
            return PredictionResult(
                prediction=float(prediction),
                feature_count=subset.shape[1],
                metadata=metadata,
                artifacts={
                    "factor_states": _factor_states_frame(
                        result_obj,
                        window,
                        stage,
                        attempt_name,
                    )
                },
            )

        if last_metadata is None:
            last_metadata = {
                "converged": False,
                "n_monthly_series": 0,
                "n_factors": config.dfm_factors,
                "cutoff_month": window.cutoff_month.isoformat(),
                "fallback_used": "all_failed",
            }
        return PredictionResult(
            prediction=np.nan,
            feature_count=window.monthly_endog.shape[1],
            metadata=last_metadata,
        )

    def _fit_attempt(
        self,
        monthly_endog: pd.DataFrame,
        window: DFMWindowData,
        config: BacktestConfig,
        idiosyncratic_ar1: bool,
        n_factors: int = 1,
    ) -> dict[str, object]:
        monthly_endog = monthly_endog.astype(float).sort_index().asfreq("ME")
        quarterly_endog = window.quarterly_endog.astype(float).sort_index().asfreq("QE-DEC")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = DynamicFactorMQ(
                    monthly_endog,
                    endog_quarterly=quarterly_endog,
                    factors=n_factors,
                    factor_orders=config.dfm_factor_orders,
                    idiosyncratic_ar1=idiosyncratic_ar1,
                    standardize=True,
                )
                result = model.fit(
                    disp=False,
                    maxiter=config.dfm_em_maxiter,
                    tolerance=config.dfm_em_tolerance,
                )
            except Exception:
                return {"converged": False, "result": None}

        converged = _is_converged(result, caught, config.dfm_em_maxiter)
        return {"converged": converged, "result": result}


def _subset_monthly_panel(window: DFMWindowData, series_cap: int) -> pd.DataFrame:
    cols = list(window.selected_columns)[:series_cap]
    return window.monthly_endog[cols].copy()


def _fit_attempt_supports_n_factors(fit_attempt: object) -> bool:
    try:
        params = inspect.signature(fit_attempt).parameters
    except (TypeError, ValueError):
        return True
    return "n_factors" in params


def _extract_target_prediction(result, window: DFMWindowData) -> float:
    target_period = pd.Period(window.target_month, freq="M")
    if hasattr(result, "predict"):
        predicted = result.predict(start=target_period, end=target_period)
    else:
        predicted = result.fittedvalues.copy()
    predicted.index = _to_timestamp_index(predicted.index)
    if window.target_month not in predicted.index:
        return np.nan
    target_col = result.model.endog_names[-1]
    return float(predicted.loc[window.target_month, target_col])


def _factor_states_frame(result, window: DFMWindowData, stage: str, fallback_used: str) -> pd.DataFrame:
    factors = result.factors.filtered.copy()
    if isinstance(factors.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        factors.index = _to_timestamp_index(factors.index)
    factors.columns = [f"factor_{idx + 1}" for idx in range(factors.shape[1])]
    if isinstance(factors.index, pd.DatetimeIndex):
        factors = factors.reset_index().rename(columns={"index": "state_date"})
    else:
        factors = factors.reset_index(drop=True)
        factors.insert(0, "state_date", pd.NaT)
    factors.insert(1, "state_step", range(len(factors)))
    factors["prediction_date"] = window.target_quarter
    factors["target_quarter"] = f"{window.target_quarter.year}-Q{window.target_quarter.quarter}"
    factors["stage"] = stage
    factors["cutoff_month"] = window.cutoff_month
    factors["fallback_used"] = fallback_used
    return factors


def _to_timestamp_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp(how="end").normalize()
    return pd.to_datetime(index).normalize()


def _is_converged(result, caught_warnings: list[warnings.WarningMessage], maxiter: int) -> bool:
    warning_text = " ".join(str(item.message) for item in caught_warnings).lower()
    if "without achieving convergence" in warning_text or "maximum number of iterations" in warning_text:
        return False
    params = np.asarray(result.params)
    return bool(np.isfinite(params).all())
