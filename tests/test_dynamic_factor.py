from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.mixed_frequency_panel import MixedFrequencyPanel
from src.models.dynamic_factor import DynamicFactorBenchmark
from src.nowcast_config import BacktestConfig


def test_dynamic_factor_benchmark_runs_on_synthetic_panel():
    config = BacktestConfig(dfm_em_maxiter=20)
    monthly_index = pd.date_range("2010-01-31", periods=72, freq="ME")
    latent = np.sin(np.arange(72) / 6)
    monthly_panel = pd.DataFrame(
        {
            "Economic_Activity_Index_Discrete_YoY": 100 + 5 * latent,
            "Industry_Real_Growth_YoY": 99 + 4 * latent,
            "Services_Real_Growth_YoY": 98 + 3 * latent,
            "Exchange_Rate_AMD_USD": 480 + latent,
        },
        index=monthly_index,
    )
    quarter_starts = pd.date_range("2010-01-01", periods=24, freq="QS")
    quarterly_target = pd.DataFrame(
        {"Real_GDP_Armenia_YoY": 100 + 4 * np.sin(np.arange(24) / 3)},
        index=pd.date_range("2010-03-31", periods=24, freq="QE"),
    )
    panel = MixedFrequencyPanel(
        monthly_panel=monthly_panel,
        quarterly_target=quarterly_target,
        target_column="Real_GDP_Armenia_YoY",
        official_columns=("Economic_Activity_Index_Discrete_YoY", "Industry_Real_Growth_YoY", "Services_Real_Growth_YoY"),
        fast_columns=("Exchange_Rate_AMD_USD",),
        priority_columns=tuple(monthly_panel.columns),
    )
    benchmark = DynamicFactorBenchmark(panel)

    train_df = pd.DataFrame(
        {"target": quarterly_target["Real_GDP_Armenia_YoY"].iloc[:12].to_list(), "stage": ["Late"] * 12},
        index=quarter_starts[:12],
    )
    test_df = pd.DataFrame(
        {
            "target": [float(quarterly_target["Real_GDP_Armenia_YoY"].iloc[12])],
            "stage": ["Late"],
            "target_quarter": ["2013-Q1"],
        },
        index=[quarter_starts[12]],
    )

    result = benchmark.predict_window(train_df, test_df, [], "target", config)

    assert np.isfinite(result.prediction)
    assert result.metadata is not None
    assert result.metadata["n_monthly_series"] > 0


def test_dynamic_factor_benchmark_uses_fallback_order(monkeypatch):
    config = BacktestConfig(dfm_reduced_monthly_series=2)
    monthly_index = pd.date_range("2015-01-31", periods=36, freq="ME")
    quarterly_target = pd.DataFrame(
        {"Real_GDP_Armenia_YoY": np.linspace(100, 108, 12)},
        index=pd.date_range("2015-03-31", periods=12, freq="QE"),
    )
    monthly_panel = pd.DataFrame(
        {
            "Economic_Activity_Index_Discrete_YoY": np.linspace(100, 110, 36),
            "Industry_Real_Growth_YoY": np.linspace(99, 109, 36),
            "Services_Real_Growth_YoY": np.linspace(98, 108, 36),
        },
        index=monthly_index,
    )
    panel = MixedFrequencyPanel(
        monthly_panel=monthly_panel,
        quarterly_target=quarterly_target,
        target_column="Real_GDP_Armenia_YoY",
        official_columns=tuple(monthly_panel.columns),
        fast_columns=(),
        priority_columns=tuple(monthly_panel.columns),
    )
    benchmark = DynamicFactorBenchmark(panel)

    calls: list[tuple[bool, int]] = []

    class FakeResult:
        def __init__(self, target_month: pd.Timestamp):
            self.model = type("Model", (), {"endog_names": ["m1", "m2", "gdp"]})()
            self.fittedvalues = pd.DataFrame(
                {"m1": [0.0], "m2": [0.0], "gdp": [105.0]},
                index=[target_month],
            )
            self.factors = type(
                "Factors",
                (),
                {
                    "filtered": pd.DataFrame(
                        {"0": [0.1, 0.2]},
                        index=pd.to_datetime(["2017-09-30", target_month]),
                    )
                },
            )()

    def fake_fit_attempt(monthly_endog, window, cfg, idiosyncratic_ar1):
        calls.append((idiosyncratic_ar1, monthly_endog.shape[1]))
        if len(calls) < 3:
            return {"converged": False, "result": None}
        return {"converged": True, "result": FakeResult(window.target_month)}

    monkeypatch.setattr(benchmark, "_fit_attempt", fake_fit_attempt)

    quarter_starts = pd.date_range("2015-01-01", periods=12, freq="QS")
    train_df = pd.DataFrame({"target": np.linspace(100, 107, 8), "stage": ["Late"] * 8}, index=quarter_starts[:8])
    test_df = pd.DataFrame(
        {"target": [108.0], "stage": ["Late"], "target_quarter": ["2017-Q1"]},
        index=[quarter_starts[8]],
    )

    result = benchmark.predict_window(train_df, test_df, [], "target", config)

    assert result.metadata["fallback_used"] == "reduced_white_noise"
    assert result.feature_count == 2
    assert calls == [(True, 3), (False, 3), (False, 2)]
