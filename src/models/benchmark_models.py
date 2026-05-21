from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.mixed_frequency_panel import MixedFrequencyPanel
from src.nowcast_config import BacktestConfig


def _safe_tscv(n_obs: int, max_splits: int = 3) -> TimeSeriesSplit | None:
    if n_obs < 12:
        return None
    n_splits = min(max_splits, max(2, n_obs // 8))
    return TimeSeriesSplit(n_splits=n_splits)


def _filter_feature_list(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target: pd.Series,
    alt_cap: int,
    method: str = "mutual_info",
) -> list[str]:
    """Select features, using mutual information to rank alternative high-freq signals."""
    usable_cols = [col for col in feature_cols if col in train_df.columns and train_df[col].notna().any()]
    base_cols = [col for col in usable_cols if not col.startswith(("FAST_GTG_", "FAST_GTL_", "FAST_GTS_", "FAST_WIKI_"))]
    alt_cols = [col for col in usable_cols if col not in base_cols]
    if not alt_cols:
        return usable_cols

    scores: list[tuple[str, float]] = []
    y = target.astype(float)
    for col in alt_cols:
        s = train_df[col].astype(float)
        valid = s.notna() & y.notna()
        if valid.sum() < 12:
            continue
        if float(s[valid].std()) == 0.0:
            continue
        if method == "mutual_info":
            try:
                X_col = s[valid].values.reshape(-1, 1)
                mi_val = mutual_info_regression(X_col, y[valid].values, random_state=42)[0]
                scores.append((col, mi_val))
                continue
            except Exception:
                pass
        # Fallback: Pearson correlation
        corr = np.corrcoef(s[valid], y[valid])[0, 1]
        if np.isnan(corr):
            continue
        scores.append((col, abs(corr)))

    top_alt = [name for name, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:alt_cap]]
    return [col for col in base_cols + top_alt if col in usable_cols]


@dataclass
class PredictionResult:
    prediction: float
    feature_count: int
    metadata: dict[str, object] | None = None
    artifacts: dict[str, pd.DataFrame] | None = None


class BenchmarkModel:
    name: str

    def predict_window(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        config: BacktestConfig,
    ) -> PredictionResult:
        raise NotImplementedError


class AutoregressiveBenchmark(BenchmarkModel):
    name = "AR"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col.startswith("AR_") or col.startswith("CURR_Dummy_")]
        cols = [col for col in cols if col in train_df.columns and train_df[col].notna().any()]
        model = LinearRegression()
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(train_df[cols])
        X_te = imputer.transform(test_df[cols])
        model.fit(X_tr, train_df[target_col])
        pred = float(model.predict(X_te)[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class RidgeBenchmark(BenchmarkModel):
    name = "Bridge"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=_safe_tscv(len(train_df)))),
            ]
        )
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class ElasticNetBenchmark(BenchmarkModel):
    name = "ElasticNet"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        cv = _safe_tscv(len(train_df))
        if cv is None:
            ridge = RidgeBenchmark()
            return ridge.predict_window(train_df, test_df, cols, target_col, config)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9],
                        alphas=30,
                        max_iter=5000,
                        cv=cv,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class MIDAsBenchmark(BenchmarkModel):
    """
    Mixed Data Sampling benchmark.

    Selects the most informative monthly-frequency features using mutual information,
    then regresses on their Almon-weighted within-quarter aggregates alongside quarterly
    predictors via RidgeCV. This replaces FactorAugmentedPCA, which catastrophically
    destroys ragged-edge timing by compressing data with PCA before forecasting.
    """

    name = "MIDAS"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        # Select ALMON/LAST/MEAN variants of monthly series (already aggregated correctly)
        monthly_derived = [
            col for col in feature_cols
            if col.startswith(("FAST_", "OFF_")) and col.endswith(("_ALMON", "_LAST", "_MEAN"))
        ]
        base_cols = [
            col for col in feature_cols
            if col not in monthly_derived and col in train_df.columns and train_df[col].notna().any()
        ]
        # Rank monthly features by mutual information — key improvement over PCA
        selected_monthly = _filter_feature_list(
            train_df, monthly_derived, train_df[target_col],
            config.alt_feature_cap + config.factor_feature_cap,
            method="mutual_info",
        )
        cols = [c for c in base_cols + selected_monthly if c in train_df.columns and train_df[c].notna().any()]
        if not cols:
            return PredictionResult(prediction=np.nan, feature_count=0)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=_safe_tscv(len(train_df)))),
        ])
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class LightGBMBenchmark(BenchmarkModel):
    """LightGBM regressor — leaf-wise growth with native missing-value handling."""

    name = "LightGBM"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            return PredictionResult(prediction=np.nan, feature_count=0)

        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        if not cols:
            return PredictionResult(prediction=np.nan, feature_count=0)

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(train_df[cols])
        X_te = imputer.transform(test_df[cols])

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=config.random_state,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_tr, train_df[target_col])
        pred = float(model.predict(X_te)[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class RandomForestBenchmark(BenchmarkModel):
    name = "RandomForest"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(train_df[cols])
        X_te = imputer.transform(test_df[cols])
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=config.random_state,
            n_jobs=1,
        )
        model.fit(X_tr, train_df[target_col])
        pred = float(model.predict(X_te)[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class GradientBoostingBenchmark(BenchmarkModel):
    name = "GradientBoosting"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(train_df[cols])
        X_te = imputer.transform(test_df[cols])
        model = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=4,
            max_iter=300,
            min_samples_leaf=3,
            l2_regularization=1.0,
            random_state=config.random_state,
        )
        model.fit(X_tr, train_df[target_col])
        pred = float(model.predict(X_te)[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class HuberBenchmark(BenchmarkModel):
    name = "Huber"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        cols = [col for col in feature_cols if col in train_df.columns]
        cols = _filter_feature_list(train_df, cols, train_df[target_col], config.alt_feature_cap, config.feature_selection_method)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("huber", HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=2000)),
            ]
        )
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class EarlyShockBridgeBenchmark(BenchmarkModel):
    name = "EarlyShockBridge"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        if str(test_df["stage"].iloc[0]) != "Early":
            return PredictionResult(prediction=np.nan, feature_count=0)
        cols = [col for col in feature_cols if col in train_df.columns and train_df[col].notna().any()]
        if not cols:
            return PredictionResult(prediction=np.nan, feature_count=0)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("huber", HuberRegressor(epsilon=1.2, alpha=0.0005, max_iter=3000)),
            ]
        )
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


class ShadowBenchmark(BenchmarkModel):
    name = "Shadow"

    def predict_window(self, train_df, test_df, feature_cols, target_col, config):
        preferred_cols = [
            "AR_LAG1",
            "AR_LAG4",
            "Q_Real_GDP_Russia_YoY",
            "Q_CPI_YoY",
            "OFF_Economic_Activity_Index_Discrete_YoY_LAST",
            "FAST_Exchange_Rate_AMD_USD_LAST",
            "FAST_Exchange_Rate_AMD_RUB_LAST",
            "FAST_Brent_Oil_Price_USD_bbl_LAST",
            "FAST_Copper_Price_USD_mt_LAST",
            "FAST_FIN_STRESS_PROXY_LAST",
        ]
        cols = [col for col in preferred_cols if col in train_df.columns and train_df[col].notna().any()][:5]
        if len(cols) < 2:
            cols = [col for col in feature_cols if col in train_df.columns and train_df[col].notna().any()][:5]
        if not cols:
            return PredictionResult(prediction=np.nan, feature_count=0)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=_safe_tscv(len(train_df)))),
            ]
        )
        model.fit(train_df[cols], train_df[target_col])
        pred = float(model.predict(test_df[cols])[0])
        return PredictionResult(prediction=pred, feature_count=len(cols))


def make_benchmark_models(dfm_panel: MixedFrequencyPanel | None = None) -> list[BenchmarkModel]:
    models: list[BenchmarkModel] = [
        AutoregressiveBenchmark(),
        RidgeBenchmark(),
        MIDAsBenchmark(),          # Replaces FactorAugmentedPCA
        ElasticNetBenchmark(),
        EarlyShockBridgeBenchmark(),
        ShadowBenchmark(),
        HuberBenchmark(),
        RandomForestBenchmark(),
        GradientBoostingBenchmark(),
        LightGBMBenchmark(),       # New: leaf-wise gradient boosting
    ]
    if dfm_panel is not None:
        from src.models.dynamic_factor import DynamicFactorBenchmark

        models.append(DynamicFactorBenchmark(dfm_panel))
    return models


def combine_simple_ensemble(predictions: dict[str, float]) -> float | None:
    """Simple ensemble: ElasticNet + RandomForest + GradientBoosting + LightGBM (Huber excluded — unstable)."""
    members = [predictions.get(name) for name in ("ElasticNet", "RandomForest", "GradientBoosting", "LightGBM")]
    members = [value for value in members if value is not None and not np.isnan(value)]
    if not members:
        return None
    return float(np.mean(members))
