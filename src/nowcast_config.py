from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


def _load_yaml_defaults() -> dict:
    root_dir = Path(__file__).resolve().parent.parent
    yaml_path = root_dir / "config.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "modeling" in data:
                    return data["modeling"]
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
    return {}


_YAML_DEFAULTS = _load_yaml_defaults()


@dataclass(frozen=True)
class BacktestConfig:
    min_train_quarters: int = _YAML_DEFAULTS.get("min_train_quarters", 40)
    target_column: str = _YAML_DEFAULTS.get("target_column", "Real_GDP_Armenia_YoY")
    workbook_name: str = _YAML_DEFAULTS.get("workbook_name", "Translated_Cleaned_Nowcasting_Data.xlsx")
    backtest_dir_name: str = _YAML_DEFAULTS.get("backtest_dir_name", "backtests")
    random_state: int = _YAML_DEFAULTS.get("random_state", 42)
    alt_feature_cap: int = _YAML_DEFAULTS.get("alt_feature_cap", 12)
    factor_feature_cap: int = _YAML_DEFAULTS.get("factor_feature_cap", 8)
    dfm_min_monthly_observations: int = _YAML_DEFAULTS.get("dfm_min_monthly_observations", 24)
    dfm_min_monthly_coverage: float = _YAML_DEFAULTS.get("dfm_min_monthly_coverage", 0.55)
    dfm_history_months: int = _YAML_DEFAULTS.get("dfm_history_months", 96)
    dfm_max_monthly_series: int = _YAML_DEFAULTS.get("dfm_max_monthly_series", 20)
    dfm_reduced_monthly_series: int = _YAML_DEFAULTS.get("dfm_reduced_monthly_series", 12)
    # DFM: try 2 factors first, fall back to 1 if non-convergent
    dfm_factors: int = _YAML_DEFAULTS.get("dfm_factors", 2)
    dfm_factor_orders: int = _YAML_DEFAULTS.get("dfm_factor_orders", 2)
    dfm_em_maxiter: int = _YAML_DEFAULTS.get("dfm_em_maxiter", 100)
    dfm_em_tolerance: float = _YAML_DEFAULTS.get("dfm_em_tolerance", 1e-4)
    # Feature selection: "mutual_info" or "correlation"
    feature_selection_method: str = _YAML_DEFAULTS.get("feature_selection_method", "mutual_info")
    # Optuna hyperparameter tuning (for RF and LightGBM)
    optuna_n_trials: int = _YAML_DEFAULTS.get("optuna_n_trials", 20)
    optuna_timeout_seconds: int = _YAML_DEFAULTS.get("optuna_timeout_seconds", 45)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path | str) -> BacktestConfig:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            modeling = data.get("modeling", {}) if data else {}
        return cls(**modeling)


def build_paths(base_dir: Path) -> dict[str, Path]:
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    return {
        "base": base_dir,
        "raw_data": data_dir / "raw",
        "processed_data": data_dir / "processed",
        "results": results_dir,
        "backtests": results_dir / BacktestConfig().backtest_dir_name,
        "figures": results_dir / "figures",
    }
