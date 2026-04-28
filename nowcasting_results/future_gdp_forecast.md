# Armenia GDP Forecast Through 2026 Q4

Last observed quarter used for training: `2026-01-01`.
Selected recursive quarterly model: `Ridge`.

This is a forward quarterly forecast, not a same-quarter nowcast. For `2026 Q2-Q4`, the model uses the historical quarterly panel and rolls predictions forward recursively.

## Model Selection

| model | n_obs | mae | mape |
| --- | --- | --- | --- |
| Ridge | 153 | 3.813 | 3.716 |
| RandomForest | 153 | 3.978 | 3.826 |
| StackingForecast | 153 | 4.053 | 3.960 |
| ElasticNet | 153 | 4.105 | 4.020 |

## Forecasts

| target_quarter | horizon | forecast | interval_lo_50 | interval_hi_50 | interval_lo_90 | interval_hi_90 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-Q2 | 1 | 105.444 | 103.205 | 107.682 | 97.616 | 113.272 |
| 2026-Q3 | 2 | 104.446 | 102.008 | 106.885 | 95.616 | 113.276 |
| 2026-Q4 | 3 | 104.206 | 101.569 | 106.843 | 93.716 | 114.696 |

The intervals are empirical error bands derived from historical recursive 3-quarter-ahead forecast errors for the selected model.
