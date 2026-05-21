from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.nowcast_dataset import _merge_monthly_extensions


def test_merge_monthly_extensions_adds_new_rows_and_columns(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    base = pd.DataFrame(
        {
            "Money_Supply_M2_Mln_AMD": [100.0],
            "Existing": [1.0],
        },
        index=pd.to_datetime(["2025-06-01"]),
    )

    extension = pd.DataFrame(
        {
            "Money_Supply_M2_Mln_AMD": [110.0, 120.0],
            "Remittance_Net_Mln_AMD": [10.0, 12.0],
        },
        index=pd.to_datetime(["2025-06-01", "2025-07-01"]),
    )
    extension.to_csv(processed_dir / "cba_nowcast_extension_monthly.csv", index_label="date")

    merged = _merge_monthly_extensions(base, processed_dir)

    assert pd.Timestamp("2025-07-01") in merged.index
    assert merged.loc[pd.Timestamp("2025-06-01"), "Money_Supply_M2_Mln_AMD"] == 100.0
    assert merged.loc[pd.Timestamp("2025-07-01"), "Money_Supply_M2_Mln_AMD"] == 120.0
    assert merged.loc[pd.Timestamp("2025-07-01"), "Remittance_Net_Mln_AMD"] == 12.0
