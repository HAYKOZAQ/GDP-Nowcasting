"""
Fetch curated Google Trends libraries for the Armenia nowcasting pipeline.

Key improvements over the older one-off scripts:
1. Uses a repeated anchor term in every batch so pytrends batches can be stitched
   onto a comparable scale.
2. Keeps each output file on a consistent geography rather than mixing geos.
3. Uses Armenia-specific English, Armenian, Russian, and local-platform terms.

Outputs:
  - data/processed/google_trends_armenia_monthly.csv
  - data/processed/google_trends_armenia_quarterly.csv
  - data/processed/google_trends_armenian_monthly.csv
  - data/processed/google_trends_armenian_quarterly.csv
  - data/processed/google_trends_shock_monthly.csv
  - data/processed/google_trends_shock_quarterly.csv
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "processed"
END_DATE = (pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TIMEFRAME = f"2018-01-01 {END_DATE}"


DATASETS = {
    "google_trends_armenia": {
        "hl": "ru-RU",
        "tz": 240,
        "geo": "RU",
        "anchor": "Армения",
        "min_series": 6,
        "batches": [
            ["Ереван", "квартира Ереван", "работа Армения", "Армения недвижимость"],
            ["виза Армения", "переезд Армения", "банк Армения", "цены Ереван"],
            ["аренда Ереван", "ВНЖ Армения", "работа в Ереване", "гражданство Армения"],
        ],
    },
    "google_trends_armenian": {
        "hl": "hy-AM",
        "tz": 240,
        "geo": "AM",
        "anchor": "Yerevan",
        "min_series": 8,
        "batches": [
            ["Idram", "Telcell", "Rate.am", "MoneyGram"],
            ["Wildberries", "List.am", "Auto.am", "Estate.am"],
            ["Booking.com", "Aviasales", "Zvartnots", "Staff.am"],
            ["վարկ", "բնակարան", "աշխատանք", "փոխարժեք"],
            ["փոխանցում", "շինարարություն", "զբոսաշրջություն", "iPhone"],
        ],
    },
    "google_trends_shock": {
        "hl": "ru-RU",
        "tz": 240,
        "geo": "RU",
        "anchor": "Армения",
        "min_series": 6,
        "batches": [
            ["квартира Ереван", "снять квартиру Ереван", "купить квартиру Ереван", "аренда Ереван"],
            ["переезд Армения", "переехать Армения", "виза Армения", "ВНЖ Армения"],
            ["открыть счет Армения", "гражданство Армения", "регистрация Ереван", "загранпаспорт Армения"],
            ["работа Армения", "работа в Ереване", "ИТ Армения", "бизнес Армения"],
        ],
    },
}


def build_client(hl: str, tz: int) -> TrendReq:
    return TrendReq(hl=hl, tz=tz, timeout=(10, 30), retries=3, backoff_factor=1)


def fetch_batch(client: TrendReq, keywords: list[str], geo: str) -> pd.DataFrame:
    client.build_payload(keywords, timeframe=TIMEFRAME, geo=geo, cat=0)
    df = client.interest_over_time()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


def compute_scale(reference_anchor: pd.Series, batch_anchor: pd.Series) -> float:
    overlap = pd.concat(
        [
            reference_anchor.rename("ref"),
            batch_anchor.rename("batch"),
        ],
        axis=1,
    ).dropna()
    overlap = overlap[(overlap["ref"] > 0) & (overlap["batch"] > 0)]
    if overlap.empty:
        return 1.0
    ratio = overlap["ref"] / overlap["batch"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 1.0
    return float(ratio.median())


def fetch_dataset(name: str, spec: dict[str, object], pause_seconds: float) -> pd.DataFrame:
    client = build_client(str(spec["hl"]), int(spec["tz"]))
    anchor = str(spec["anchor"])
    geo = str(spec["geo"])
    batches: list[list[str]] = list(spec["batches"])

    print("=" * 72)
    print(f"Fetching {name} | geo={geo or 'GLOBAL'} | anchor={anchor}")
    print("=" * 72)

    anchor_reference: pd.Series | None = None
    collected: list[pd.DataFrame] = []

    for idx, batch_terms in enumerate(batches, start=1):
        keywords = [anchor] + [term for term in batch_terms if term != anchor][:4]
        print(f"Batch {idx}/{len(batches)}: {keywords}")
        try:
            raw = fetch_batch(client, keywords, geo)
        except Exception as exc:  # pragma: no cover - depends on live Google response
            print(f"  failed: {exc}")
            time.sleep(10)
            continue
        if raw.empty:
            print("  empty response")
            time.sleep(6)
            continue

        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        anchor_series = pd.to_numeric(raw[anchor], errors="coerce")
        scaled = raw.copy()
        if anchor_reference is None:
            anchor_reference = anchor_series
        else:
            scale = compute_scale(anchor_reference, anchor_series)
            non_anchor = [col for col in scaled.columns if col != anchor]
            scaled.loc[:, non_anchor] = scaled[non_anchor] * scale
            print(f"  scale factor: {scale:.3f}")

        collected.append(scaled.drop(columns=[anchor], errors="ignore"))
        print(f"  rows: {len(scaled)}")
        time.sleep(pause_seconds)

    if not collected:
        raise RuntimeError(f"No Google Trends data fetched for {name}")

    combined = pd.concat(collected, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(axis=1, how="all")
    nonzero_cols = [col for col in combined.columns if pd.to_numeric(combined[col], errors="coerce").fillna(0).abs().sum() > 0]
    combined = combined[nonzero_cols]
    combined.index.name = "date"
    return combined


def save_dataset(name: str, monthly: pd.DataFrame, min_series: int) -> None:
    if len(monthly.columns) < min_series:
        raise RuntimeError(f"{name} fetched only {len(monthly.columns)} usable series; expected at least {min_series}")

    quarterly = monthly.resample("QS").mean()
    monthly_path = PROC_DIR / f"{name}_monthly.csv"
    quarterly_path = PROC_DIR / f"{name}_quarterly.csv"
    if monthly_path.exists():
        shutil.copy2(monthly_path, monthly_path.with_suffix(monthly_path.suffix + ".bak"))
    if quarterly_path.exists():
        shutil.copy2(quarterly_path, quarterly_path.with_suffix(quarterly_path.suffix + ".bak"))
    monthly.to_csv(monthly_path)
    quarterly.to_csv(quarterly_path)
    print(f"Saved {monthly_path}")
    print(f"Saved {quarterly_path}")
    print(f"Usable series: {len(monthly.columns)}")
    if not monthly.empty:
        top = monthly.mean().sort_values(ascending=False).head(10)
        print("Top average terms:")
        print(top.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Fetch only one curated Google Trends library.")
    parser.add_argument("--pause-seconds", type=float, default=5.0, help="Sleep between successful batch requests.")
    parser.add_argument("--cooldown-seconds", type=float, default=0.0, help="Sleep between datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    items = [(args.dataset, DATASETS[args.dataset])] if args.dataset else list(DATASETS.items())
    for idx, (name, spec) in enumerate(items):
        monthly = fetch_dataset(name, spec, args.pause_seconds)
        save_dataset(name, monthly, int(spec["min_series"]))
        if args.cooldown_seconds > 0 and idx < len(items) - 1:
            print(f"Cooling down for {args.cooldown_seconds:.0f} seconds")
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
