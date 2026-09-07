#!/usr/bin/env python3
"""Fetch CBOE volatility indices (VIX, VXN, VIX3M, VIX9D, VVIX) and derive term-structure features.

Sources
-------
* CBOE daily history CSVs: https://cdn.cboe.com/api/global/us_indices/daily_prices/<NAME>_History.csv
  (same-evening availability, so the day-T close is known at the close).
* FRED keyless graph CSVs (VXNCLS from 2001, VXVCLS from 2007-12) backfill the years before
  the CBOE files start (VXN 2009-09, VIX3M 2009-09). FRED posts these the next morning, which
  only matters for the backfilled years.

Output: data/processed/cboe_vol_daily.csv with one row per CBOE trading day.
"""

from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

CBOE_INDICES = {
    "VIX": "vix",
    "VXN": "vxn",
    "VIX3M": "vix3m",
    "VIX9D": "vix9d",
    "VVIX": "vvix",
}
FRED_BACKFILL = {  # column -> FRED series id
    "vxn_close": "VXNCLS",
    "vix3m_close": "VXVCLS",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "qqq-research-pipeline (academic project)"})


def download(url: str) -> str:
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_cboe(name: str, prefix: str, raw_dir: Path) -> pd.DataFrame:
    text = download(CBOE_URL.format(name=name))
    (raw_dir / f"{name}_History.csv").write_text(text, encoding="utf-8")
    frame = pd.read_csv(StringIO(text))
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if "close" in frame.columns:
        out = frame[["date", "open", "high", "low", "close"]].copy()
        out.columns = ["date", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"]
    else:  # VVIX file has a single value column
        value_col = [c for c in frame.columns if c != "date"][0]
        out = frame[["date", value_col]].copy()
        out.columns = ["date", prefix]
    logging.info("%s: %d rows, %s -> %s", name, len(out), out["date"].min().date(), out["date"].max().date())
    return out


def fetch_fred_graph(series: str, raw_dir: Path, retries: int = 3) -> pd.Series:
    # FRED's graph endpoint stalls on descriptive user agents (verified: "curl/8.x" answers in
    # under a second, "... academic research pipeline" times out). Keep it minimal.
    headers = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}
    text = None
    for attempt in range(retries):
        try:
            resp = requests.get(FRED_GRAPH_URL.format(series=series), headers=headers, timeout=120)
            resp.raise_for_status()
            text = resp.text
            break
        except requests.RequestException as exc:
            logging.warning("FRED %s attempt %d failed: %s", series, attempt + 1, exc)
    if text is None:
        raise RuntimeError(f"Could not download FRED series {series}")
    (raw_dir / f"fred_{series}.csv").write_text(text, encoding="utf-8")
    frame = pd.read_csv(StringIO(text))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    series_out = frame.dropna(subset=["date"]).set_index("date")["value"]
    logging.info("FRED %s: %d rows from %s", series, series_out.notna().sum(), series_out.first_valid_index().date())
    return series_out


def build(raw_dir: Path, start: str) -> pd.DataFrame:
    frames = [fetch_cboe(name, prefix, raw_dir) for name, prefix in CBOE_INDICES.items()]
    table = frames[0]
    for frame in frames[1:]:
        table = table.merge(frame, on="date", how="outer")
    table = table.sort_values("date").reset_index(drop=True)

    for column, series in FRED_BACKFILL.items():
        fred = fetch_fred_graph(series, raw_dir)
        aligned = fred.reindex(table["date"]).to_numpy()
        missing = table[column].isna()
        table.loc[missing, column] = aligned[missing.to_numpy()]
        table[f"{column}_source"] = "cboe"
        table.loc[missing & table[column].notna(), f"{column}_source"] = "fred"

    # Derived features (levels; log changes are taken in the feature table builder)
    table["vxn_minus_vix"] = table["vxn_close"] - table["vix_close"]
    table["vix_term_ratio"] = table["vix_close"] / table["vix3m_close"]
    table["vix9d_ratio"] = table["vix9d_close"] / table["vix_close"]

    table = table[table["date"] >= pd.Timestamp(start)]
    table = table.dropna(subset=["vix_close"])  # keep CBOE trading days only
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return table.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CBOE volatility indices")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw" / "cboe"
    proc_dir = Path(args.out_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    table = build(raw_dir, args.start)
    path = proc_dir / "cboe_vol_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows, %d columns)", path, len(table), len(table.columns))
    logging.info("Coverage: %s", table.drop(columns=["date"]).notna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
