#!/usr/bin/env python3
"""Fetch the daily US Economic Policy Uncertainty (EPU) and Equity Market Uncertainty (EMU) indices.

Source: Baker, Bloom & Davis, https://www.policyuncertainty.com (CC BY 4.0).
Both series are calendar-day (7-day week) news-count indices from 1985-01-01.

Availability: the value for day T is posted the next morning (FRED mirrors USEPUINDXD /
WLEMUINDXD at ~08:00 CT on T+1), and only the trailing ~30 days are revised. The feature
table builder therefore attaches the value dated T-1 to trading day T.

Output: data/processed/epu_daily.csv with columns date, epu_daily, emu_daily (calendar days).
"""

from __future__ import annotations

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SOURCES = {
    "epu_daily": ("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv", "daily_policy_index"),
    "emu_daily": ("https://www.policyuncertainty.com/media/All_Daily_Equity_Data.csv", "daily_equity_index"),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (academic research pipeline)"})


def fetch(column: str, url: str, value_col: str, raw_dir: Path) -> pd.DataFrame:
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    (raw_dir / Path(url).name).write_text(resp.text, encoding="utf-8")
    frame = pd.read_csv(StringIO(resp.text))
    frame = frame.dropna(subset=["year", "month", "day"])
    frame["date"] = pd.to_datetime(
        dict(year=frame["year"].astype(int), month=frame["month"].astype(int), day=frame["day"].astype(int)),
        errors="coerce",
    )
    out = frame[["date", value_col]].rename(columns={value_col: column})
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    logging.info("%s: %d rows, %s -> %s", column, len(out), out["date"].min().date(), out["date"].max().date())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily EPU and EMU indices")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw" / "epu"
    proc_dir = Path(args.out_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    table = None
    for column, (url, value_col) in SOURCES.items():
        frame = fetch(column, url, value_col, raw_dir)
        table = frame if table is None else table.merge(frame, on="date", how="outer")

    table = table[table["date"] >= pd.Timestamp(args.start)].sort_values("date").reset_index(drop=True)
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    path = proc_dir / "epu_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows)", path, len(table))


if __name__ == "__main__":
    main()
