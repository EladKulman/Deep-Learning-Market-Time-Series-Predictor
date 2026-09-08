#!/usr/bin/env python3
"""Fetch FRED macro series with publication-aware handling.

What changed versus the first version
-------------------------------------
* CPI is no longer forward-filled by reference month. July CPI is dated 07-01 on FRED but
  is published around the 12th of August. The new columns are built from ALFRED vintages
  (``get_series_all_releases``): ``cpi`` and ``cpi_yoy`` on day T are the values that had
  actually been published on or before T, and ``cpi_release_day`` flags the publication
  dates so they can be used as a known-future covariate.
* Daily rates are stored on reference dates. The builder maps H.15 observations to the
  first NYSE close after next-business-day 16:15 ET publication. T10Y2Y and T5YIE use
  direct Treasury data from June 21, 2019 and are available after that day's close.
* Added: DFII10 (10y real yield), T5YIE (5y breakeven). The ICE BofA high-yield spread
  (BAMLH0A0HYM2) was dropped: since April 2026 FRED serves only its last three years, so
  the HYG ETF return in cross_asset_daily.csv (from 2007) is the credit signal instead.

Requires FRED_API_KEY in .env (free: https://fred.stlouisfed.org/docs/api/api_key.html).
Output: data/processed/fred_macro_daily.csv, one row per calendar day.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DAILY_SERIES = {
    "DFF": "dff",
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "T10Y2Y": "t10y2y",
    "DFII10": "dfii10",
    "T5YIE": "t5yie",
    "VIXCLS": "vix_fred",
}
CPI_SERIES = "CPIAUCSL"


def get_fred() -> Fred:
    load_dotenv()
    key = os.environ.get("FRED_API_KEY", "")
    if not key or key.startswith("your_"):
        raise SystemExit("Set FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api/api_key.html).")
    return Fred(api_key=key)


def fetch_daily(fred: Fred | None, start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    frames = {}
    for series_id, column in DAILY_SERIES.items():
        logging.info("Fetching %s", series_id)
        if fred is None:
            saved = pd.read_csv(raw_dir / f"{series_id}.csv", parse_dates=[0], index_col=0)
            values = saved.iloc[:, 0].loc[start:end]
        else:
            values = fred.get_series(series_id, observation_start=start, observation_end=end)
        values.index.name = "date"
        if fred is not None:
            values.to_csv(raw_dir / f"{series_id}.csv", header=[column])
        frames[column] = values
    return pd.DataFrame(frames)


def fetch_cpi_vintages(fred: Fred | None, raw_dir: Path) -> pd.DataFrame:
    """Return one row per CPI release: realtime_start, latest reference month, level, yoy.

    yoy is computed within the vintage that was current on the release date, so a revision
    to the year-ago month (annual seasonal-factor updates) is reflected only from the date
    it was actually published.
    """
    logging.info("Fetching all CPI releases from ALFRED")
    releases = (pd.read_csv(raw_dir / f"{CPI_SERIES}_all_releases.csv", parse_dates=["date", "realtime_start"])
                if fred is None else fred.get_series_all_releases(CPI_SERIES))
    releases = releases.dropna(subset=["value"]).sort_values(["realtime_start", "date"])
    if fred is not None:
        releases.to_csv(raw_dir / f"{CPI_SERIES}_all_releases.csv", index=False)

    current: dict[pd.Timestamp, float] = {}
    rows = []
    for release_date, group in releases.groupby("realtime_start", sort=True):
        for _, row in group.iterrows():
            current[pd.Timestamp(row["date"])] = float(row["value"])
        latest_month = max(current)
        year_ago = latest_month - pd.DateOffset(years=1)
        level = current[latest_month]
        yoy = level / current[year_ago] - 1 if year_ago in current else float("nan")
        rows.append({"realtime_start": pd.Timestamp(release_date), "cpi_ref_month": latest_month, "cpi": level, "cpi_yoy": yoy})
    vintages = pd.DataFrame(rows)
    # Revisions affect the known CPI level from their publication date, even when they
    # do not advance the reference month. Only the monthly-print flag excludes them.
    vintages["is_monthly_print"] = vintages["cpi_ref_month"] > vintages["cpi_ref_month"].shift(1).fillna(pd.Timestamp("1900-01-01"))
    logging.info("CPI releases: %d (%s -> %s)", len(vintages), vintages["realtime_start"].min().date(), vintages["realtime_start"].max().date())
    return vintages.reset_index(drop=True)


def build(fred: Fred, start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    daily = fetch_daily(fred, start, end, raw_dir)
    table = days.merge(daily, left_on="date", right_index=True, how="left")

    vintages = fetch_cpi_vintages(fred, raw_dir)
    table = pd.merge_asof(
        table.sort_values("date"),
        vintages.sort_values("realtime_start"),
        left_on="date",
        right_on="realtime_start",
        direction="backward",
    )
    table["cpi_release_day"] = table["date"].isin(set(vintages.loc[vintages["is_monthly_print"], "realtime_start"])).astype(int)
    table["cpi_ref_month"] = table["cpi_ref_month"].dt.strftime("%Y-%m")
    table = table.drop(columns=["realtime_start", "is_monthly_print"])
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FRED macro data with ALFRED-vintaged CPI")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--cached", action="store_true", help="Rebuild from existing raw series/vintages without API access")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw" / "fred"
    proc_dir = Path(args.out_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    table = build(None if args.cached else get_fred(), args.start, end, raw_dir)
    path = proc_dir / "fred_macro_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows, %d columns)", path, len(table), len(table.columns))
    logging.info("Coverage: %s", table.drop(columns=["date", "cpi_ref_month"]).notna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
