#!/usr/bin/env python3
"""Build known-in-advance calendar covariates for every trading day.

These are the "known-future" covariates Chronos-2 can exploit: every value is fixed on a
published schedule before the day arrives, so it is legitimate to feed them for the
forecast horizon as well as the context.

Columns
-------
is_fomc_day, is_fomc_day_before, days_to_fomc, days_since_fomc,
fomc_cycle_week            (0-5: weeks since the last scheduled meeting, Cieslak-Morse-
                            Vissing-Jorgensen even/odd cycle)
is_cpi_day, is_nfp_day     (BLS 08:30 ET releases; from the FRED release-dates API when a
                            key is available, otherwise omitted)
days_to_month_end          (-3..+3 window around the last trading day of the month, else 0)
is_month_end, is_quarter_end
is_opex_day, is_opex_week  (monthly index option expiration: third Friday, or Thursday
                            when Friday is a holiday)
is_quad_witching           (opex in March, June, September, December)
day_of_week                (0 = Monday)
is_pre_holiday             (last trading day before an exchange holiday)

Sources: FOMC statement dates from data/processed/fomc_events_daily.csv plus scheduled
future meetings parsed from the Fed calendar page; NYSE trading calendar from
pandas_market_calendars; CPI / employment release dates from FRED releases 10 and 50.

Output: data/processed/event_calendar_daily.csv, one row per NYSE trading day.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FED_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
FRED_RELEASES = {"is_cpi_day": 10, "is_nfp_day": 50}  # Consumer Price Index, Employment Situation
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}


def trading_days(start: str, end: str) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).normalize()


def scheduled_fomc_from_fed_page() -> list[pd.Timestamp]:
    """Parse meeting end dates from the Fed calendar page (covers recent and upcoming years)."""
    try:
        html = requests.get(FED_CALENDAR_URL, timeout=30).text
    except requests.RequestException as exc:
        logging.warning("Could not fetch the Fed calendar page: %s", exc)
        return []
    dates: list[pd.Timestamp] = []
    for year_block in re.split(r'<div class="panel panel-default">', html)[1:]:
        year_match = re.search(r"(\d{4}) FOMC Meetings", year_block)
        if not year_match:
            continue
        year = int(year_match.group(1))
        months = re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([A-Za-z/]+)</strong>', year_block)
        days = re.findall(r'fomc-meeting__date[^>]*>\s*([0-9]{1,2}(?:-[0-9]{1,2})?)', year_block)
        for month_text, day_text in zip(months, days):
            month_name = month_text.split("/")[-1].lower()  # "Jan/Feb" -> meeting ends in Feb
            end_day = int(day_text.split("-")[-1])
            month = MONTHS.get(month_name) or MONTHS.get(month_name[:3])
            if month is None:
                for name, number in MONTHS.items():
                    if name.startswith(month_name[:3]):
                        month = number
            if month:
                dates.append(pd.Timestamp(year=year, month=month, day=end_day))
    logging.info("Fed calendar page: %d scheduled meetings (%s -> %s)", len(dates), min(dates).date() if dates else None, max(dates).date() if dates else None)
    return dates


def fred_release_dates(release_id: int, api_key: str, start: str) -> set[pd.Timestamp]:
    params = {"release_id": release_id, "api_key": api_key, "file_type": "json",
              "realtime_start": start, "realtime_end": "9999-12-31", "include_release_dates_with_no_data": "true", "limit": 10000}
    resp = requests.get(FRED_RELEASE_DATES_URL, params=params, timeout=60)
    resp.raise_for_status()
    return {pd.Timestamp(item["date"]) for item in resp.json().get("release_dates", [])}


def opex_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Third Friday of each month, or the preceding trading day if the Friday is a holiday."""
    result = []
    for (year, month), _ in days.to_series().groupby([days.year, days.month]):
        first = pd.Timestamp(year=year, month=month, day=1)
        fridays = [d for d in pd.date_range(first, first + pd.offsets.MonthEnd(0)) if d.weekday() == 4]
        third_friday = fridays[2]
        candidates = days[(days <= third_friday) & (days >= first)]
        if len(candidates):
            result.append(candidates[-1])
    return pd.DatetimeIndex(result)


def build(start: str, end: str, fomc_file: Path) -> pd.DataFrame:
    days = trading_days(start, end)
    table = pd.DataFrame({"date": days})

    # --- FOMC ---
    fomc = pd.read_csv(fomc_file)
    statement_dates = pd.to_datetime(fomc.loc[fomc["is_fomc_day"] == 1, "date"])
    # Seed with the last meetings before the sample so days_since_fomc and the cycle week
    # are defined from the first row (the statement file starts in 2006).
    seed = {pd.Timestamp("2005-11-01"), pd.Timestamp("2005-12-13")}
    all_fomc = sorted(set(statement_dates) | set(scheduled_fomc_from_fed_page()) | seed)
    fomc_index = pd.DatetimeIndex(all_fomc)
    table["is_fomc_day"] = table["date"].isin(fomc_index).astype(int)
    next_idx = fomc_index.searchsorted(table["date"], side="left")
    prev_idx = fomc_index.searchsorted(table["date"], side="right") - 1
    next_dates = pd.Series(np.where(next_idx < len(fomc_index), fomc_index[np.minimum(next_idx, len(fomc_index) - 1)], pd.NaT))
    prev_dates = pd.Series(np.where(prev_idx >= 0, fomc_index[np.maximum(prev_idx, 0)], pd.NaT))
    table["days_to_fomc"] = (pd.to_datetime(next_dates) - table["date"]).dt.days
    table["days_since_fomc"] = (table["date"] - pd.to_datetime(prev_dates)).dt.days
    table["is_fomc_day_before"] = (table["days_to_fomc"] == 1).astype(int)
    table["fomc_cycle_week"] = (table["days_since_fomc"] // 7).clip(upper=5)

    # --- BLS releases ---
    # CPI days come from the ALFRED vintage file when it exists: FRED's release calendar
    # (release 10) also lists the February seasonal-factor revision, which is not the
    # monthly print markets trade on. Payroll days come from FRED release 50.
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "")
    fred_file = fomc_file.with_name("fred_macro_daily.csv")
    if fred_file.exists() and "cpi_release_day" in pd.read_csv(fred_file, nrows=1).columns:
        fred = pd.read_csv(fred_file, parse_dates=["date"])
        cpi_days = set(fred.loc[fred["cpi_release_day"] == 1, "date"])
        table["is_cpi_day"] = table["date"].isin(cpi_days).astype(int)
        logging.info("is_cpi_day: %d release days (from ALFRED vintages)", int(table["is_cpi_day"].sum()))
    elif api_key and not api_key.startswith("your_"):
        dates = fred_release_dates(FRED_RELEASES["is_cpi_day"], api_key, start)
        table["is_cpi_day"] = table["date"].isin(dates).astype(int)
        logging.info("is_cpi_day: %d release days (from FRED release calendar)", int(table["is_cpi_day"].sum()))
    if api_key and not api_key.startswith("your_"):
        dates = fred_release_dates(FRED_RELEASES["is_nfp_day"], api_key, start)
        table["is_nfp_day"] = table["date"].isin(dates).astype(int)
        logging.info("is_nfp_day: %d release days", int(table["is_nfp_day"].sum()))
    else:
        logging.warning("FRED_API_KEY not set: skipping is_nfp_day")

    # --- month / quarter structure ---
    month_key = table["date"].dt.to_period("M")
    last_of_month = table.groupby(month_key)["date"].transform("max")
    position = table.groupby(month_key).cumcount()
    count = table.groupby(month_key)["date"].transform("count")
    from_end = count - 1 - position  # 0 on the last trading day
    table["is_month_end"] = (from_end == 0).astype(int)
    table["is_quarter_end"] = ((from_end == 0) & table["date"].dt.month.isin([3, 6, 9, 12])).astype(int)
    # -3..-1 on the last trading days before month end, 0 on month end, +1..+3 on the
    # first trading days of the next month, 0 elsewhere (McConnell-Xu / Etula et al. window).
    window = pd.Series(0, index=table.index)
    before = from_end.between(1, 3)
    after = position.between(0, 2)
    window[before] = -from_end[before]
    window[after] = position[after] + 1
    table["days_to_month_end"] = window.astype(int)

    # --- option expiration ---
    opex = opex_days(days)
    table["is_opex_day"] = table["date"].isin(opex).astype(int)
    opex_weeks = set(pd.DatetimeIndex(opex).to_period("W"))
    table["is_opex_week"] = table["date"].dt.to_period("W").isin(opex_weeks).astype(int)
    table["is_quad_witching"] = (table["is_opex_day"].astype(bool) & table["date"].dt.month.isin([3, 6, 9, 12])).astype(int)

    # --- weekday / holidays ---
    table["day_of_week"] = table["date"].dt.weekday
    gap_to_next = (table["date"].shift(-1) - table["date"]).dt.days
    expected_gap = table["date"].dt.weekday.map({4: 3}).fillna(1)
    table["is_pre_holiday"] = (gap_to_next > expected_gap).fillna(False).astype(int)

    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build known-future calendar covariates")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None, help="Defaults to 120 calendar days after today so future flags exist")
    parser.add_argument("--fomc-file", type=Path, default=Path("data/processed/fomc_events_daily.csv"))
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    end = args.end or (pd.Timestamp.today() + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    # Build 45 days further so month-end and opex flags in the final month are computed
    # against the complete month, then trim.
    build_end = (pd.Timestamp(end) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    table = build(args.start, build_end, args.fomc_file)
    table = table[table["date"] <= end].reset_index(drop=True)
    path = Path(args.out_dir) / "processed" / "event_calendar_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d trading days, %d columns)", path, len(table), len(table.columns))
    logging.info("Flag totals: %s", table.filter(like="is_").sum().to_dict())


if __name__ == "__main__":
    main()
