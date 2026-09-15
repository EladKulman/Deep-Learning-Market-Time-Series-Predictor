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
is_cpi_day, is_nfp_day     (published BLS schedules plus saved FRED/ALFRED history;
                            NaN beyond the published schedule)
days_to_month_end          (-3..+3 window around the last trading day of the month, else 0)
is_month_end, is_quarter_end
is_opex_day, is_opex_week  (monthly index option expiration: third Friday, or Thursday
                            when Friday is a holiday)
is_quad_witching           (opex in March, June, September, December)
day_of_week                (0 = Monday)
is_pre_holiday             (last trading day before an exchange holiday)

Sources: scheduled Fed meeting headings (emergency statements excluded); NYSE calendar
from pandas_market_calendars; BLS release schedules and FRED/ALFRED release history.
Historical schedules are reconstructed, not a vintage archive of every calendar change.

Output: data/processed/event_calendar_daily.csv, one row per NYSE trading day.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup

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


def _month_number(text: str) -> int | None:
    key = text.split("/")[-1].strip().lower()
    return next((n for m, n in MONTHS.items() if m.startswith(key[:3])), None) if key else None


def parse_meeting_heading(label: str, year: int) -> pd.Timestamp | None:
    """Return the meeting END date from a historical-page heading.

    Handles "January 30-31 Meeting", "Jan/Feb 31-1 Meeting" (month pair, end day after the
    dash) and the cross-month "July 31-August 1 Meeting" form (end month named explicitly).
    The statement is released on the last day of a two-day meeting, so the end date is the
    event date.
    """
    match = re.match(r"([A-Za-z/]+)\s+(\d{1,2})(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2}))?", label.strip())
    if not match:
        return None
    start_month, first, end_month_text, last = match.groups()
    month = _month_number(end_month_text) if end_month_text else _month_number(start_month)
    if month is None:
        return None
    return pd.Timestamp(year=year, month=month, day=int(last or first))


def scheduled_fomc_from_fed_page(cache_dir: Path = Path("data/raw/calendar"), cached: bool = False) -> list[pd.Timestamp]:
    """Parse meeting end dates from the Fed calendar page (covers recent and upcoming years)."""
    path = cache_dir / "fomccalendars.htm"
    if cached:
        html = path.read_text()
    else:
        response = requests.get(FED_CALENDAR_URL, timeout=30)
        response.raise_for_status()
        html = response.text
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
    dates: list[pd.Timestamp] = []
    for year_block in re.split(r'<div class="panel panel-default">', html)[1:]:
        year_match = re.search(r"(\d{4}) FOMC Meetings", year_block)
        if not year_match:
            continue
        year = int(year_match.group(1))
        for row in BeautifulSoup(year_block, "html.parser").select(".fomc-meeting"):
            month_node, date_node = row.select_one(".fomc-meeting__month"), row.select_one(".fomc-meeting__date")
            if month_node is None or date_node is None:
                continue
            month_text, day_text = month_node.get_text(strip=True), date_node.get_text(strip=True)
            if any(word in day_text.lower() for word in ("unscheduled", "notation", "conference")):
                continue
            match = re.match(r"\d{1,2}(?:-\d{1,2})?", day_text)
            if not match:
                continue
            day_text = match.group()
            month_name = month_text.split("/")[-1].lower()  # "Jan/Feb" -> meeting ends in Feb
            end_day = int(day_text.split("-")[-1])
            month = MONTHS.get(month_name) or MONTHS.get(month_name[:3])
            if month is None:
                for name, number in MONTHS.items():
                    if name.startswith(month_name[:3]):
                        month = number
            if month:
                dates.append(pd.Timestamp(year=year, month=month, day=end_day))
    # Historical statement dates include emergency calls; use the scheduled meeting
    # headings instead. Keep the originally scheduled March 18, 2020 date, subsequently
    # cancelled, so the historical feature does not learn the cancellation in advance.
    for year in range(2006, 2021):
        historical = cache_dir / f"fomchistorical{year}.htm"
        if not historical.exists():
            raise FileNotFoundError(f"{historical}: run fetch_fomc_statements.py first")
        for heading in BeautifulSoup(historical.read_text(), "html.parser").find_all("h5"):
            label = heading.get_text(" ", strip=True)
            if "Meeting" not in label or "unscheduled" in label.lower():
                continue
            parsed = parse_meeting_heading(label, year)
            if parsed is not None:
                dates.append(parsed)
    if not dates:
        raise ValueError("No scheduled FOMC meetings parsed")
    logging.info("Scheduled FOMC meetings: %d (%s -> %s)", len(set(dates)), min(dates).date(), max(dates).date())
    return sorted(set(dates))


def bls_schedule(flag: str, cache_dir: Path, cached: bool) -> tuple[set[pd.Timestamp], pd.Timestamp]:
    name = "cpi" if flag == "is_cpi_day" else "empsit"
    url = f"https://www.bls.gov/schedule/news_release/{name}.htm"
    path = cache_dir / f"bls_{name}.csv"
    if not cached:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            table = next(t for t in tables if "Release Date" in t.columns)
            dates = pd.to_datetime(table["Release Date"].str.replace(".", "", regex=False), format="mixed")
            result = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "source_url": url,
                                   "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat()})
            result["coverage_through"] = (dates.max() + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            result.to_csv(path, index=False)
        except (requests.RequestException, ValueError, StopIteration) as exc:
            if not path.exists():
                raise RuntimeError(f"Could not load official BLS schedule {url}; no cached schedule") from exc
            logging.warning("BLS schedule download unavailable (%s); using %s", type(exc).__name__, path)
    saved = pd.read_csv(path)
    return set(pd.to_datetime(saved["date"])), pd.Timestamp(saved["coverage_through"].max())


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


def build(start: str, end: str, fomc_file: Path, cached: bool = False) -> pd.DataFrame:
    days = trading_days(start, end)
    table = pd.DataFrame({"date": days})

    # --- FOMC ---
    # Seed with the last meetings before the sample so days_since_fomc and the cycle week
    # are defined from the first row (the statement file starts in 2006).
    seed = {pd.Timestamp("2005-11-01"), pd.Timestamp("2005-12-13")}
    cache_dir = fomc_file.parent.parent / "raw" / "calendar"
    all_fomc = sorted(set(scheduled_fomc_from_fed_page(cache_dir, cached)) | seed)
    fomc_index = pd.DatetimeIndex(all_fomc)
    table["is_fomc_day"] = table["date"].isin(fomc_index).astype(int)
    next_idx = fomc_index.searchsorted(table["date"], side="left")
    prev_idx = fomc_index.searchsorted(table["date"], side="right") - 1
    next_dates = pd.Series(np.where(next_idx < len(fomc_index), fomc_index[np.minimum(next_idx, len(fomc_index) - 1)], pd.NaT))
    prev_dates = pd.Series(np.where(prev_idx >= 0, fomc_index[np.maximum(prev_idx, 0)], pd.NaT))
    table["days_to_fomc"] = (pd.to_datetime(next_dates) - table["date"]).dt.days
    table["days_since_fomc"] = (table["date"] - pd.to_datetime(prev_dates)).dt.days
    table["is_fomc_day_before"] = table["is_fomc_day"].shift(-1).fillna(0).astype(int)
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
    if api_key and not api_key.startswith("your_") and not cached:
        dates = fred_release_dates(FRED_RELEASES["is_nfp_day"], api_key, start)
        table["is_nfp_day"] = table["date"].isin(dates).astype(int)
        pd.DataFrame({"date": sorted(dates)}).to_csv(cache_dir / "nfp_historical.csv", index=False)
        logging.info("is_nfp_day: %d release days", int(table["is_nfp_day"].sum()))
    else:
        history = pd.read_csv(cache_dir / "nfp_historical.csv", parse_dates=["date"])
        table["is_nfp_day"] = table["date"].isin(history["date"]).astype(int)

    # ALFRED knows published CPI values, not upcoming release dates. Merge the BLS
    # schedule explicitly, and represent dates beyond the published horizon as unknown.
    for flag in ("is_cpi_day", "is_nfp_day"):
        release_dates, coverage_end = bls_schedule(flag, cache_dir, cached)
        if flag not in table:
            raise ValueError(f"Historical calendar is unavailable for {flag}")
        table.loc[table["date"].isin(release_dates), flag] = 1
        table.loc[table["date"] > coverage_end, flag] = np.nan

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
    parser.add_argument("--cached", action="store_true", help="Use saved Fed/BLS schedules and historical release dates")
    args = parser.parse_args()

    end = args.end or (pd.Timestamp.today() + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    # Build 45 days further so month-end and opex flags in the final month are computed
    # against the complete month, then trim.
    build_end = (pd.Timestamp(end) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    table = build(args.start, build_end, args.fomc_file, args.cached)
    table = table[table["date"] <= end].reset_index(drop=True)
    path = Path(args.out_dir) / "processed" / "event_calendar_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d trading days, %d columns)", path, len(table), len(table.columns))
    logging.info("Flag totals: %s", table.filter(like="is_").sum().to_dict())


if __name__ == "__main__":
    main()
