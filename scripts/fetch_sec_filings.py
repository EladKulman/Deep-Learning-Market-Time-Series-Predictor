#!/usr/bin/env python3
"""Fetch SEC EDGAR filing events for the largest Nasdaq-100 companies.

Changes versus the first version
--------------------------------
* EDGAR's submissions endpoint only returns the most recent ~1,000 filings inline; older
  ones live in the paginated files listed under ``filings.files``. Those are now fetched, so
  history is complete back to 2006 instead of silently starting in 2015.
* TSMC is a foreign private issuer and files 20-F (annual) and 6-K (current) instead of
  10-K / 10-Q / 8-K. Those forms are now counted.
* 8-K filings carry an ``items`` list. Item 2.02 ("Results of Operations and Financial
  Condition") is the earnings release, which gives a free, authoritative earnings calendar
  back to 2003. The acceptance timestamp splits it into pre-market (before 09:30 ET) and
  post-market (16:00 ET or later) events so the feature table can attach the post-market
  ones to the next trading day.

Output: data/processed/sec_filings_daily.csv, one row per calendar day from --start.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "INTC", "TSM", "AVGO"]
# Companies whose earlier filings live under a predecessor registrant:
# Alphabet Inc. (2015-) succeeded Google Inc.; Broadcom Inc. (2018-) succeeded Broadcom Ltd / Avago.
PREDECESSOR_CIKS = {"GOOGL": ["0001288776"], "AVGO": ["0001441634"]}
FORM_COLUMNS = {
    "10-K": "sec_10k_count",
    "10-Q": "sec_10q_count",
    "8-K": "sec_8k_count",
    "20-F": "sec_20f_count",
    "6-K": "sec_6k_count",
}
EARNINGS_ITEM = "2.02"


def get_headers() -> dict[str, str]:
    load_dotenv()
    email = os.environ.get("SEC_USER_AGENT_EMAIL", "")
    if not email or "example.com" in email or email.startswith("your_"):
        raise SystemExit("Set SEC_USER_AGENT_EMAIL in .env to a real contact address (SEC requires it).")
    return {"User-Agent": f"Nasdaq100ForecastingProject/0.2 contact: {email}", "Accept-Encoding": "gzip, deflate"}


def get_json(url: str, headers: dict[str, str], retries: int = 3) -> dict:
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            logging.warning("%s returned 404", url)
            return {}
        logging.warning("%s -> %s (attempt %d)", url, resp.status_code, attempt + 1)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_company_tickers(headers: dict[str, str]) -> dict[str, str]:
    data = get_json("https://www.sec.gov/files/company_tickers.json", headers)
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in data.values()}


def fetch_all_filings(cik: str, headers: dict[str, str], raw_dir: Path, ticker: str) -> pd.DataFrame:
    """Return every filing for a CIK, combining the inline 'recent' block with the paginated files."""
    main = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", headers)
    if not main:
        return pd.DataFrame()
    (raw_dir / f"sec_submissions_{ticker}.json").write_text(json.dumps(main), encoding="utf-8")

    parts = [pd.DataFrame(main["filings"]["recent"])]
    for extra in main["filings"].get("files", []):
        time.sleep(0.15)
        older = get_json(f"https://data.sec.gov/submissions/{extra['name']}", headers)
        if older:
            (raw_dir / f"sec_submissions_{ticker}_{extra['name']}").write_text(json.dumps(older), encoding="utf-8")
            parts.append(pd.DataFrame(older))
            logging.info("  %s: fetched %s (%s -> %s)", ticker, extra["name"], extra["filingFrom"], extra["filingTo"])
    return pd.concat(parts, ignore_index=True)


def process_filings(frame: pd.DataFrame, ticker: str, start: str, end: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame[frame["form"].isin(FORM_COLUMNS)].copy()
    frame["date"] = frame["filingDate"]
    frame = frame[frame["date"] >= start]
    if end:
        frame = frame[frame["date"] <= end]
    frame["ticker"] = ticker
    items = frame.get("items", pd.Series("", index=frame.index)).fillna("")
    frame["is_earnings"] = (frame["form"] == "8-K") & items.str.split(",").apply(
        lambda parts: EARNINGS_ITEM in [p.strip() for p in parts]
    )
    # Anything accepted before the 16:00 ET close (pre-market or intraday) can affect day T;
    # acceptance at or after the close affects the next session. Missing timestamps are
    # treated as before the close, the conservative choice for a same-day flag.
    accepted = pd.to_datetime(frame.get("acceptanceDateTime"), errors="coerce")
    minutes = accepted.dt.hour * 60 + accepted.dt.minute
    frame["session"] = "pre"
    frame.loc[minutes >= 16 * 60, "session"] = "post"
    return frame[["date", "ticker", "form", "is_earnings", "session", "accessionNumber"]]


def build_daily(filings: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    daily = pd.DataFrame({"date": pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")})

    counts = filings.groupby(["date", "form"]).size().unstack(fill_value=0)
    for form, column in FORM_COLUMNS.items():
        daily[column] = daily["date"].map(counts[form] if form in counts else pd.Series(dtype=int)).fillna(0).astype(int)
    daily["sec_total_filings"] = daily[list(FORM_COLUMNS.values())].sum(axis=1)

    for ticker in TICKERS:
        dates = set(filings.loc[filings["ticker"] == ticker, "date"])
        daily[f"{ticker.lower()}_filing_event"] = daily["date"].isin(dates).astype(int)

    earnings = filings[filings["is_earnings"]]
    daily["ndx_earnings_count"] = daily["date"].map(earnings.groupby("date").size()).fillna(0).astype(int)
    for session in ("pre", "post"):
        subset = earnings[earnings["session"] == session]
        daily[f"ndx_earnings_{session}market"] = daily["date"].map(subset.groupby("date").size()).fillna(0).astype(int)
    for ticker in TICKERS:
        dates = set(earnings.loc[earnings["ticker"] == ticker, "date"])
        if dates:  # foreign issuers (TSMC) report via 6-K, so they have no Item 2.02 flag
            daily[f"{ticker.lower()}_earnings_event"] = daily["date"].isin(dates).astype(int)
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SEC filings for major Nasdaq-100 companies")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw" / "sec"
    proc_dir = Path(args.out_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    headers = get_headers()
    ticker_to_cik = fetch_company_tickers(headers)

    frames = []
    for ticker in TICKERS:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            logging.warning("No CIK for %s", ticker)
            continue
        logging.info("Fetching %s (CIK %s)", ticker, cik)
        parts = [fetch_all_filings(cik, headers, raw_dir, ticker)]
        for old_cik in PREDECESSOR_CIKS.get(ticker, []):
            logging.info("  %s: predecessor CIK %s", ticker, old_cik)
            parts.append(fetch_all_filings(old_cik, headers, raw_dir, f"{ticker}_pred{old_cik}"))
        filings = process_filings(pd.concat(parts, ignore_index=True), ticker, args.start, args.end)
        filings = filings.drop_duplicates(subset=["accessionNumber"])
        if filings.empty:
            logging.warning("No target filings for %s", ticker)
            continue
        logging.info("  %s: %d filings, %d earnings 8-Ks, %s -> %s", ticker, len(filings),
                     int(filings["is_earnings"].sum()), filings["date"].min(), filings["date"].max())
        frames.append(filings)
        time.sleep(0.2)

    filings = pd.concat(frames, ignore_index=True)
    filings.to_csv(raw_dir / "sec_filings_long.csv", index=False)

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    daily = build_daily(filings, args.start, end)
    path = proc_dir / "sec_filings_daily.csv"
    daily.to_csv(path, index=False)
    per_year = filings.groupby(filings["date"].str[:4]).size()
    logging.info("Filings per year:\n%s", per_year.to_string())
    logging.info("Wrote %s (%d days, %d columns)", path, len(daily), len(daily.columns))


if __name__ == "__main__":
    main()
