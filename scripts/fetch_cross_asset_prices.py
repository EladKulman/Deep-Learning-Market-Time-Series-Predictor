#!/usr/bin/env python3
"""Fetch daily closes for cross-asset ETFs and indices used as QQQ covariates.

Tickers (yfinance):
  TLT  20+ year Treasury ETF      stock-bond correlation regime
  HYG  high-yield credit ETF      same-day credit signal (FRED OAS lags a day)
  SPY  S&P 500 ETF                only used as QQQ - SPY relative return
  IWM  Russell 2000 ETF           size tilt (IWM - SPY)
  SMH  semiconductor ETF          semis leadership (SMH - QQQ)
  GLD  gold ETF
  CL=F WTI crude front-month
  DX-Y.NYB  US dollar index

All values are close-to-close, so the day-T value is known at the day-T close.
Output: data/processed/cross_asset_daily.csv with <ticker>_close and <ticker>_log_ret columns.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TICKERS = {
    "TLT": "tlt",
    "HYG": "hyg",
    "SPY": "spy",
    "IWM": "iwm",
    "SMH": "smh",
    "GLD": "gld",
    "CL=F": "oil",
    "DX-Y.NYB": "dxy",
}


def fetch_one(ticker: str, start: str, end: str | None, retries: int = 4) -> pd.Series:
    for attempt in range(retries):
        try:
            frame = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if not frame.empty:
                closes = frame["Close"].copy()
                closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
                return closes[~closes.index.duplicated(keep="last")]
            logging.warning("%s: empty response (attempt %d)", ticker, attempt + 1)
        except Exception as exc:  # yfinance raises assorted exceptions on rate limits
            logging.warning("%s: %s (attempt %d)", ticker, exc, attempt + 1)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {ticker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch cross-asset daily closes")
    parser.add_argument("--start", default="2005-12-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    raw_dir = Path(args.out_dir) / "raw" / "prices"
    proc_dir = Path(args.out_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame()
    for ticker, prefix in TICKERS.items():
        closes = fetch_one(ticker, args.start, args.end)
        closes.to_csv(raw_dir / f"{prefix}_close_raw.csv", header=["close"])
        logging.info("%s: %d rows, %s -> %s", ticker, len(closes), closes.index.min().date(), closes.index.max().date())
        table[f"{prefix}_close"] = closes
        table[f"{prefix}_log_ret"] = np.log(closes).diff()
        time.sleep(2)

    table.index.name = "date"
    table = table.sort_index().reset_index()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    path = proc_dir / "cross_asset_daily.csv"
    table.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows, %d columns)", path, len(table), len(table.columns))


if __name__ == "__main__":
    main()
