#!/usr/bin/env python3
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_yfinance_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    logging.info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        logging.warning(f"No data returned for {ticker_symbol}")
    return df

def process_qqq_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # yfinance returns index as Date (or Datetime if intraday). We want it as a column normalized to YYYY-MM-DD
    df = df.reset_index()
    # Handle both 'Date' and 'Datetime' returned by different versions of yfinance
    if 'Datetime' in df.columns:
        df['date'] = df['Datetime'].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['Datetime'])
    elif 'Date' in df.columns:
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['Date'])
    
    # Make columns lowercase and select required
    df.columns = [c.lower() for c in df.columns]
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    # yfinance column for adj close is usually 'Adj Close'. 
    # But `.history(auto_adjust=False)` returns 'Close' and 'Adj Close'
    if 'adj close' not in df.columns:
        raise ValueError("Adjusted close is required for the total-return target")
        
    df = df.rename(columns={'adj close': 'adj_close'})
    df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Compute features
    # Use adjusted close for returns to account for dividends/splits
    df['return_1d'] = df['adj_close'].pct_change()
    df['return_5d'] = df['adj_close'].pct_change(periods=5)
    df['log_return_1d'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    
    # Volatility (20-day standard deviation of log returns * sqrt(252) for annualized or just standard dev)
    # The requirement says `volatility_20d`. We'll compute the rolling standard deviation of 1d log returns.
    df['volatility_20d'] = df['log_return_1d'].rolling(window=20).std()
    
    # Volume change
    df['volume_change_1d'] = df['volume'].pct_change()
    
    return df

def validate_data(df: pd.DataFrame, ticker: str):
    if df.empty:
        logging.error(f"Validation failed: Data for {ticker} is empty.")
        return
    
    # 1. Unique dates
    if not df['date'].is_unique:
        logging.warning("Validation issue: Dates are not unique!")
    
    # 2. Positive close
    if (df['close'] <= 0).any():
        logging.warning("Validation issue: Some close prices are <= 0!")
        
    # 3. Volume exists
    if df['volume'].isna().any() or (df['volume'] == 0).all():
        logging.warning("Validation issue: Volume data is missing or entirely zeros!")
        
    # Print date range
    min_date, max_date = df['date'].min(), df['date'].max()
    logging.info(f"Date range: {min_date} to {max_date}")
    
    # Warn for large missing periods (US trading days usually miss weekends and holidays, so >4 days is a warning)
    date_diffs = pd.to_datetime(df['date']).diff().dt.days
    max_gap = date_diffs.max()
    if max_gap > 5:
        logging.warning(f"Large missing period detected! Max gap is {max_gap} days.")

def main():
    parser = argparse.ArgumentParser(description="Fetch and process QQQ prices")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Source start date; include a year before modeling for rolling-feature warm-up")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default is today")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "prices"
    proc_dir = out_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch QQQ
    qqq_raw = fetch_yfinance_data("QQQ", args.start, args.end)
    
    if not qqq_raw.empty:
        # Save raw
        raw_path = raw_dir / "qqq_ohlcv_raw.csv"
        qqq_raw.to_csv(raw_path)
        
        # Process
        qqq_proc = process_qqq_prices(qqq_raw)
        
        # Validate
        validate_data(qqq_proc, "QQQ")
        
        # Save processed
        proc_path = proc_dir / "qqq_ohlcv_daily.csv"
        qqq_proc.to_csv(proc_path, index=False)
        
        # Print summary
        logging.info("--- SUMMARY: QQQ ---")
        logging.info(f"Rows: {len(qqq_proc)}")
        logging.info(f"Missing values:\n{qqq_proc.isna().sum()}")
        logging.info(f"Raw Output: {raw_path}")
        logging.info(f"Processed Output: {proc_path}")
        logging.info("--------------------\n")

    # Optional: fetch ^NDX as well to have the raw index data
    ndx_raw = fetch_yfinance_data("^NDX", args.start, args.end)
    if not ndx_raw.empty:
        ndx_raw_path = raw_dir / "ndx_ohlcv_raw.csv"
        ndx_raw.to_csv(ndx_raw_path)
        logging.info(f"Saved ^NDX raw data to {ndx_raw_path}")

if __name__ == "__main__":
    main()
