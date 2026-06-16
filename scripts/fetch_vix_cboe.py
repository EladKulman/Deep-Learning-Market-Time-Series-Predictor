#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pandas as pd
import requests
import io
import pandas_datareader.data as web

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

def fetch_cboe_vix():
    logging.info(f"Attempting to fetch VIX from CBOE: {CBOE_VIX_URL}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(CBOE_VIX_URL, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df, "cboe"
    except Exception as e:
        logging.warning(f"Failed to fetch CBOE VIX: {e}")
        return pd.DataFrame(), None

def fetch_fred_vix(start_date, end_date):
    logging.info("Falling back to FRED VIXCLS...")
    try:
        s = web.DataReader('VIXCLS', 'fred', start_date, end_date)
        df = s.reset_index()
        df = df.rename(columns={'DATE': 'date', 'VIXCLS': 'vix_close'})
        return df, "fred"
    except Exception as e:
        logging.error(f"Failed to fetch FRED VIXCLS: {e}")
        return pd.DataFrame(), None

def process_vix(df, source, start_date, end_date):
    if df.empty:
        return df
        
    if source == "cboe":
        # CBOE format usually has DATE, OPEN, HIGH, LOW, CLOSE
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Try to parse date
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        rename_map = {
            'open': 'vix_open',
            'high': 'vix_high',
            'low': 'vix_low',
            'close': 'vix_close'
        }
        df = df.rename(columns=rename_map)
        
        expected_cols = ['date', 'vix_open', 'vix_high', 'vix_low', 'vix_close']
        available_cols = [c for c in expected_cols if c in df.columns]
        df = df[available_cols]
        
    elif source == "fred":
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        # FRED only gives close
        df = df[['date', 'vix_close']]
        
    # Filter dates
    df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
        
    df = df.sort_values('date').reset_index(drop=True)
    return df

def validate_data(df, source):
    if df.empty:
        logging.error("Validation failed: Data is empty.")
        return
        
    logging.info("--- VALIDATION ---")
    min_date, max_date = df['date'].min(), df['date'].max()
    logging.info(f"Date range: {min_date} to {max_date}")
    
    # Positive VIX values
    if (df['vix_close'] <= 0).any():
        logging.warning("Validation issue: Some VIX close values are <= 0 or missing!")
        
    if source == 'cboe':
        logging.info("Source is CBOE. Includes OHLC.")
    else:
        logging.info("Source is FRED. Includes only Close.")

def main():
    parser = argparse.ArgumentParser(description="Fetch and process VIX data")
    parser.add_argument("--start", type=str, default="2006-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "cboe"
    proc_dir = out_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Try CBOE first
    raw_df, source = fetch_cboe_vix()
    
    # Fallback
    if raw_df.empty:
        raw_df, source = fetch_fred_vix(args.start, args.end)
        
    if not raw_df.empty:
        raw_path = raw_dir / "vix_raw.csv"
        raw_df.to_csv(raw_path, index=False)
        
        proc_df = process_vix(raw_df, source, args.start, args.end)
        validate_data(proc_df, source)
        
        proc_path = proc_dir / "vix_daily.csv"
        proc_df.to_csv(proc_path, index=False)
        
        logging.info("--- SUMMARY: VIX ---")
        logging.info(f"Rows: {len(proc_df)}")
        logging.info(f"Missing values:\n{proc_df.isna().sum()}")
        logging.info(f"Raw Output: {raw_path}")
        logging.info(f"Processed Output: {proc_path}")
        logging.info("--------------------\n")

if __name__ == "__main__":
    main()
