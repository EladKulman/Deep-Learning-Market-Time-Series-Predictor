#!/usr/bin/env python3
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import pandas_datareader.data as web
import ssl

# Workaround for macOS Python SSL certificate verify failed errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_series_fredapi(series_id, start_date, end_date, api_key):
    fred = Fred(api_key=api_key)
    try:
        s = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        return s
    except Exception as e:
        logging.error(f"Error fetching {series_id} via fredapi: {e}")
        return pd.Series(dtype=float)

def fetch_series_pdr(series_id, start_date, end_date):
    try:
        df = web.DataReader(series_id, "fred", start_date, end_date)
        return df[series_id]
    except Exception as e:
        logging.error(f"Error fetching {series_id} via pandas_datareader: {e}")
        return pd.Series(dtype=float)

def fetch_macro_data(start_date, end_date):
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    
    series_dict = {
        'DFF': 'dff',
        'DGS10': 'dgs10',
        'DGS2': 'dgs2',
        'T10Y2Y': 't10y2y',
        'CPIAUCSL': 'cpi',
        'VIXCLS': 'vix_fred'
    }
    
    data = {}
    for sid in series_dict.keys():
        logging.info(f"Fetching {sid}...")
        if api_key and HAS_FREDAPI:
            s = fetch_series_fredapi(sid, start_date, end_date, api_key)
            if s.empty: # Fallback
                logging.info(f"Fallback to pandas_datareader for {sid}")
                s = fetch_series_pdr(sid, start_date, end_date)
        else:
            s = fetch_series_pdr(sid, start_date, end_date)
        
        s.name = series_dict[sid]
        data[sid] = s
        
    return data

def process_macro_data(raw_data_dict, start_date, end_date):
    # Combine all series into a dataframe
    df = pd.DataFrame(raw_data_dict)
    
    # Process CPI YoY BEFORE forward-filling daily
    cpi = raw_data_dict.get('CPIAUCSL')
    if cpi is not None and not cpi.dropna().empty:
        # CPI is monthly. Drop na to get only the reported months
        cpi_monthly = cpi.dropna()
        cpi_yoy = (cpi_monthly / cpi_monthly.shift(12)) - 1
        df['cpi_yoy'] = cpi_yoy
    else:
        df['cpi_yoy'] = float('nan')

    # Rename columns to their simple names
    col_map = {
        'DFF': 'dff',
        'DGS10': 'dgs10',
        'DGS2': 'dgs2',
        'T10Y2Y': 't10y2y',
        'CPIAUCSL': 'cpi',
        'VIXCLS': 'vix_fred',
        'cpi_yoy': 'cpi_yoy'
    }
    df = df.rename(columns=col_map)
    
    # We want a daily calendar
    if not df.empty:
        # Reindex to a complete daily calendar
        idx = pd.date_range(start=start_date, end=end_date if end_date else pd.Timestamp.today().strftime('%Y-%m-%d'), freq='D')
        df = df.reindex(idx)
        
        # Forward fill values (prices, rates, CPI)
        df = df.ffill()
        
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Ensure correct column order
    expected_cols = ['date', 'dff', 'dgs10', 'dgs2', 't10y2y', 'cpi', 'cpi_yoy', 'vix_fred']
    # Add any missing columns as nan
    for c in expected_cols:
        if c not in df.columns:
            df[c] = float('nan')
            
    df = df[expected_cols]
    return df

def validate_data(df: pd.DataFrame):
    if df.empty:
        logging.error("Validation failed: Data is empty.")
        return
        
    logging.info("--- VALIDATION ---")
    
    # Missing percentages
    missing_pct = (df.isna().sum() / len(df)) * 100
    logging.info(f"Missing percentage per column:\n{missing_pct.round(2)}")
    
    # Confirm numeric
    numeric_cols = ['dff', 'dgs10', 'dgs2', 't10y2y', 'cpi', 'cpi_yoy', 'vix_fred']
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            logging.warning(f"Column {c} is not numeric!")
            
    # Confirm CPI filling
    # Originally CPI changes only once a month. Since we ffill, `cpi` should have many consecutive duplicates.
    cpi_changes = (df['cpi'] != df['cpi'].shift(1)).sum()
    logging.info(f"CPI changed {cpi_changes} times over {len(df)} days (confirms monthly ffill pattern).")
    
    min_date, max_date = df['date'].min(), df['date'].max()
    logging.info(f"Date range: {min_date} to {max_date}")

def main():
    parser = argparse.ArgumentParser(description="Fetch and process FRED macro data")
    parser.add_argument("--start", type=str, default="2006-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "fred"
    proc_dir = out_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data = fetch_macro_data(args.start, args.end)
    
    if raw_data:
        # Save raw (as one combined CSV or separate, let's combine for raw)
        raw_df = pd.DataFrame(raw_data)
        raw_df.index.name = 'date'
        raw_path = raw_dir / "fred_macro_raw.csv"
        raw_df.to_csv(raw_path)
        
        proc_df = process_macro_data(raw_data, args.start, args.end)
        validate_data(proc_df)
        
        proc_path = proc_dir / "fred_macro_daily.csv"
        proc_df.to_csv(proc_path, index=False)
        
        logging.info("--- SUMMARY: FRED ---")
        logging.info(f"Rows: {len(proc_df)}")
        logging.info(f"Raw Output: {raw_path}")
        logging.info(f"Processed Output: {proc_path}")
        logging.info("---------------------\n")

if __name__ == "__main__":
    main()
