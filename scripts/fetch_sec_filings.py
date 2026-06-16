#!/usr/bin/env python3
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import requests
import json
import time
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "INTC", "TSM", "AVGO"]
TARGET_FORMS = ["10-K", "10-Q", "8-K"]

def get_headers():
    load_dotenv()
    email = os.environ.get("SEC_USER_AGENT_EMAIL", "test@example.com")
    return {
        "User-Agent": f"Nasdaq100ForecastingProject/0.1 contact: {email}",
        "Accept-Encoding": "gzip, deflate"
    }

def fetch_company_tickers(headers):
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # Map ticker to CIK
    ticker_to_cik = {}
    for idx, company in data.items():
        ticker_to_cik[company['ticker']] = str(company['cik_str']).zfill(10)
    return ticker_to_cik

def fetch_submissions(cik, headers):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 404:
        # Some foreign issuers like TSM might not have normal JSON submissions or use different formats
        logging.warning(f"CIK {cik} returned 404. Might be a foreign issuer or unmapped.")
        return {}
    response.raise_for_status()
    return response.json()

def process_submissions(data, ticker, start_date, end_date):
    if not data or 'filings' not in data or 'recent' not in data['filings']:
        return pd.DataFrame()
        
    recent = data['filings']['recent']
    df = pd.DataFrame(recent)
    
    if df.empty:
        return df
        
    # Filter forms
    df = df[df['form'].isin(TARGET_FORMS)]
    
    # Required columns: date (we will use filingDate as the event date), ticker, form, filing_date, report_date, accession_number, primary_document
    df = df.rename(columns={
        'filingDate': 'filing_date',
        'reportDate': 'report_date',
        'accessionNumber': 'accession_number',
        'primaryDocument': 'primary_document'
    })
    
    # The event date is the filing date
    df['date'] = df['filing_date']
    df['ticker'] = ticker
    
    # Filter by date range
    df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
        
    cols = ['date', 'ticker', 'form', 'filing_date', 'report_date', 'accession_number', 'primary_document']
    # Keep available
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    return df

def main():
    parser = argparse.ArgumentParser(description="Fetch SEC filings for major tech companies")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "sec"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    headers = get_headers()
    
    logging.info("Fetching SEC company_tickers.json...")
    ticker_to_cik = fetch_company_tickers(headers)
    
    all_filings = []
    
    for ticker in TICKERS:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            logging.warning(f"Could not find CIK for {ticker}")
            continue
            
        logging.info(f"Fetching submissions for {ticker} (CIK: {cik})...")
        data = fetch_submissions(cik, headers)
        
        if data:
            raw_path = raw_dir / f"sec_submissions_{ticker}.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
                
            df = process_submissions(data, ticker, args.start, args.end)
            if not df.empty:
                all_filings.append(df)
                latest = df['date'].max()
                logging.info(f"Found {len(df)} filings for {ticker}. Latest: {latest}")
            else:
                logging.warning(f"No targeted filings found for {ticker} in date range.")
                
        # SEC rate limit is 10 requests per second. Be safe with 5 per second.
        time.sleep(0.2)
        
    if not all_filings:
        logging.error("No filings found for any company.")
        return
        
    combined_df = pd.concat(all_filings, ignore_index=True)
    
    # Aggregate daily
    # We want: date, sec_10k_count, sec_10q_count, sec_8k_count, sec_total_filings, and event flags per company
    dates = pd.date_range(start=args.start, end=args.end if args.end else pd.Timestamp.today().strftime('%Y-%m-%d'), freq='D').strftime('%Y-%m-%d')
    daily_df = pd.DataFrame({'date': dates})
    
    # Form counts
    form_counts = combined_df.groupby(['date', 'form']).size().unstack(fill_value=0).reset_index()
    # Ensure form columns exist
    for f in TARGET_FORMS:
        if f not in form_counts.columns:
            form_counts[f] = 0
            
    form_counts = form_counts.rename(columns={
        '10-K': 'sec_10k_count',
        '10-Q': 'sec_10q_count',
        '8-K': 'sec_8k_count'
    })
    form_counts['sec_total_filings'] = form_counts['sec_10k_count'] + form_counts['sec_10q_count'] + form_counts['sec_8k_count']
    
    daily_df = pd.merge(daily_df, form_counts, on='date', how='left').fillna(0)
    
    # Event flags
    # Which companies filed on which date?
    for ticker in TICKERS:
        col_name = f"{ticker.lower()}_filing_event"
        ticker_dates = combined_df[combined_df['ticker'] == ticker]['date'].unique()
        daily_df[col_name] = daily_df['date'].isin(ticker_dates).astype(int)
        
    # Cast count columns to integer to look cleaner
    count_cols = ['sec_10k_count', 'sec_10q_count', 'sec_8k_count', 'sec_total_filings']
    daily_df[count_cols] = daily_df[count_cols].astype(int)
        
    proc_path = proc_dir / "sec_filings_daily.csv"
    daily_df.to_csv(proc_path, index=False)
    
    logging.info("--- SUMMARY: SEC FILINGS ---")
    logging.info(f"Days processed: {len(daily_df)}")
    logging.info(f"Total Filings Found: {len(combined_df)}")
    logging.info(f"Processed Output: {proc_path}")
    logging.info("----------------------------\n")

if __name__ == "__main__":
    main()
