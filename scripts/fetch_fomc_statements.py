#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "https://www.federalreserve.gov"

HAWKISH_KEYWORDS = ["inflation", "tightening", "restrictive", "rate hike", "elevated inflation", "price stability"]
DOVISH_KEYWORDS = ["slowdown", "unemployment", "easing", "rate cut", "accommodative", "downside risks"]

def fetch_statement_links():
    links = []
    # Fetch recent calendar
    calendars = [
        "/monetarypolicy/fomccalendars.htm"
    ]
    # Add historical years going back to 2006 to match the dataset bounds
    for year in range(2026, 2005, -1):
        calendars.append(f"/monetarypolicy/fomchistorical{year}.htm")
        
    for cal in calendars:
        url = BASE_URL + cal
        try:
            logging.info(f"Scraping FOMC calendar: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            
            # Find all links that might be a statement
            for a in soup.find_all('a', href=True):
                text = a.text.strip().lower()
                href = a['href']
                if "statement" in text and "/newsevents/pressreleases/monetary" in href:
                    if not href.startswith("http"):
                        href = BASE_URL + href
                    links.append(href)
        except Exception as e:
            logging.warning(f"Failed to scrape {url}: {e}")
            
    # Remove duplicates
    return list(set(links))

def fetch_statement_text(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Text is usually within an element with id="article"
        article = soup.find(id="article")
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            # Fallback
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text(strip=True) for p in paragraphs])
            
        return text
    except Exception as e:
        logging.warning(f"Failed to fetch statement text from {url}: {e}")
        return ""

def count_keywords(text, keywords):
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        # Simple string count
        count += text_lower.count(kw.lower())
    return count

def extract_date_from_url(url):
    # e.g., https://www.federalreserve.gov/newsevents/pressreleases/monetary20230726a.htm
    match = re.search(r'monetary(\d{8})[a-z]?\.htm', url)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None

def main():
    parser = argparse.ArgumentParser(description="Fetch FOMC statements")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "fomc"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    links = fetch_statement_links()
    
    raw_data = []
    
    for link in links:
        date_str = extract_date_from_url(link)
        if not date_str:
            continue
            
        logging.info(f"Fetching statement for {date_str}...")
        text = fetch_statement_text(link)
        if text:
            raw_data.append({
                "date": date_str,
                "title": f"FOMC Statement {date_str}",
                "url": link,
                "text": text
            })
        time.sleep(0.5) # Polite scraping
        
    if not raw_data:
        logging.error("No FOMC statements fetched.")
        return
        
    # Save raw
    raw_path = raw_dir / "fomc_statements_raw.jsonl"
    with open(raw_path, 'w', encoding='utf-8') as f:
        for r in raw_data:
            f.write(json.dumps(r) + "\n")
            
    # Process
    df = pd.DataFrame(raw_data)
    df = df.sort_values('date').reset_index(drop=True)
    
    # We DO NOT filter df here. We need all historical data to forward-fill properly.
        
    # Compute base metrics
    df['fomc_statement_length'] = df['text'].apply(len)
    df['fomc_hawkish_keyword_count'] = df['text'].apply(lambda x: count_keywords(x, HAWKISH_KEYWORDS))
    df['fomc_dovish_keyword_count'] = df['text'].apply(lambda x: count_keywords(x, DOVISH_KEYWORDS))
    df['is_fomc_day'] = 1
    df['last_fomc_statement_text'] = df['text']
    
    df = df[['date', 'is_fomc_day', 'last_fomc_statement_text', 'fomc_statement_length', 'fomc_hawkish_keyword_count', 'fomc_dovish_keyword_count']]
    
    # Create daily aggregate covering the entire history to allow forward-filling
    min_date = df['date'].min() if not df.empty else args.start
    start_dt = min(args.start, min_date) if args.start else min_date
    end_dt = args.end if args.end else pd.Timestamp.today().strftime('%Y-%m-%d')
    
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D').strftime('%Y-%m-%d')
    daily_df = pd.DataFrame({'date': dates})
    
    daily_df = pd.merge(daily_df, df, on='date', how='left')
    
    daily_df['is_fomc_day'] = daily_df['is_fomc_day'].fillna(0).astype(int)
    
    # Forward fill the features
    daily_df['last_fomc_statement_text'] = daily_df['last_fomc_statement_text'].ffill()
    daily_df['fomc_statement_length'] = daily_df['fomc_statement_length'].ffill().fillna(0)
    daily_df['fomc_hawkish_keyword_count'] = daily_df['fomc_hawkish_keyword_count'].ffill().fillna(0)
    daily_df['fomc_dovish_keyword_count'] = daily_df['fomc_dovish_keyword_count'].ffill().fillna(0)
    
    # Calculate days since fomc
    # We find indices of FOMC days
    fomc_indices = daily_df[daily_df['is_fomc_day'] == 1].index
    
    # If there are no FOMC days, we can't calculate days since.
    daily_df['days_since_fomc'] = pd.NA
    if not fomc_indices.empty:
        # A simple forward pass to count days
        days_since = 0
        has_seen_first = False
        for i in range(len(daily_df)):
            if daily_df.loc[i, 'is_fomc_day'] == 1:
                days_since = 0
                has_seen_first = True
            elif has_seen_first:
                days_since += 1
            
            if has_seen_first:
                daily_df.loc[i, 'days_since_fomc'] = days_since

    # Now filter the bounds for the final output
    if args.start:
        daily_df = daily_df[daily_df['date'] >= args.start]
    if args.end:
        daily_df = daily_df[daily_df['date'] <= args.end]

    # Drop the textual column so the final CSV is purely numeric for time-series modeling
    if 'last_fomc_statement_text' in daily_df.columns:
        daily_df = daily_df.drop(columns=['last_fomc_statement_text'])

    proc_path = proc_dir / "fomc_events_daily.csv"
    daily_df.to_csv(proc_path, index=False)
    
    logging.info("--- SUMMARY: FOMC ---")
    logging.info(f"Statements found: {len(df)}")
    if len(df) > 0:
        logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logging.info(f"Days processed: {len(daily_df)}")
    logging.info(f"Processed Output: {proc_path}")
    logging.info("---------------------\n")

if __name__ == "__main__":
    main()
