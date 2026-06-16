#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import pandas as pd
import requests
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

TOPICS = {
    "ai": '("artificial intelligence" OR "generative AI" OR "machine learning" OR OpenAI OR Nvidia)',
    "semiconductor": '(semiconductor OR chipmaker OR Nvidia OR AMD OR TSMC OR ASML OR Intel)',
    "fed": '("Federal Reserve" OR Fed OR FOMC OR "interest rates" OR "rate hike" OR "rate cut")',
    "inflation": '(inflation OR CPI OR "consumer prices" OR "core inflation")',
    "big_tech_earnings": '(Apple OR Microsoft OR Amazon OR Google OR Alphabet OR Meta OR Nvidia) AND (earnings OR revenue OR guidance OR profit)',
    "recession": '(recession OR "hard landing" OR slowdown OR "economic contraction")'
}

def fetch_gdelt_day(topic_name, query, date_str):
    """
    date_str format: YYYY-MM-DD
    GDELT requires StartDateTime and EndDateTime in YYYYMMDDHHMMSS.
    """
    start_dt = date_str.replace("-", "") + "000000"
    end_dt = date_str.replace("-", "") + "235959"
    
    # Enforce English
    full_query = f"{query} sourcelang:eng"
    
    params = {
        "query": full_query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "maxrecords": 1
    }
    
    try:
        response = requests.get(GDELT_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return articles
    except requests.exceptions.JSONDecodeError:
        # GDELT sometimes returns empty body instead of valid JSON if no results
        return []
    except Exception as e:
        logging.error(f"Failed fetching {topic_name} on {date_str}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Fetch and process GDELT news")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=str, default="data", help="Base output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "gdelt"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    date_range = pd.date_range(start=args.start, end=args.end, freq='D')
    
    all_articles = []
    
    for topic_name, query in TOPICS.items():
        logging.info(f"--- Fetching Topic: {topic_name} ---")
        topic_articles = []
        
        for dt in date_range:
            date_str = dt.strftime("%Y-%m-%d")
            articles = fetch_gdelt_day(topic_name, query, date_str)
            
            for art in articles:
                art["topic"] = topic_name
                # Extract simple YYYY-MM-DD date
                seendate = art.get("seendate", "")
                if len(seendate) >= 8:
                    art["date"] = f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"
                else:
                    art["date"] = date_str
                    
            topic_articles.extend(articles)
            time.sleep(1) # Rate limit politely
            
        if topic_articles:
            # Save raw jsonl
            raw_path = raw_dir / f"gdelt_articles_{topic_name}.jsonl"
            with open(raw_path, 'w', encoding='utf-8') as f:
                for art in topic_articles:
                    f.write(json.dumps(art) + "\n")
            
            all_articles.extend(topic_articles)
            logging.info(f"Saved {len(topic_articles)} raw articles for {topic_name}")
        else:
            logging.warning(f"No articles found for {topic_name}")
            
    # Process
    if not all_articles:
        logging.error("No articles fetched for any topic. Exiting.")
        return
        
    df = pd.DataFrame(all_articles)
    
    # Required raw columns: topic, date, title, url, domain, language, source_country, seendate, tone_if_available
    df = df.rename(columns={"sourcecountry": "source_country"})
    if "tone" not in df.columns:
        df["tone_if_available"] = float('nan')
    else:
        df["tone_if_available"] = df["tone"]
        
    cols = ['topic', 'date', 'title', 'url', 'domain', 'language', 'source_country', 'seendate', 'tone_if_available']
    # Keep only available cols
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])
    
    # Aggregate to daily features
    # Required: date, <topic>_news_count, <topic>_avg_tone
    aggs = {}
    for topic_name in TOPICS.keys():
        topic_df = df[df['topic'] == topic_name]
        counts = topic_df.groupby('date').size().rename(f'{topic_name}_news_count')
        aggs[f'{topic_name}_news_count'] = counts
        aggs[f'{topic_name}_avg_tone'] = pd.Series(float('nan'), index=counts.index, name=f'{topic_name}_avg_tone')
        
    agg_df = pd.DataFrame(aggs)
    agg_df.index.name = 'date'
    agg_df = agg_df.reset_index()
    
    # Ensure all days in range exist
    full_idx = pd.date_range(start=args.start, end=args.end, freq='D').strftime('%Y-%m-%d')
    agg_df = agg_df.set_index('date').reindex(full_idx).fillna(0).reset_index()
    agg_df = agg_df.rename(columns={'index': 'date'})
    
    # For _avg_tone columns, set missing to NaN instead of 0
    for c in agg_df.columns:
        if c.endswith('_avg_tone'):
            agg_df[c] = agg_df[c].replace(0, float('nan'))
            
    proc_path = proc_dir / "gdelt_topic_daily.csv"
    agg_df.to_csv(proc_path, index=False)
    
    logging.info("--- SUMMARY: GDELT ---")
    logging.info(f"Days processed: {len(agg_df)}")
    logging.info(f"Total Unique Articles: {len(df)}")
    for topic_name in TOPICS.keys():
        cnt = agg_df[f'{topic_name}_news_count'].sum()
        logging.info(f"  {topic_name}: {int(cnt)} articles")
    logging.info(f"Processed Output: {proc_path}")
    logging.info("----------------------\n")

if __name__ == "__main__":
    main()
