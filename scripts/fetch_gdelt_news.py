#!/usr/bin/env python3
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_DATE = "2017-01-01"
PACING_SLEEP = 240        # seconds between successful requests (your IP is throttled hard)
RATE_LIMIT_WAITS = [120, 240, 300, 300, 600, 600]  # backoff on 429 / bad responses

TOPICS = {
    "ai": '("artificial intelligence" OR "generative AI" OR "machine learning" OR OpenAI OR Nvidia)',
    "semiconductor": '(semiconductor OR chipmaker OR Nvidia OR AMD OR TSMC OR ASML OR Intel)',
    "fed": '("Federal Reserve" OR FOMC OR "interest rates" OR "rate hike" OR "rate cut")',
    "inflation": '(inflation OR CPI OR "consumer prices" OR "core inflation")',
    "big_tech_earnings": '(Apple OR Microsoft OR Amazon OR Alphabet OR Meta OR Nvidia) (earnings OR revenue OR guidance)',
    "recession": '(recession OR "hard landing" OR "economic contraction")',
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "qqq-research-pipeline (academic project)"})


def valid_date(value: str) -> str:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use full YYYY-MM-DD, e.g. 2026-07-01."
        )
    if len(value.strip()) < 10:
        raise argparse.ArgumentTypeError(
            f"Ambiguous date '{value}'. Use full YYYY-MM-DD, e.g. 2026-07-01."
        )
    return ts.strftime("%Y-%m-%d")


def fetch_timeline(query: str, mode: str, start: str, end: str) -> pd.DataFrame:
    """One request for the full range. Returns DataFrame indexed by date.

    Columns: 'value' (count or tone) and, for timelinevolraw, 'norm'
    (total articles GDELT monitored that day, useful for normalization).
    """
    params = {
        "query": f"{query} sourcelang:eng",
        "mode": mode,
        "format": "json",
        "timelinesmooth": 0,
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
    }
    for attempt, wait in enumerate([0] + RATE_LIMIT_WAITS):
        if wait:
            logging.warning("Backing off %ds before retry %d...", wait, attempt)
            time.sleep(wait)
        try:
            resp = SESSION.get(GDELT_API_URL, params=params, timeout=120)
            if resp.status_code == 429:
                logging.warning("429 Too Many Requests (%s)", mode)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except json.JSONDecodeError:
            logging.warning("Non-JSON response (likely soft rate limit).")
            continue
        except requests.RequestException as exc:
            logging.warning("Request error: %s", exc)
            continue
    else:
        raise RuntimeError(f"GDELT request failed after all retries: mode={mode}")

    timeline = data.get("timeline", [])
    if not timeline:
        return pd.DataFrame(columns=["value"])

    rows = {}
    for point in timeline[0].get("data", []):
        digits = "".join(ch for ch in point.get("date", "") if ch.isdigit())[:8]
        if len(digits) < 8:
            continue
        date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        entry = rows.setdefault(date, {"value": 0.0, "norm": 0.0, "n": 0})
        entry["value"] += float(point.get("value") or 0)
        entry["norm"] += float(point.get("norm") or 0)
        entry["n"] += 1

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "date"
    if mode == "timelinetone" and not frame.empty:
        frame["value"] = frame["value"] / frame["n"]  # average if multiple points per day
    return frame.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT daily volume+tone per topic")
    parser.add_argument("--start", type=valid_date, default="2017-01-01")
    parser.add_argument("--end", type=valid_date, default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    if args.start < GDELT_MIN_DATE:
        logging.warning("GDELT DOC API starts %s; clamping start date.", GDELT_MIN_DATE)
        args.start = GDELT_MIN_DATE
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")

    proc_dir = Path(args.out_dir) / "processed"
    cache_dir = Path(args.out_dir) / "raw" / "gdelt_timeline"
    proc_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    full_idx = pd.date_range(args.start, args.end, freq="D").strftime("%Y-%m-%d")
    out = pd.DataFrame(index=pd.Index(full_idx, name="date"))

    n_topics = len(TOPICS)
    for i, (topic, query) in enumerate(TOPICS.items(), 1):
        cache_path = cache_dir / f"{topic}.csv"
        if cache_path.exists():
            logging.info("[%d/%d] %s: cached (%s) - delete to refetch", i, n_topics, topic, cache_path)
            cached = pd.read_csv(cache_path, dtype={"date": str}).set_index("date")
        else:
            logging.info("[%d/%d] Topic: %s (2 requests, %ds apart)", i, n_topics, topic, PACING_SLEEP)
            counts = fetch_timeline(query, "timelinevolraw", args.start, args.end)
            logging.info("    volume: %d days", len(counts))
            time.sleep(PACING_SLEEP)
            tone = fetch_timeline(query, "timelinetone", args.start, args.end)
            logging.info("    tone:   %d days", len(tone))
            time.sleep(PACING_SLEEP)

            cached = pd.DataFrame({
                "count": counts.get("value"),
                "norm": counts.get("norm"),
                "tone": tone.get("value"),
            })
            cached.index.name = "date"
            cached.to_csv(cache_path)
            logging.info("    cached -> %s", cache_path)

        cached = cached.reindex(full_idx)
        # Days GDELT did not cover (API outages, the partial final day) have no
        # 'norm' (total monitored articles). Leave them NaN: Chronos-2 masks
        # missing values, and a zero would be a fake "no news" observation.
        covered = cached["norm"].fillna(0) > 0
        out[f"{topic}_news_count"] = cached["count"].where(covered)
        out[f"{topic}_avg_tone"] = cached["tone"].where(covered)
        # Share of all GDELT-monitored articles that day, which removes the
        # long-run growth in GDELT's source coverage from the raw counts.
        out[f"{topic}_news_share"] = (cached["count"] / cached["norm"]).where(covered)

    out = out.reset_index()
    path = proc_dir / "gdelt_topic_daily.csv"
    out.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows, %d columns)", path, len(out), len(out.columns))


if __name__ == "__main__":
    main()