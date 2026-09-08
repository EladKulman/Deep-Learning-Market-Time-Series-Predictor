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
        raise RuntimeError(f"GDELT returned no timeline: {mode}, {start} -> {end}")

    rows = {}
    for point in timeline[0].get("data", []):
        digits = "".join(ch for ch in point.get("date", "") if ch.isdigit())[:8]
        if len(digits) < 8:
            continue
        date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        entry = rows.setdefault(date, {"value": 0.0, "norm": 0.0, "n": 0})
        entry["value"] += float(point["value"]) if point.get("value") is not None else float("nan")
        entry["norm"] += float(point.get("norm") or 0)
        entry["n"] += 1

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "date"
    if frame.empty:
        raise RuntimeError(f"GDELT returned no dated observations: {mode}, {start} -> {end}")
    if mode == "timelinetone" and not frame.empty:
        frame["value"] = frame["value"] / frame["n"]  # average if multiple points per day
    return frame.sort_index()


def merge_cache(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Update covered observations without deleting older history on a partial response."""
    return new.combine_first(old).sort_index()


def refresh_topic(topic: str, query: str, cache_dir: Path, start: str, end: str,
                  refresh: bool, overlap_days: int, pacing: float) -> pd.DataFrame:
    path = cache_dir / f"{topic}.csv"
    old = pd.read_csv(path, dtype={"date": str}).set_index("date") if path.exists() else pd.DataFrame()
    fetch_start = start
    if not old.empty and not refresh:
        covered = old.index[old["norm"].fillna(0) > 0]
        if len(covered):
            fetch_start = max(start, (pd.Timestamp(max(covered)) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d"))
    if fetch_start > end:
        return old
    frames = {}
    for mode in ("timelinevolraw", "timelinetone"):
        # Persist each successful request so interruption/rate limiting does not discard it.
        part = cache_dir / f"{topic}_{mode}_{fetch_start}_{end}.csv"
        if part.exists() and not refresh:
            frames[mode] = pd.read_csv(part, dtype={"date": str}).set_index("date")
        else:
            logging.info("%s: %s %s -> %s", topic, mode, fetch_start, end)
            frame = fetch_timeline(query, mode, fetch_start, end)
            frame.to_csv(part)
            frames[mode] = frame
            if pacing:
                time.sleep(pacing)
    counts, tone = frames["timelinevolraw"], frames["timelinetone"]
    new = pd.DataFrame({"count": counts["value"], "norm": counts["norm"], "tone": tone["value"]})
    new = new.where(new["norm"].fillna(0) > 0)
    merged = merge_cache(old, new)
    merged.index.name = "date"
    merged.to_csv(path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT daily volume+tone per topic")
    parser.add_argument("--start", type=valid_date, default="2017-01-01")
    parser.add_argument("--end", type=valid_date, default=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--refresh", action="store_true", help="Re-query the requested range, including historical outages")
    parser.add_argument("--cached", action="store_true", help="Rebuild processed output from completed topic caches without requests")
    parser.add_argument("--topics", nargs="+", choices=sorted(TOPICS), help="Refresh only these topics; reuse other saved topics")
    parser.add_argument("--overlap-days", type=int, default=7, help="Revisit recent cached days when extending coverage")
    parser.add_argument("--pacing-seconds", type=float, default=PACING_SLEEP)
    args = parser.parse_args()

    if args.start < GDELT_MIN_DATE:
        logging.warning("GDELT DOC API starts %s; clamping start date.", GDELT_MIN_DATE)
        args.start = GDELT_MIN_DATE
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")
    if args.overlap_days < 0 or args.pacing_seconds < 0:
        raise SystemExit("Overlap and pacing must be non-negative")
    if args.cached and args.refresh:
        raise SystemExit("--cached and --refresh cannot be combined")

    proc_dir = Path(args.out_dir) / "processed"
    cache_dir = Path(args.out_dir) / "raw" / "gdelt_timeline"
    proc_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_start, output_end = args.start, args.end
    existing_output = proc_dir / "gdelt_topic_daily.csv"
    if existing_output.exists():
        previous_dates = pd.read_csv(existing_output, usecols=["date"])["date"]
        output_start = min(output_start, previous_dates.min())
        output_end = max(output_end, previous_dates.max())
    full_idx = pd.date_range(output_start, output_end, freq="D").strftime("%Y-%m-%d")
    out = pd.DataFrame(index=pd.Index(full_idx, name="date"))

    n_topics = len(TOPICS)
    failed = []
    for i, (topic, query) in enumerate(TOPICS.items(), 1):
        logging.info("[%d/%d] Updating %s", i, n_topics, topic)
        cached = None
        if not args.cached and (args.topics is None or topic in args.topics):
            try:
                cached = refresh_topic(topic, query, cache_dir, args.start, args.end,
                                       args.refresh, args.overlap_days, args.pacing_seconds)
            except RuntimeError as exc:
                failed.append(topic)
                logging.error("%s failed; retaining cached observations: %s", topic, exc)
        if cached is None:
            cached = pd.read_csv(cache_dir / f"{topic}.csv", dtype={"date": str}).set_index("date")

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
    if failed:
        raise SystemExit(f"Incomplete GDELT refresh: {', '.join(failed)}. Successful requests are cached; retry these topics.")


if __name__ == "__main__":
    main()
