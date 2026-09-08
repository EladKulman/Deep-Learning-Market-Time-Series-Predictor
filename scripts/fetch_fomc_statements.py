#!/usr/bin/env python3
"""Fetch every FOMC policy statement since 2006 and build a daily FOMC feature file.

Why this rewrite
----------------
The previous scraper required the word "statement" in the link text, which the current
Fed calendar page no longer uses, and it only matched the post-2011 URL pattern. The
result was one statement per year after 2020 and nothing before 2011.

Collect statement links in both old and new URL formats, including policy releases
with a 'b' suffix. Verify the text describes the Committee's federal funds rate policy.
Use the printed release date when a historical URL contains a typo. Cache exclusions
as well as policy statements so offline replay never silently drops an unknown link.

Outputs
-------
* data/raw/fomc/fomc_statements_raw.jsonl : date, url, full text (input for tone scoring)
* data/processed/fomc_events_daily.csv     : calendar-day file with
    is_fomc_day, fomc_statement_length, fomc_hawkish_keyword_count,
    fomc_dovish_keyword_count, days_since_fomc

Timing note: statements are released at 14:00 ET (14:15 before March 2013), i.e. before
the close, so the day-T flag is legitimately known at the day-T close.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URLS = ["/monetarypolicy/fomccalendars.htm"] + [
    f"/monetarypolicy/fomchistorical{year}.htm" for year in range(2006, 2021)
]
LINK_PATTERNS = [
    re.compile(r'href="(/newsevents/pressreleases/monetary(\d{8})([a-z])\.htm)"'),
    re.compile(r'href="(/newsevents/press/monetary/(\d{8})([a-z])\.htm)"'),
]

HAWKISH_KEYWORDS = ["inflation", "tightening", "restrictive", "rate hike", "elevated inflation", "price stability"]
DOVISH_KEYWORDS = ["slowdown", "unemployment", "easing", "rate cut", "accommodative", "downside risks"]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "qqq-research-pipeline (academic project)"})


class ArticleText(HTMLParser):
    """Collect paragraph text, preferring the div with id="article" when present."""

    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.article_depth = 0
        self.in_p = False
        self.article_parts: list[str] = []
        self.all_parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "div":
            if attrs.get("id") == "article":
                self.in_article = True
                self.article_depth = 1
            elif self.in_article:
                self.article_depth += 1
        if tag == "p":
            self.in_p = True

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag == "div" and self.in_article:
            self.article_depth -= 1
            if self.article_depth == 0:
                self.in_article = False
        if tag == "p":
            self.in_p = False

    def handle_data(self, data):
        if self.skip or not self.in_p:
            return
        text = data.strip()
        if not text:
            return
        self.all_parts.append(text)
        if self.in_article:
            self.article_parts.append(text)

    def text(self) -> str:
        parts = self.article_parts or self.all_parts
        return " ".join(parts)


def get(url: str) -> str:
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def collect_statement_links(cache_dir: Path | None = None, cached: bool = False) -> dict[str, str]:
    """Collect statement links regardless of URL suffix; some policy releases use 'b'."""
    links: dict[str, str] = {}
    for path in CALENDAR_URLS:
        url = BASE_URL + path
        cache_path = None if cache_dir is None else cache_dir / Path(path).name
        if cached:
            if cache_path is None or not cache_path.exists():
                raise FileNotFoundError(cache_path)
            html = cache_path.read_text()
        else:
            html = get(url)  # Fail instead of silently publishing an incomplete inventory.
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(html)
        found = 0
        for anchor in BeautifulSoup(html, "html.parser").find_all("a", href=True):
            if anchor.get_text(" ", strip=True) not in {"Statement", "HTML"}:
                continue
            for pattern in LINK_PATTERNS:
                match = pattern.search(f'href="{anchor["href"]}"')
                if not match:
                    continue
                href, ymd, suffix = match.groups()
                date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
                links.setdefault(date, BASE_URL + href)
                found += 1
        logging.info("%s: %d statement links", path, found)
        time.sleep(0.3)
    return dict(sorted(links.items()))


def release_date(text: str, fallback: str) -> str:
    """The printed release date wins over a typo in a URL (e.g. June 28, 2007)."""
    match = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", text[:300])
    return pd.Timestamp(match.group()).strftime("%Y-%m-%d") if match else fallback


def is_policy_statement(text: str) -> bool:
    """Policy statements always set or reaffirm the federal funds rate target.

    Since April 2020 the statement body refers to "the Committee" rather than spelling out
    "Federal Open Market Committee", so only the funds-rate wording is required. The
    Statement on Longer-Run Goals (revised at the 2025 Jackson Hole meeting, among others)
    is published under the same URL pattern and is excluded by its title.
    """
    lowered = text.lower()
    if "longer-run goals and monetary policy strategy" in lowered[:400]:
        return False
    return "committee" in lowered and "federal funds rate" in lowered


def count_keywords(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(kw) for kw in keywords)


def fetch_statements(links: dict[str, str], cache_path: Path, refresh: bool, cached_only: bool = False) -> list[dict]:
    if cached_only and refresh:
        raise ValueError("--cached and --refresh cannot be combined")
    cached: dict[str, dict] = {}
    rejected_path = cache_path.with_name("fomc_nonpolicy_releases.json")
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() and not refresh else {}
    if cache_path.exists() and not refresh:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cached[record["url"]] = record

    records: list[dict] = []
    for date, url in links.items():
        if url in rejected:
            continue
        if url in cached and cached[url].get("text"):
            record = cached[url].copy()
            record["date"] = release_date(record["text"], date)
            record["title"] = f"FOMC Statement {record['date']}"
            records.append(record)
            continue
        if cached_only:
            raise ValueError(f"Statement not cached: {url}")
        html = get(url)  # A failed request must not replace the inventory with a partial one.
        parser = ArticleText()
        parser.feed(html)
        text = parser.text()
        if not is_policy_statement(text):
            if len(text) < 100:
                raise ValueError(f"No article text parsed from {url}")
            rejected[url] = {"date": release_date(text, date), "text": text,
                             "reason": "Does not satisfy is_policy_statement"}
            rejected_path.write_text(json.dumps(rejected, indent=2) + "\n")
            logging.info("Skipping non-policy release %s (%s)", date, url)
            continue
        date = release_date(text, date)
        records.append({"date": date, "title": f"FOMC Statement {date}", "url": url, "text": text})
        logging.info("Fetched statement %s (%d chars)", date, len(text))
        time.sleep(0.5)

    records.sort(key=lambda r: r["date"])
    with cache_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def build_daily(records: list[dict], start: str, end: str) -> pd.DataFrame:
    statements = pd.DataFrame(records)
    statements["fomc_statement_length"] = statements["text"].str.len()
    statements["fomc_hawkish_keyword_count"] = statements["text"].apply(lambda t: count_keywords(t, HAWKISH_KEYWORDS))
    statements["fomc_dovish_keyword_count"] = statements["text"].apply(lambda t: count_keywords(t, DOVISH_KEYWORDS))
    statements["is_fomc_day"] = 1
    statements = statements[
        ["date", "is_fomc_day", "fomc_statement_length", "fomc_hawkish_keyword_count", "fomc_dovish_keyword_count"]
    ]

    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")})
    daily = days.merge(statements, on="date", how="left")
    daily["is_fomc_day"] = daily["is_fomc_day"].fillna(0).astype(int)
    for column in ("fomc_statement_length", "fomc_hawkish_keyword_count", "fomc_dovish_keyword_count"):
        daily[column] = daily[column].ffill()  # value of the most recent statement; NaN before the first one

    last_fomc = pd.Series(pd.to_datetime(daily["date"]).where(daily["is_fomc_day"] == 1)).ffill()
    daily["days_since_fomc"] = (pd.to_datetime(daily["date"]) - last_fomc).dt.days
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FOMC statements")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--refresh", action="store_true", help="Re-download statements already cached")
    parser.add_argument("--cached", action="store_true", help="Rebuild entirely from the saved calendars and statements")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw" / "fomc"
    proc_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    links = collect_statement_links(out_dir / "raw" / "calendar", args.cached)
    logging.info("Found %d candidate statement links (%s -> %s)", len(links), min(links), max(links))
    records = fetch_statements(links, raw_dir / "fomc_statements_raw.jsonl", args.refresh, args.cached)
    records = [r for r in records if r["date"] >= args.start]
    if not records:
        raise SystemExit("No FOMC statements fetched")

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    daily = build_daily(records, args.start, end)
    path = proc_dir / "fomc_events_daily.csv"
    daily.to_csv(path, index=False)

    per_year = pd.Series([r["date"][:4] for r in records]).value_counts().sort_index()
    logging.info("Statements per year:\n%s", per_year.to_string())
    logging.info("Wrote %s (%d days, %d statements)", path, len(daily), len(records))


if __name__ == "__main__":
    main()
