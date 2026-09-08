#!/usr/bin/env python3
"""Report source coverage and enforce freshness before expensive model runs."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from chronos_data import fingerprint


def session_window(cutoff: pd.Timestamp, horizon: int, as_of=None):
    """Use completed exchange sessions, including holidays and early closes."""
    instant = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    instant = instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")
    exchange = mcal.get_calendar("NYSE")
    recent = exchange.schedule(start_date=instant.date() - pd.Timedelta(days=14), end_date=instant.date())
    completed = recent.loc[recent.market_close <= instant]
    expected_end = completed.index[-1]
    upcoming = exchange.schedule(start_date=cutoff + pd.Timedelta(days=1),
                                 end_date=cutoff + pd.Timedelta(days=max(30, horizon * 3)))
    return instant, expected_end, upcoming.index[:horizon]


def calendar_horizon(calendar: pd.DataFrame, names: list[str], dates: pd.DatetimeIndex):
    """Reindex exact sessions so a missing date cannot be replaced by a later row."""
    if calendar.date.duplicated().any() or not set(names) <= set(calendar):
        return None
    future = calendar.set_index("date").reindex(dates)
    if future[names].isna().any().any():
        return None
    return future.reset_index(names="date")


def check_build_snapshot(input_dir: Path) -> list[str]:
    path = input_dir / "daily_feature_table.metadata.json"
    if not path.exists():
        return ["table: missing build metadata; rebuild the feature table"]
    metadata = json.loads(path.read_text())
    expected = {"daily_feature_table.csv": metadata["table_sha256"],
                "daily_feature_table_groups.json": metadata["groups_sha256"],
                **metadata["source_sha256"]}
    changed = [name for name, digest in expected.items()
               if not (input_dir / name).exists() or fingerprint(input_dir / name) != digest]
    return [f"table: source/build snapshot changed ({', '.join(changed)}); rebuild the feature table"] if changed else []


def inspect(input_dir: Path, horizon: int = 5, as_of=None) -> dict:
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    table = pd.read_csv(input_dir / "daily_feature_table.csv", parse_dates=["date"])
    groups = json.loads((input_dir / "daily_feature_table_groups.json").read_text())
    cutoff = table.date.max()
    instant, expected_end, future_dates = session_window(cutoff, horizon, as_of)
    sources = {
        "prices": ("qqq_ohlcv_daily.csv", "adj_close", 0),
        "volatility": ("cboe_vol_daily.csv", "vxn_close", 1),
        "cross_asset": ("cross_asset_daily.csv", "spy_log_ret", 0),
        "macro": ("fred_macro_daily.csv", "dgs2", 1),
        "uncertainty": ("epu_daily.csv", "epu_daily", 1),
        "news": ("gdelt_topic_daily.csv", "ai_news_share", 1),
        "sec": ("sec_filings_daily.csv", "sec_total_filings", 0),
        "fed": ("fomc_tone_daily.csv", "fomc_net_hawkish_last", 0),
    }
    rows = []
    failures = check_build_snapshot(input_dir)
    if cutoff != expected_end:
        failures.append(f"prices: table ends {cutoff.date()}, latest completed NYSE session is {expected_end.date()}")
    if table.date.isna().any() or table.date.duplicated().any() or not table.date.is_monotonic_increasing:
        failures.append("table: dates must be valid, unique and increasing")
    for name, (filename, value, lag) in sources.items():
        path = input_dir / filename
        if not path.exists():
            failures.append(f"{name}: missing {filename}")
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        if value not in frame:
            failures.append(f"{name}: missing {value} in {filename}")
            continue
        observed = frame.loc[frame[value].notna() & (frame.date <= cutoff), "date"]
        latest = observed.max()
        # Allow one publication day and weekend/holiday gaps; report the exact lag too.
        stale = pd.isna(latest) or (cutoff - latest).days > max(4, lag + 3)
        entry = {"source": name, "file": filename, "first_observation": str(observed.min().date()),
                 "last_observation": str(latest.date()), "rows": len(frame), "stale": bool(stale),
                 "sha256": fingerprint(path)}
        if name not in {"fed", "macro"}:
            # Inspect every signal, not just a representative topic or ticker.
            values = frame.select_dtypes(include="number").columns
            per_column = {c: frame.loc[frame[c].notna() & (frame.date <= cutoff), "date"].max() for c in values}
            entry["per_column_last_observation"] = {c: str(d.date()) for c, d in per_column.items()}
            stale_columns = [c for c, d in per_column.items() if pd.isna(d) or (cutoff - d).days > 4]
            stale = stale or bool(stale_columns)
            entry["stale_columns"] = stale_columns
            entry["stale"] = bool(stale)
        if name == "news":
            entry["structural_limit"] = "DOC timeline history starts in 2017; missing source days remain NaN"
        if stale:
            failures.append(f"{name}: stale observations in {filename}; table ends {cutoff.date()} (see source detail)")
        rows.append(entry)
    future_names = [c for c, g in groups.items() if g["role"] == "known_future"]
    calendar = pd.read_csv(input_dir / "event_calendar_daily.csv", parse_dates=["date"])
    future = calendar_horizon(calendar, future_names, future_dates)
    if future is None or len(future) < horizon:
        failures.append("Known-future calendar does not cover the complete forecast horizon")
    statements = pd.read_json(input_dir.parent / "raw/fomc/fomc_statements_raw.jsonl", lines=True)
    scores = pd.read_csv(input_dir.parent / "raw/fomc/fomc_statement_scores.csv")
    if set(pd.to_datetime(statements.date)) != set(pd.to_datetime(scores.date)):
        failures.append("Fed tone scores do not match the statement inventory")
    tone_meta = input_dir / "fomc_tone_daily.metadata.json"
    if not tone_meta.exists():
        failures.append("fed: missing tone model/source provenance; rerun scoring")
    else:
        meta = json.loads(tone_meta.read_text())
        for name, digest in [(input_dir.parent / "raw/fomc/fomc_statements_raw.jsonl", meta["input_sha256"]),
                             (input_dir.parent / "raw/fomc/fomc_statement_scores.csv", meta["scores_sha256"]),
                             (input_dir / "fomc_tone_daily.csv", meta["daily_sha256"])]:
            if fingerprint(name) != digest:
                failures.append(f"fed: {name.name} changed since scoring; rerun scoring")
    return {"ready": not failures, "as_of_utc": instant.isoformat(),
            "latest_completed_session": str(expected_end.date()),
            "table_rows": len(table), "table_end": str(cutoff.date()),
            "table_sha256": fingerprint(input_dir / "daily_feature_table.csv"), "forecast_horizon": horizon,
            "future_dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "limitations": ["Snapshot freshness does not prove historical point-in-time availability",
                            "SEC/Fed calendar-day coverage includes zero/carried observations; inspect raw provenance too"],
            "sources": rows, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--as-of", help="UTC evaluation timestamp; defaults to now. Freeze this for historical snapshot replay.")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    parser.add_argument("--allow-stale-news", action="store_true", help="For infrastructure smoke tests only; records the exception")
    args = parser.parse_args()
    result = inspect(args.input_dir, args.horizon, args.as_of)
    if args.allow_stale_news:
        result["accepted_exceptions"] = [f for f in result["failures"] if f.startswith("news:")]
        result["failures"] = [f for f in result["failures"] if not f.startswith("news:")]
        result["ready"] = not result["failures"]
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
