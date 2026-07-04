#!/usr/bin/env python3
"""Build the daily modeling table used by forecasting scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = {
    "prices": "qqq_ohlcv_daily.csv",
    "macro": "fred_macro_daily.csv",
    "vix": "vix_daily.csv",
    "fomc": "fomc_events_daily.csv",
    "sec": "sec_filings_daily.csv",
    "gdelt": "gdelt_topic_daily.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join processed daily sources into one QQQ modeling table."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed daily CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/daily_feature_table.csv"),
        help="Where to write the joined feature table.",
    )
    parser.add_argument(
        "--drop-missing-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows where log_return_1d is missing.",
    )
    return parser.parse_args()


def read_daily_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected input file: {path}")

    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} does not contain a 'date' column")

    frame = frame.drop(columns=["date_parsed"], errors="ignore")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame.drop_duplicates(subset=["date"], keep="last")


def fill_joined_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").reset_index(drop=True)

    zero_fill_patterns = (
        "_count",
        "_event",
        "is_",
        "days_since_",
        "_keyword_count",
        "_statement_length",
        "sec_total_filings",
    )
    zero_fill_columns = [
        column
        for column in frame.columns
        if column != "date" and column.startswith(zero_fill_patterns)
    ]
    zero_fill_columns.extend(
        [
            column
            for column in frame.columns
            if column.endswith(("_count", "_event", "_keyword_count"))
        ]
    )
    zero_fill_columns = sorted(set(zero_fill_columns) & set(frame.columns))
    frame[zero_fill_columns] = frame[zero_fill_columns].fillna(0)

    forward_fill_columns = [
        column
        for column in frame.columns
        if column not in {"date", "log_return_1d", "return_1d", "return_5d"}
        and column not in zero_fill_columns
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    frame[forward_fill_columns] = frame[forward_fill_columns].ffill()
    return frame


def build_feature_table(input_dir: Path, drop_missing_target: bool) -> pd.DataFrame:
    sources = {
        name: read_daily_csv(input_dir / file_name)
        for name, file_name in DEFAULT_INPUTS.items()
    }

    table = sources["prices"]
    for name in ("macro", "vix", "fomc", "sec", "gdelt"):
        table = table.merge(sources[name], on="date", how="left", validate="one_to_one")

    table = fill_joined_features(table)
    if drop_missing_target and "log_return_1d" in table.columns:
        table = table.dropna(subset=["log_return_1d"])

    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    return table.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    table = build_feature_table(
        input_dir=args.input_dir,
        drop_missing_target=args.drop_missing_target,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table):,} rows and {len(table.columns):,} columns to {args.output}")


if __name__ == "__main__":
    main()
