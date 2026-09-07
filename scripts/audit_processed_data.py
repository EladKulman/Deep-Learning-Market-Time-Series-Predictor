#!/usr/bin/env python3
"""Audit every processed CSV: date coverage, missing values, dead columns, and leakage smells.

Run after any fetch or rebuild:

    python scripts/audit_processed_data.py            # all files in data/processed
    python scripts/audit_processed_data.py --leakage  # also regress each feature on the same-day return

The leakage check regresses each covariate in the feature table on the *same-day* QQQ log
return. A covariate is allowed to correlate with today's return if it is measured at the
close (VXN, TLT). A high fit for something that should only be known the next day (a
FRED yield, a news count) means a timestamp is off by one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAME_DAY_OK = {"overnight_gap", "parkinson_vol_1d", "qqq_minus_spy", "smh_minus_qqq", "iwm_minus_spy", "vxn_log_chg",
               "vxn_minus_vix", "vix_term_ratio", "vix9d_ratio", "vvix_log_chg", "tlt_log_ret", "hyg_log_ret",
               "gld_log_ret", "oil_log_ret", "dxy_log_ret", "d_dfii10_bp", "d_t5yie_bp", "volume_z_20d", "volume_change_1d",
               "mom_21d", "mom_63d", "dist_50dma", "dist_200dma", "volatility_20d", "parkinson_vol_22d", "vxn_close"}


def audit_file(path: Path) -> None:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        print(f"\n=== {path.name}: no date column, skipped")
        return
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    print(f"\n=== {path.name}: {len(frame):,} rows x {len(frame.columns)} cols, {frame['date'].min().date()} -> {frame['date'].max().date()}")
    numeric = frame.select_dtypes("number")
    missing = numeric.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("  missing share:", ", ".join(f"{c} {v:.0%}" for c, v in missing.head(12).items()), "..." if len(missing) > 12 else "")
    dead = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    if dead:
        print("  CONSTANT / all-zero columns:", dead)
    dup = frame["date"].duplicated().sum()
    if dup:
        print(f"  DUPLICATE dates: {dup}")


def leakage_check(path: Path) -> None:
    frame = pd.read_csv(path)
    if "log_return_1d" not in frame.columns:
        return
    target = frame["log_return_1d"]
    print(f"\n=== Same-day fit of each feature on log_return_1d ({path.name})")
    rows = []
    for column in frame.select_dtypes("number").columns:
        if column in ("log_return_1d",):
            continue
        pair = pd.concat([frame[column], target], axis=1).dropna()
        if len(pair) < 200 or pair[column].std() == 0:
            continue
        corr = np.corrcoef(pair[column], pair["log_return_1d"])[0, 1]
        rows.append((column, corr ** 2, column in SAME_DAY_OK))
    table = pd.DataFrame(rows, columns=["feature", "r2_same_day", "same_day_allowed"]).sort_values("r2_same_day", ascending=False)
    suspicious = table[(~table["same_day_allowed"]) & (table["r2_same_day"] > 0.02)]
    print(table.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(suspicious):
        print("\n  SUSPICIOUS (not measured at the close but fits today's return):")
        print(suspicious.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("\n  No next-day-only feature explains more than 2% of today's return. Good.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--leakage", action="store_true")
    args = parser.parse_args()
    for path in sorted(args.input_dir.glob("*.csv")):
        audit_file(path)
    if args.leakage:
        leakage_check(args.input_dir / "daily_feature_table.csv")


if __name__ == "__main__":
    main()
