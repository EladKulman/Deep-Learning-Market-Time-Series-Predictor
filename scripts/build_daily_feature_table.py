#!/usr/bin/env python3
"""Build the daily modeling table used by the Chronos-2 scripts.

Design (see docs/DATA_NOTES.md and the covariate brief)
--------------------------------------------------------
* One row per QQQ trading day; the target is ``log_return_1d``.
* Every covariate is attached by the time it became knowable, not its reference date:
  FRED rates use their publication sessions; CBOE, oil and dollar bars are lagged one
  day; EPU and GDELT use completed days. The SEC fetcher attaches all filings to the first NYSE close after acceptance.
* Levels that drift (rates, spreads, volatility indices, prices) are differenced here;
  bounded ratios, trailing z-scores and regime levels are kept.
* Nothing is zero-filled. Where a source does not exist yet (GDELT before 2017, VIX3M
  before Dec 2007) the column is NaN and Chronos-2 masks it.
* ``feature_groups.json`` is written next to the table: it maps every column to a group and
  a role (target / past / known_future / raw) so the fine-tuning and ablation scripts can
  select feature sets by name instead of hard-coding columns.

``--profile core`` writes the compact set recommended in the brief (21 past-only
covariates plus the calendar flags); ``--profile full`` (default) writes every group.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUTS = {
    "prices": "qqq_ohlcv_daily.csv",
    "vol": "cboe_vol_daily.csv",
    "cross": "cross_asset_daily.csv",
    "fred": "fred_macro_daily.csv",
    "epu": "epu_daily.csv",
    "gdelt": "gdelt_topic_daily.csv",
    "fomc": "fomc_events_daily.csv",
    "tone": "fomc_tone_daily.csv",
    "sec": "sec_filings_daily.csv",
    "calendar": "event_calendar_daily.csv",
}
OPTIONAL = {"vol", "cross", "epu", "gdelt", "tone", "sec", "calendar"}

# Keep legacy suffixes stable; actual alignment uses release dates below.
FRED_LAG_ONE_DAY = ["dff", "dgs10", "dgs2", "t10y2y"]
FRED_SAME_DAY = ["cpi", "cpi_yoy"]
GDELT_TOPICS = ["ai", "semiconductor", "fed", "inflation", "big_tech_earnings", "recession"]

CORE_PAST = [
    "overnight_gap", "parkinson_vol_1d", "parkinson_vol_22d", "volume_z_20d", "mom_21d", "dist_200dma",
    "vxn_log_chg", "vxn_minus_vix", "vix_term_ratio", "tlt_log_ret", "hyg_log_ret",
    "d_dgs2_bp", "t10y2y_lag1", "d_dfii10_bp", "cpi_yoy",
    "epu_log", "emu_log", "fomc_net_hawkish_ewma", "gdelt_fed_news_share", "gdelt_ai_news_share",
]
CORE_PAST.append("ndx_earnings_count")
CORE_FUTURE = ["is_fomc_day", "is_cpi_day", "is_nfp_day", "days_to_fomc", "days_to_month_end", "is_opex_week"]


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    sources = {}
    for name, file_name in INPUTS.items():
        path = input_dir / file_name
        if path.exists():
            sources[name] = read_csv(path)
        elif name in OPTIONAL:
            logging.warning("Optional input missing, skipping: %s", path)
        else:
            raise FileNotFoundError(path)
    return sources


def asof_join(base: pd.DataFrame, other: pd.DataFrame, columns: list[str], lag_days: int = 0, suffix: str = "") -> pd.DataFrame:
    """Attach the latest value of `columns` whose availability date is <= trading day."""
    columns = [c for c in columns if c in other.columns]
    if not columns:
        return base
    base = base.sort_values("date")
    # One as-of merge per column, dropping that column's NaN rows first: calendar-day source
    # files have blank weekend rows, and a joint merge would hand Monday the blank Sunday row
    # instead of Friday's value.
    for column in columns:
        right = other[["date", column]].dropna(subset=[column]).copy()
        right["avail"] = right["date"] + pd.Timedelta(days=lag_days)
        # Several reference days may share an availability session (e.g. Good Friday).
        # Preserve reference-date order so merge_asof chooses the latest observation.
        right = right.drop(columns=["date"]).sort_values("avail", kind="stable")
        if suffix:
            right = right.rename(columns={column: f"{column}{suffix}"})
        base = pd.merge_asof(base, right, left_on="date", right_on="avail", direction="backward").drop(columns=["avail"])
    return base


def window_mean_join(base: pd.DataFrame, other: pd.DataFrame, columns: list[str], lag_days: int = 1) -> pd.DataFrame:
    """For calendar-day series: mean of the values that became available since the prior trading day."""
    columns = [c for c in columns if c in other.columns]
    if not columns:
        return base
    right = other[["date", *columns]].copy()
    right["avail"] = right["date"] + pd.Timedelta(days=lag_days)
    trading = base["date"].sort_values().to_numpy()
    idx = np.searchsorted(trading, right["avail"].to_numpy(), side="left")
    right = right[idx < len(trading)]
    right["trading_date"] = trading[idx[idx < len(trading)]]
    agg = right.groupby("trading_date")[columns].mean()
    return base.merge(agg, left_on="date", right_index=True, how="left")


def fred_available_dates(dates: pd.Series, spread: bool = False) -> pd.Series:
    """Conservative first NYSE close after the source's scheduled publication.

    H.15 reports the prior business day's observations at 16:15 ET. Since 2019-06-21,
    FRED spreads use direct Treasury data and update the same evening. Both are after
    the NYSE close, including early closes. Federal holidays can differ from NYSE ones.
    This schedule reconstruction is not a historical archive of exceptional delays.
    """
    references = pd.DatetimeIndex(dates)
    end = references.max() + pd.Timedelta(days=30)
    fed_days = pd.date_range(references.min(), end,
                            freq=pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    releases = pd.Series(fed_days[fed_days.searchsorted(references, side="right")], index=dates.index)
    if spread:
        direct = dates >= pd.Timestamp("2019-06-21")
        releases.loc[direct] = dates.loc[direct]
    nyse = mcal.get_calendar("NYSE").schedule(start_date=references.min(), end_date=end).index
    return pd.Series(nyse[nyse.searchsorted(releases, side="right")], index=dates.index)


def qqq_features(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    close, open_, high, low = p["close"], p["open"], p["high"], p["low"]
    p["overnight_gap"] = np.log(open_ / close.shift(1))
    p["parkinson_vol_1d"] = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
    p["parkinson_vol_22d"] = p["parkinson_vol_1d"].rolling(22, min_periods=15).mean()
    log_volume = np.log(p["volume"].replace(0, np.nan))
    p["volume_z_20d"] = (log_volume - log_volume.rolling(20, min_periods=15).mean()) / log_volume.rolling(20, min_periods=15).std()
    p["mom_21d"] = np.log(close / close.shift(21))
    p["mom_63d"] = np.log(close / close.shift(63))
    p["dist_50dma"] = np.log(close / close.rolling(50, min_periods=40).mean())
    p["dist_200dma"] = np.log(close / close.rolling(200, min_periods=150).mean())
    return p.drop(columns=[c for c in ("return_1d", "return_5d") if c in p.columns])


def build(input_dir: Path, profile: str, start: str = "2006-01-01") -> tuple[pd.DataFrame, dict]:
    src = load_inputs(input_dir)
    table = qqq_features(src["prices"])
    groups: dict[str, dict] = {"log_return_1d": {"group": "target", "role": "target"}}
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        groups[column] = {"group": "raw", "role": "raw"}
    for column in ("overnight_gap", "parkinson_vol_1d", "parkinson_vol_22d", "volatility_20d", "volume_z_20d",
                   "volume_change_1d", "mom_21d", "mom_63d", "dist_50dma", "dist_200dma"):
        groups[column] = {"group": "qqq", "role": "past"}

    # CBOE daily closes may be finalized after the 16:00 ET forecast cutoff. Lag the
    # entire observation, including ratios and FRED backfills, consistently by one day.
    if "vol" in src:
        vol = src["vol"]
        table = asof_join(table, vol, ["vxn_close", "vix_close", "vxn_minus_vix", "vix_term_ratio", "vix9d_ratio", "vvix"], lag_days=1)
        table["vxn_log_chg"] = np.log(table["vxn_close"]).diff()
        table["vvix_log_chg"] = np.log(table["vvix"]).diff()
        table = table.drop(columns=["vix_close", "vvix"])
        for column in ("vxn_close", "vxn_log_chg", "vxn_minus_vix", "vix_term_ratio", "vix9d_ratio", "vvix_log_chg"):
            groups[column] = {"group": "market", "role": "past"}

    # --- cross-asset (same-day close) ---
    if "cross" in src:
        table = asof_join(table, src["cross"], ["tlt_log_ret", "hyg_log_ret", "spy_log_ret", "iwm_log_ret", "smh_log_ret", "gld_log_ret"])
        # Yahoo's daily commodity/FX bar cutoff is not guaranteed to precede the NYSE close.
        table = asof_join(table, src["cross"], ["oil_log_ret", "dxy_log_ret"], lag_days=1)
        table["qqq_minus_spy"] = table["log_return_1d"] - table["spy_log_ret"]
        table["iwm_minus_spy"] = table["iwm_log_ret"] - table["spy_log_ret"]
        table["smh_minus_qqq"] = table["smh_log_ret"] - table["log_return_1d"]
        table = table.drop(columns=["spy_log_ret", "iwm_log_ret", "smh_log_ret"])
        for column in ("tlt_log_ret", "hyg_log_ret", "qqq_minus_spy", "iwm_minus_spy", "smh_minus_qqq", "gld_log_ret", "oil_log_ret", "dxy_log_ret"):
            groups[column] = {"group": "market", "role": "past"}

    # --- FRED: release-aware H.15 / spreads, vintaged CPI ---
    fred = src["fred"]
    for columns, spread in [(["dff", "dgs10", "dgs2", "dfii10"], False), (["t10y2y", "t5yie"], True)]:
        published = fred.copy()
        published["date"] = fred_available_dates(fred["date"], spread=spread)
        for column in columns:
            suffix = "_lag1" if column in FRED_LAG_ONE_DAY else ""
            table = asof_join(table, published, [column], suffix=suffix)
    table = asof_join(table, fred, FRED_SAME_DAY)
    for column in FRED_LAG_ONE_DAY:
        lagged = f"{column}_lag1"
        if lagged in table.columns:
            table[f"d_{column}_bp"] = table[lagged].diff() * 100
            groups[f"d_{column}_bp"] = {"group": "rates", "role": "past"}
    for column in ("dfii10", "t5yie"):
        if column in table.columns:
            table[f"d_{column}_bp"] = table[column].diff() * 100
            groups[f"d_{column}_bp"] = {"group": "rates", "role": "past"}
            table = table.drop(columns=[column])
    keep_levels = {"t10y2y_lag1": "rates", "cpi_yoy": "rates"}
    for column, group in keep_levels.items():
        if column in table.columns:
            groups[column] = {"group": group, "role": "past"}
    table = table.drop(columns=[c for c in ("dff_lag1", "dgs10_lag1", "dgs2_lag1", "cpi") if c in table.columns])

    # --- uncertainty and news: calendar days, available the next morning ---
    if "epu" in src:
        # Single-day EPU values are dominated by weekend days with few articles (the raw
        # maxima are all Saturdays and Sundays), so the primary series is the trailing
        # 7-calendar-day mean; the raw day is kept as a secondary column.
        epu = src["epu"].copy()
        for column in ("epu_daily", "emu_daily"):
            epu[f"{column}_7d"] = epu[column].rolling(7, min_periods=4).mean()
        table = window_mean_join(table, epu, ["epu_daily", "emu_daily", "epu_daily_7d", "emu_daily_7d"], lag_days=1)
        table["epu_log"] = np.log(table["epu_daily_7d"])
        table["emu_log"] = np.log(table["emu_daily_7d"])
        table["epu_log_1d"] = np.log(table["epu_daily"])
        table["emu_log_1d"] = np.log(table["emu_daily"])
        table = table.drop(columns=["epu_daily", "emu_daily", "epu_daily_7d", "emu_daily_7d"])
        for column in ("epu_log", "emu_log", "epu_log_1d", "emu_log_1d"):
            groups[column] = {"group": "uncertainty", "role": "past"}
    if "gdelt" in src:
        columns = [f"{t}_news_share" for t in GDELT_TOPICS] + [f"{t}_avg_tone" for t in GDELT_TOPICS]
        table = window_mean_join(table, src["gdelt"], columns, lag_days=1)
        table = table.rename(columns={c: f"gdelt_{c}" for c in columns if c in table.columns})
        for column in columns:
            if f"gdelt_{column}" in table.columns:
                groups[f"gdelt_{column}"] = {"group": "news", "role": "past"}

    # --- Fed tone (14:00 ET, before the close) ---
    if "tone" in src:
        tone = src["tone"].copy()
        tone["fomc_tone_change_last"] = tone["fomc_tone_change"].ffill()
        table = asof_join(table, tone, ["fomc_net_hawkish_last", "fomc_net_hawkish_ewma", "fomc_tone_change_last"])
        for column in ("fomc_net_hawkish_last", "fomc_net_hawkish_ewma", "fomc_tone_change_last"):
            groups[column] = {"group": "fed", "role": "past"}
    table = asof_join(table, src["fomc"], ["days_since_fomc"])
    groups["days_since_fomc"] = {"group": "fed", "role": "past"}

    # SEC source is already aligned to the first actual NYSE close after acceptance.
    if "sec" in src:
        sec = src["sec"]
        metadata_path = input_dir / "sec_filings_daily.metadata.json"
        if not metadata_path.exists() or json.loads(metadata_path.read_text()).get("alignment") != "first_nyse_close_after_acceptance":
            raise ValueError("Rebuild SEC data with fetch_sec_filings.py --cached before joining; legacy timestamps are unsafe")
        table = table.merge(sec[["date", "sec_8k_count", "sec_total_filings", "ndx_earnings_count"]], on="date", how="left")
        # Realized 8-K counts are observations, not an advance earnings schedule.
        groups["ndx_earnings_count"] = {"group": "sec", "role": "past"}
        groups.update({"sec_8k_count": {"group": "sec", "role": "past"}, "sec_total_filings": {"group": "sec", "role": "past"}})

    # --- known-future calendar ---
    if "calendar" in src:
        cal = src["calendar"]
        columns = [c for c in cal.columns if c not in ("date", "days_since_fomc")]
        table = table.merge(cal[["date", *columns]], on="date", how="left")
        for column in columns:
            groups[column] = {"group": "calendar", "role": "known_future"}

    table = table.dropna(subset=["log_return_1d"]).sort_values("date").reset_index(drop=True)
    # Compute rolling features on the source warm-up history before trimming the sample.
    table = table.loc[table["date"] >= pd.Timestamp(start)].reset_index(drop=True)

    if profile == "core":
        keep = ["date", "log_return_1d", *[c for c in CORE_PAST if c in table.columns], *[c for c in CORE_FUTURE if c in table.columns]]
        table = table[keep]
        groups = {k: v for k, v in groups.items() if k in keep}

    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    ordered = ["date"] + [c for c in groups if c in table.columns]
    return table[ordered], groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Join processed daily sources into one QQQ modeling table.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/daily_feature_table.csv"))
    parser.add_argument("--profile", choices=["full", "core"], default="full")
    parser.add_argument("--start", default="2006-01-01", help="Modeling sample start, applied after rolling-feature warm-up")
    args = parser.parse_args()

    table, groups = build(args.input_dir, args.profile, args.start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    groups_path = args.output.with_name(args.output.stem + "_groups.json")
    groups_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {"profile": args.profile, "start": args.start, "rows": len(table),
                "table_sha256": digest(args.output), "groups_sha256": digest(groups_path),
                "builder_sha256": digest(Path(__file__)),
                "source_sha256": {name: digest(args.input_dir / name) for name in INPUTS.values()
                                  if (args.input_dir / name).exists()}}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    roles = pd.Series({k: v["role"] for k, v in groups.items()}).value_counts().to_dict()
    logging.info("Wrote %s: %d rows x %d columns (%s -> %s)", args.output, len(table), len(table.columns), table["date"].iloc[0], table["date"].iloc[-1])
    logging.info("Roles: %s; groups written to %s", roles, groups_path)
    coverage = table.drop(columns=["date"]).notna().mean()
    logging.info("Lowest coverage: %s", coverage.sort_values().head(8).round(3).to_dict())


if __name__ == "__main__":
    main()
