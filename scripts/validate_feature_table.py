#!/usr/bin/env python3
"""Sanity-check the feature table against known market history and internal consistency.

The audit script checks shape and leakage; this one checks that the numbers mean what
they should. Every check prints PASS / WARN / FAIL with the observed value so a reader
can judge it, and the script exits non-zero on any FAIL.

    python scripts/validate_feature_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

TABLE = Path("data/processed/daily_feature_table.csv")
GROUPS = Path("data/processed/daily_feature_table_groups.json")
RESULTS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")


def main() -> None:
    t = pd.read_csv(TABLE, parse_dates=["date"]).set_index("date")
    groups = json.loads(GROUPS.read_text())
    past = [c for c, g in groups.items() if g["role"] == "past" and c in t.columns]
    future = [c for c, g in groups.items() if g["role"] == "known_future" and c in t.columns]

    print("\n== Structure ==")
    check("unique, sorted dates", t.index.is_unique and t.index.is_monotonic_increasing, f"{len(t):,} rows")
    nyse = mcal.get_calendar("NYSE").schedule(start_date=t.index.min(), end_date=t.index.max()).index.normalize()
    extra = t.index.difference(nyse); missing = nyse.difference(t.index)
    check("dates match NYSE trading calendar", len(extra) == 0 and len(missing) <= 2, f"{len(extra)} non-trading rows, {len(missing)} trading days absent")
    check("groups JSON covers every column", set(groups) == set(t.columns), f"{len(past)} past, {len(future)} known-future")
    check("known-future columns complete", t[future].isna().sum().sum() == 0, "no NaN in known-future block")
    check("no all-NaN or constant columns", all(t[c].nunique(dropna=True) > 1 for c in past + future), "")

    print("\n== Target ==")
    recomputed = np.log(t["adj_close"]).diff()
    err = (recomputed - t["log_return_1d"]).abs().max()
    check("log_return_1d equals diff(log adj_close)", err < 1e-6, f"max abs diff {err:.2e}")
    r = t["log_return_1d"]
    check("daily return scale plausible", 0.008 < r.std() < 0.02 and abs(r.mean()) < 0.001, f"mean {r.mean():.5f}, std {r.std():.4f}")
    check("worst day is a known crash", r.idxmin().strftime("%Y-%m-%d") in {"2020-03-16", "2008-10-15", "2008-12-01", "2008-09-29"}, f"{r.idxmin().date()} {r.min():+.3f}")
    check("best day is a known rebound", r.idxmax().strftime("%Y-%m-%d") in {"2008-10-13", "2008-10-28", "2020-03-13", "2020-03-24", "2025-04-09"}, f"{r.idxmax().date()} {r.max():+.3f}")

    print("\n== QQQ-derived ==")
    gap = np.log(t["open"] / t["close"].shift(1))
    check("overnight_gap definition", (gap - t["overnight_gap"]).abs().max() < 1e-9, "log(open / prior close)")
    check("parkinson vol positive and comparable to close-to-close vol",
          (t["parkinson_vol_1d"] > 0).all() and 0.5 < t["parkinson_vol_22d"].mean() / r.rolling(22).std().mean() < 1.5,
          f"ratio {t['parkinson_vol_22d'].mean() / r.rolling(22).std().mean():.2f}")
    check("volume z-score centred", abs(t["volume_z_20d"].mean()) < 0.1 and 0.7 < t["volume_z_20d"].std() < 1.3, f"mean {t['volume_z_20d'].mean():.3f}, std {t['volume_z_20d'].std():.2f}")
    check("dist_200dma most negative in 2008-09 crash", t["dist_200dma"].idxmin().year in (2008, 2009, 2020, 2022), f"{t['dist_200dma'].idxmin().date()} {t['dist_200dma'].min():+.2f}")

    print("\n== Market covariates ==")
    check("VXN peaks in 2008 or 2020 panic", t["vxn_close"].idxmax().year in (2008, 2020), f"{t['vxn_close'].idxmax().date()} = {t['vxn_close'].max():.1f}")
    check("March 16 VIX inversion available on March 17", t.loc["2020-03-17", "vix_term_ratio"] > 1.2, f"VIX/VIX3M = {t.loc['2020-03-17', 'vix_term_ratio']:.2f}")
    check("VIX term structure in contango in calm 2017", t.loc["2017", "vix_term_ratio"].median() < 0.9, f"median {t.loc['2017', 'vix_term_ratio'].median():.2f}")
    check("VXN above VIX on average (tech premium)", t["vxn_minus_vix"].mean() > 0, f"mean {t['vxn_minus_vix'].mean():+.2f} points")
    check("HYG fell hard on 2020-03-09", t.loc["2020-03-09", "hyg_log_ret"] < -0.03, f"{t.loc['2020-03-09', 'hyg_log_ret']:+.3f}")
    check("TLT rallied on 2020-03-09", t.loc["2020-03-09", "tlt_log_ret"] > 0.015, f"{t.loc['2020-03-09', 'tlt_log_ret']:+.3f}")
    check("QQQ-SPY spread is small on average", abs(t["qqq_minus_spy"].mean()) < 0.001, f"mean {t['qqq_minus_spy'].mean():+.5f}/day")
    c2022 = t.loc["2022", ["log_return_1d", "tlt_log_ret"]].corr().iloc[0, 1]; c2010s = t.loc["2010":"2019", ["log_return_1d", "tlt_log_ret"]].corr().iloc[0, 1]
    check("stock-bond correlation rose sharply in 2022", c2022 > c2010s + 0.3, f"2022 corr {t.loc['2022', ['log_return_1d', 'tlt_log_ret']].corr().iloc[0, 1]:+.2f} vs 2010-2019 {t.loc['2010':'2019', ['log_return_1d', 'tlt_log_ret']].corr().iloc[0, 1]:+.2f}")

    print("\n== Rates and CPI ==")
    check("2y yield change in basis points scale", 2 < t["d_dgs2_bp"].std() < 12, f"std {t['d_dgs2_bp'].std():.1f} bp/day")
    check("curve inverted mid-2022 to mid-2024", (t.loc["2022-08":"2024-06", "t10y2y_lag1"] < 0).mean() > 0.95, f"{(t.loc['2022-08':'2024-06', 't10y2y_lag1'] < 0).mean():.0%} of days negative")
    fred = pd.read_csv(TABLE.with_name("fred_macro_daily.csv"), parse_dates=["date"]).set_index("date")
    expected = fred["dgs2"].dropna()
    from build_daily_feature_table import fred_available_dates
    expected.index = fred_available_dates(pd.Series(expected.index)).to_numpy()
    expected = expected.groupby(level=0).last()
    expected = expected.reindex(expected.index.union(t.index)).ffill().reindex(t.index).diff() * 100
    lag_error = (expected - t["d_dgs2_bp"]).abs().max()
    check("2y yield changes respect publication lag", lag_error < 1e-9, f"max difference {lag_error:.2e}")
    check("CPI yoy peaked at ~9% in summer 2022", 0.085 < t["cpi_yoy"].max() < 0.095 and t["cpi_yoy"].idxmax().year == 2022, f"{t['cpi_yoy'].max():.3f} on {t['cpi_yoy'].idxmax().date()} (vintage date)")
    check("CPI yoy negative in 2009 deflation", t.loc["2009", "cpi_yoy"].min() < -0.01, f"min {t.loc['2009', 'cpi_yoy'].min():+.3f}")
    jumps = t["cpi_yoy"].diff().abs() > 0
    vintages = pd.read_csv(Path("data/raw/fred/CPIAUCSL_all_releases.csv"), parse_dates=["realtime_start"])
    available = pd.DatetimeIndex(sorted(vintages.realtime_start.unique()))
    idx = t.index.searchsorted(available)
    effective_dates = set(t.index[idx[idx < len(t.index)]])
    check("CPI changes only when an ALFRED release/revision is available", set(t.index[jumps]) <= effective_dates,
          f"{int(jumps.sum())} changes; pure historical revisions included")

    print("\n== Uncertainty and news ==")
    check("EPU (7-day) peaks in the COVID or 2025 tariff shock", t["epu_log"].idxmax().strftime("%Y-%m") in {"2020-03", "2020-04", "2020-05", "2025-04", "2025-05"}, f"max on {t['epu_log'].idxmax().date()}")
    check("EMU (7-day) peaks in a known equity panic", t["emu_log"].idxmax().strftime("%Y-%m") in {"2020-03", "2020-04", "2008-10", "2025-04"}, f"max on {t['emu_log'].idxmax().date()}")
    check("EPU 7-day mean is much less noisy than the raw day", t["epu_log"].diff().std() < 0.5 * t["epu_log_1d"].diff().std(), f"day-to-day std {t['epu_log'].diff().std():.3f} vs raw {t['epu_log_1d'].diff().std():.3f}")
    fed_share = t["gdelt_fed_news_share"]
    after_fomc = fed_share[t["is_fomc_day"].shift(1) == 1].mean(); other = fed_share[t["is_fomc_day"].shift(1) != 1].mean()
    check("Fed news share jumps the day after FOMC", after_fomc / other > 1.3, f"{after_fomc / other:.2f}x the non-FOMC level")
    check("GDELT is NaN before 2017, present after", t.loc[:"2016", "gdelt_ai_news_share"].isna().all() and t.loc["2018":, "gdelt_ai_news_share"].notna().mean() > 0.95, f"coverage 2018+: {t.loc['2018':, 'gdelt_ai_news_share'].notna().mean():.0%}")
    ai_2017 = t.loc["2017", "gdelt_ai_news_share"].mean(); ai_2024 = t.loc["2024", "gdelt_ai_news_share"].mean()
    check("AI news share rose 2017 -> 2024", ai_2024 > ai_2017 * 1.5, f"{ai_2024 / ai_2017:.1f}x")

    print("\n== Fed tone ==")
    tone = t["fomc_net_hawkish_ewma"]
    check("tone most hawkish in the 2022-23 hiking cycle", tone.loc["2022":"2023"].mean() > tone.loc["2010":"2019"].mean() + 0.2, f"2022-23 {tone.loc['2022':'2023'].mean():+.2f} vs 2010s {tone.loc['2010':'2019'].mean():+.2f}")
    check("tone most dovish in 2009-2012 and 2020", tone.loc["2009":"2012"].mean() < -0.3 and tone.loc["2020-04":"2020-12"].mean() < -0.3, f"2009-12 {tone.loc['2009':'2012'].mean():+.2f}, late 2020 {tone.loc['2020-04':'2020-12'].mean():+.2f}")

    print("\n== Calendar ==")
    check("about 8 FOMC days a year", 7.5 <= t.loc["2007":"2025", "is_fomc_day"].groupby(t.loc["2007":"2025"].index.year).sum().mean() <= 9, f"mean {t.loc['2007':'2025', 'is_fomc_day'].groupby(t.loc['2007':'2025'].index.year).sum().mean():.1f}")
    check("days_to_fomc is 0 on FOMC days", (t.loc[t["is_fomc_day"] == 1, "days_to_fomc"] == 0).all(), "")
    check("emergency Fed decisions are not known-future meetings", (t.loc[["2008-01-22", "2008-10-08", "2020-03-03", "2020-03-23", "2025-08-22"], "is_fomc_day"] == 0).all(), "unscheduled calls and notation votes excluded")
    check("earnings counts are past-only", groups["ndx_earnings_count"]["role"] == "past", "8-K observations are not an advance schedule")
    check("about 12 CPI and 12 NFP days a year", 11 <= t.loc["2007":"2024", "is_cpi_day"].groupby(t.loc["2007":"2024"].index.year).sum().mean() <= 12.5 and 11 <= t.loc["2007":"2024", "is_nfp_day"].groupby(t.loc["2007":"2024"].index.year).sum().mean() <= 12.5,
          f"CPI {t.loc['2007':'2024', 'is_cpi_day'].groupby(t.loc['2007':'2024'].index.year).sum().mean():.1f}, NFP {t.loc['2007':'2024', 'is_nfp_day'].groupby(t.loc['2007':'2024'].index.year).sum().mean():.1f}")
    opex = t.index[t["is_opex_day"] == 1]
    check("opex days are Fridays (or Thursday before a holiday)", (pd.Series(opex.weekday).isin([3, 4])).all() and (pd.Series(opex.weekday) == 4).mean() > 0.9, f"{(pd.Series(opex.weekday) == 4).mean():.0%} Fridays")
    check("Apple/Amazon/Meta earnings of 2024-02-01 (after close) land on 2024-02-02", t.loc["2024-02-02", "ndx_earnings_count"] >= 3 and t.loc["2024-02-01", "ndx_earnings_count"] == 0, f"count on 02-02 = {t.loc['2024-02-02', 'ndx_earnings_count']:.0f}, on 02-01 = {t.loc['2024-02-01', 'ndx_earnings_count']:.0f}")
    season = t["ndx_earnings_count"].groupby(t.index.month).sum()
    check("earnings cluster in Jan/Feb, Apr/May, Jul/Aug, Oct/Nov", season[[1, 2, 4, 5, 7, 8, 10, 11]].sum() / season.sum() > 0.9, f"{season[[1, 2, 4, 5, 7, 8, 10, 11]].sum() / season.sum():.0%} in reporting months")

    print("\n== Event-day behaviour (Savor-Wilson style) ==")
    absr = r.abs()
    for flag in ("is_fomc_day", "is_cpi_day", "is_nfp_day"):
        on = absr[t[flag] == 1].mean(); off = absr[t[flag] == 0].mean()
        check(f"|return| higher on {flag}", on > off, f"{on:.4f} vs {off:.4f} ({on / off:.2f}x)", warn_only=True)
    on = absr[t["ndx_earnings_count"] > 0].mean(); off = absr[t["ndx_earnings_count"] == 0].mean()
    check("|return| higher on mega-cap earnings days", on > off, f"{on:.4f} vs {off:.4f} ({on / off:.2f}x)", warn_only=True)

    print("\n== Redundancy among past covariates ==")
    corr = t[past].corr().abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(past) for b in past[i + 1:] if corr.loc[a, b] > 0.9]
    check("no near-duplicate past covariates (|corr| > 0.9)", len(pairs) == 0, "; ".join(f"{a}~{b} {c:.2f}" for a, b, c in pairs) or "none", warn_only=True)

    print("\n== Drift (level features are expected to drift; changes should not) ==")
    drifting = []
    for c in past:
        s = t[c].dropna()
        if len(s) < 1000:
            continue
        half = len(s) // 2
        shift = abs(s.iloc[:half].mean() - s.iloc[half:].mean()) / s.std()
        if shift > 0.5:
            drifting.append(f"{c} {shift:.2f}")
    check("only regime-level columns drift", all(not d.startswith(("d_", "vxn_log", "vvix_log", "tlt", "hyg", "overnight", "qqq_minus")) for d in drifting), "; ".join(drifting) or "none", warn_only=True)

    fails = [x for x in RESULTS if x[0] == "FAIL"]; warns = [x for x in RESULTS if x[0] == "WARN"]
    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
