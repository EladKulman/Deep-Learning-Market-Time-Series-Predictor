"""Regression checks for time alignment and Chronos future-covariate isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_sec_filings import process_filings, build_daily
from fetch_fomc_statements import release_date
from fetch_fred_macro import fetch_cpi_vintages
from fetch_gdelt_news import merge_cache
from compare_chronos2_validation import prepare_modeling_frame, build_validation_windows, gaussian_baseline, metric_row
from chronos_data import forecast_input, training_inputs
from predict_chronos2 import make_forecast_dates
from build_event_calendar import scheduled_fomc_from_fed_page
from check_data_readiness import session_window, calendar_horizon, check_build_snapshot
from chronos_data import fingerprint
from fetch_fomc_statements import fetch_statements
from unittest.mock import patch
from build_daily_feature_table import fred_available_dates, asof_join


class DataContractTests(unittest.TestCase):
    def test_fred_publication_crosses_market_cutoff_and_federal_holidays(self):
        dates = pd.Series(pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04", "2024-10-11", "2024-11-27"]))
        expected = ["2026-09-04", "2026-09-08", "2026-09-09", "2024-10-16", "2024-12-02"]
        self.assertEqual(fred_available_dates(dates).dt.strftime("%Y-%m-%d").tolist(), expected)
        spread = pd.Series(pd.to_datetime(["2019-05-24", "2026-09-04"]))
        self.assertEqual(fred_available_dates(spread, spread=True).dt.strftime("%Y-%m-%d").tolist(), ["2019-05-29", "2026-09-08"])

    def test_shared_availability_uses_latest_reference_observation(self):
        dates = pd.Series(pd.to_datetime(["2006-04-12", "2006-04-13"]))
        source = pd.DataFrame({"date": fred_available_dates(dates), "yield": [4.91, 4.96]})
        origin = pd.DataFrame({"date": pd.to_datetime(["2006-04-17"])})
        self.assertEqual(asof_join(origin, source, ["yield"])["yield"].iloc[0], 4.96)

    def test_refreshed_source_requires_rebuilding_the_model_table(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("daily_feature_table.csv", "daily_feature_table_groups.json", "news.csv"):
                (folder / name).write_text("old snapshot")
            metadata = {"table_sha256": fingerprint(folder / "daily_feature_table.csv"),
                        "groups_sha256": fingerprint(folder / "daily_feature_table_groups.json"),
                        "source_sha256": {"news.csv": fingerprint(folder / "news.csv")}}
            (folder / "daily_feature_table.metadata.json").write_text(json.dumps(metadata))
            self.assertEqual(check_build_snapshot(folder), [])
            (folder / "news.csv").write_text("new observations")
            self.assertIn("news.csv", check_build_snapshot(folder)[0])

    def test_readiness_uses_completed_nyse_sessions(self):
        _, expected, future = session_window(pd.Timestamp("2026-09-04"), 5, "2026-09-08T12:00:00Z")
        self.assertEqual(str(expected.date()), "2026-09-04")
        self.assertEqual(future.strftime("%Y-%m-%d").tolist(), ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"])
        _, expected, _ = session_window(pd.Timestamp("2026-09-04"), 5, "2026-09-08T21:00:00Z")
        self.assertEqual(str(expected.date()), "2026-09-08")

    def test_missing_calendar_session_cannot_be_replaced_by_later_row(self):
        dates = pd.to_datetime(["2026-09-08", "2026-09-09"])
        calendar = pd.DataFrame({"date": pd.to_datetime(["2026-09-08", "2026-09-10"]), "event": [0, 1]})
        self.assertIsNone(calendar_horizon(calendar, ["event"], dates))

    def test_failed_fed_download_preserves_existing_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "statements.jsonl"
            original = json.dumps({"url": "old", "date": "2020-01-01", "text": "Existing policy statement"}) + "\n"
            cache.write_text(original)
            with patch("fetch_fomc_statements.get", side_effect=RuntimeError("Unavailable")):
                with self.assertRaises(RuntimeError):
                    fetch_statements({"2020-01-01": "old", "2020-02-01": "new"}, cache, False)
            self.assertEqual(cache.read_text(), original)

    def test_sec_utc_early_close_and_missing_timestamp(self):
        source = pd.DataFrame({
            "form": ["8-K"] * 4, "filingDate": ["2024-11-29"] * 4,
            "items": ["2.02"] * 4, "accessionNumber": ["a", "b", "c", "d"],
            "acceptanceDateTime": ["2024-11-29T17:00:00Z", "2024-11-29T18:00:00Z", None, "2024-11-29T17:30:00Z"],
        })
        filings = process_filings(source, "AAPL", "2024-01-01", None)
        self.assertEqual(filings.available_date.tolist(), ["2024-11-29", "2024-12-02", "2024-12-02", "2024-11-29"])
        daily = build_daily(filings, "2024-11-29", "2024-12-02").set_index("date")
        self.assertEqual(daily.loc["2024-11-29", "ndx_earnings_count"], 2)
        self.assertEqual(daily.loc["2024-12-02", "ndx_earnings_count"], 2)
        self.assertEqual(daily.sec_total_filings.sum(), len(source))

    def test_printed_fed_date_overrides_url_typo(self):
        self.assertEqual(release_date("June 28, 2007 For immediate release...", "2007-06-18"), "2007-06-28")

    def test_scheduled_calendar_excludes_emergency_and_notation_votes(self):
        dates = scheduled_fomc_from_fed_page(cached=True)
        for day in ["2008-01-22", "2008-10-08", "2020-03-03", "2025-08-22"]:
            self.assertNotIn(pd.Timestamp(day), dates)
        self.assertIn(pd.Timestamp("2007-06-28"), dates)
        self.assertIn(pd.Timestamp("2020-03-18"), dates)  # originally published, later cancelled

    def test_pure_cpi_revision_updates_level_without_monthly_release_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"date": ["2023-01-01", "2024-01-01", "2024-01-01"],
                          "realtime_start": ["2024-02-13", "2024-02-13", "2024-02-20"],
                          "value": [100., 105., 106.]}).to_csv(path / "CPIAUCSL_all_releases.csv", index=False)
            result = fetch_cpi_vintages(None, path)
            self.assertEqual(result.is_monthly_print.tolist(), [True, False])
            self.assertAlmostEqual(result.iloc[-1].cpi_yoy, .06)

    def test_partial_news_refresh_preserves_cached_history(self):
        old = pd.DataFrame({"count": [1., 2.]}, index=["2026-01-01", "2026-01-02"])
        new = pd.DataFrame({"count": [np.nan, 3.]}, index=["2026-01-02", "2026-01-03"])
        self.assertEqual(merge_cache(old, new)["count"].tolist(), [1., 2., 3.])

    def test_missing_news_keeps_trading_sessions(self):
        frame = self.frame()
        frame["news"] = np.nan
        result = prepare_modeling_frame(frame, "date", "target", ["news"], None)
        self.assertEqual(result.date.tolist(), frame.date.tolist())
        self.assertTrue(result.news.isna().all())

    @staticmethod
    def frame():
        return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=12),
                             "target": np.arange(12) * .001, "news": np.arange(12, dtype=float),
                             "event": [0, 1] * 6})

    def test_future_target_and_news_never_reach_model_input(self):
        frame = self.frame()
        kwargs = dict(split_index=8, prediction_length=2, stride=2, max_windows=1, max_context_rows=None,
                      timestamp_column="date", target_column="target", feature_columns=["news", "event"],
                      known_covariates_names=["event"])
        inputs, _ = build_validation_windows(frame, **kwargs)
        changed = frame.copy()
        changed.loc[8:, ["target", "news"]] = 999
        again, _ = build_validation_windows(changed, **kwargs)
        np.testing.assert_array_equal(inputs[0]["target"], again[0]["target"])
        np.testing.assert_array_equal(inputs[0]["past_covariates"]["news"], again[0]["past_covariates"]["news"])
        self.assertEqual(set(inputs[0]["future_covariates"]), {"event"})
        np.testing.assert_array_equal(inputs[0]["future_covariates"]["event"], frame.event.iloc[8:10])

    def test_unknown_future_calendar_is_rejected(self):
        frame = self.frame()
        future = frame.iloc[-2:].copy()
        future.loc[future.index[-1], "event"] = np.nan
        with self.assertRaisesRegex(ValueError, "calendar is incomplete"):
            forecast_input(frame.iloc[:-2], future, "target", ["event"], ["event"])

    def test_target_only_training_and_masked_covariates(self):
        frame = self.frame()
        frame.loc[:5, "news"] = np.nan
        prepared = training_inputs(frame, "target", "date", ["news", "event"], ["event"], 2)
        self.assertEqual(prepared[0]["n_future_covariates"], 1)
        self.assertEqual(prepared[0]["context"].shape, (3, 12))
        self.assertEqual(int(prepared[0]["context"].isnan().sum()), 6)
        target_only = training_inputs(frame, "target", "date", [], [], 2)
        self.assertEqual(target_only[0]["n_covariates"], 0)

    def test_prediction_dates_skip_exchange_holidays(self):
        context = pd.DataFrame({"date": pd.to_datetime(["2026-09-04"])})
        result = make_forecast_dates(context, None, "date", 3, "B")
        self.assertEqual(result.dt.strftime("%Y-%m-%d").tolist(), ["2026-09-08", "2026-09-09", "2026-09-10"])

    def test_gaussian_baseline_uses_context_only_and_reports_quantile_loss(self):
        inputs = [{"target": np.array([-.02, .01, -.01, .02])}]
        actuals = pd.DataFrame({"window": [0, 0], "horizon": [1, 2], "actual": [.01, -.01]})
        predictions = gaussian_baseline(inputs, actuals, [.1, .5, .9])
        np.testing.assert_array_equal(predictions.prediction, [0, 0])
        self.assertAlmostEqual(predictions["q0.9"].iloc[0], predictions["q0.9"].iloc[1])
        metrics = metric_row("zero_gaussian", "overall", predictions)
        self.assertGreater(metrics["weighted_quantile_loss"], 0)


if __name__ == "__main__":
    unittest.main()
