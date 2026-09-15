#!/usr/bin/env python3
"""Per-window permutation importance of covariates for a Chronos-2 forecaster, no retraining.

Method ("swap permutation"). For feature f and validation window i, the feature's context
(and its future block when f is a known-future covariate) is replaced by the same feature
taken from a different randomly chosen validation window j, aligned to the tail of the
context. This keeps each series' own autocorrelation and scale while breaking its link to
window i. The window is re-forecast and

    importance(f, i) = WQL_permuted(f, i) - WQL_baseline(i)

where WQL is the weighted quantile loss used by compare_chronos2_validation.py
(2 * mean pinball over the quantiles / mean |actual| within the window). Positive values mean
the model relied on the feature. Repeats draw different donors and are averaged downstream.

Outputs (in --output-dir):
  baseline_predictions.csv   unpermuted forecasts per window and horizon
  importance_by_window.csv   window, forecast_origin, feature, repeat, donor_window,
                             delta_wql, delta_h1..delta_hH
  importance_summary.csv     per feature: mean/std delta, share of windows > 0, per-horizon means
  metadata.json              provenance (features, model, data hash, seed, versions)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_data import add_feature_args, provenance, select_features  # noqa: E402
from compare_chronos2_validation import (  # noqa: E402
    DEFAULT_DATA,
    build_validation_windows,
    load_frame,
    parse_quantiles,
    predict_validation_windows,
    prepare_modeling_frame,
    split_train_validation,
)

DEFAULT_QUANTILES = "0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fine-tuned checkpoint / adapter dir; overrides --model-id")
    parser.add_argument("--output-dir", type=Path, default=Path("models/permutation-importance"))
    parser.add_argument("--target-column", default="log_return_1d")
    parser.add_argument("--timestamp-column", default="date")
    parser.add_argument("--feature-columns", default=None, help="Comma-separated covariates; otherwise the feature profile")
    parser.add_argument("--exclude-columns", default="open,high,low,close,adj_close,volume,return_1d")
    parser.add_argument("--features-to-permute", default=None, help="Comma-separated subset to permute (default: all selected)")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--validation-rows", type=int, default=252)
    parser.add_argument("--prediction-length", type=int, default=5)
    parser.add_argument("--stride", type=int, default=None, help="Defaults to --prediction-length (non-overlapping windows)")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-context-rows", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--quantiles", default=DEFAULT_QUANTILES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device-map", default=None, help="Default: mps if available, else cuda if available, else cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Only affects the core profile's feature subset")
    parser.add_argument("--prepare-only", action="store_true")
    add_feature_args(parser)
    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.prediction_length
    if args.device_map is None:
        import torch
        args.device_map = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return args


# ----------------------------------------------------------------------------- losses

def window_losses(predictions: pd.DataFrame, quantile_levels: Sequence[float], prediction_length: int) -> pd.DataFrame:
    """Per-window WQL plus per-horizon mean pinball, all normalized by the window's mean |actual|."""
    qcols = [f"q{q:g}" for q in quantile_levels]
    rows = []
    for window, group in predictions.groupby("window", sort=True):
        group = group.sort_values("horizon")
        actual = group["actual"].to_numpy()[:, None]
        preds = group[qcols].to_numpy()
        error = actual - preds
        q = np.asarray(quantile_levels)[None, :]
        pinball = np.maximum(q * error, (q - 1) * error)  # horizon x quantile
        denom = float(np.abs(actual).mean())
        per_h = 2 * pinball.mean(axis=1) / denom if denom else np.full(len(group), np.nan)
        row = {"window": int(window), "wql": float(2 * pinball.mean() / denom) if denom else np.nan, "denom": denom}
        for h in range(1, prediction_length + 1):
            row[f"wql_h{h}"] = float(per_h[h - 1]) if h - 1 < len(per_h) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


# ----------------------------------------------------------------------------- swapping

def swap_feature(inputs: Sequence[dict], feature: str, is_known: bool, donors: Sequence[int]) -> list[dict]:
    """Return shallow copies of `inputs` where `feature` in window i comes from window donors[i]."""
    permuted = []
    for i, source in enumerate(inputs):
        target = copy.copy(source)
        target["past_covariates"] = dict(source["past_covariates"])
        donor = inputs[donors[i]]
        own = source["past_covariates"][feature]
        other = donor["past_covariates"][feature]
        n = len(own)
        if len(other) >= n:
            replacement = other[-n:]
        else:  # shorter donor: left-pad with NaN so the observation mask hides the gap
            replacement = np.concatenate([np.full(n - len(other), np.nan, dtype="float32"), other])
        target["past_covariates"][feature] = np.asarray(replacement, dtype="float32")
        if is_known and "future_covariates" in source:
            target["future_covariates"] = dict(source["future_covariates"])
            target["future_covariates"][feature] = np.asarray(donor["future_covariates"][feature], dtype="float32")
        permuted.append(target)
    return permuted


def draw_donors(rng: np.random.Generator, n_windows: int) -> np.ndarray:
    """Every window gets a donor different from itself: one random cycle over all windows."""
    if n_windows < 2:
        raise ValueError("Need at least two windows to swap between")
    order = rng.permutation(n_windows)
    donors = np.empty(n_windows, dtype=int)
    donors[order] = np.roll(order, -1)  # window order[k] borrows from order[k+1]; a single n-cycle has no fixed point
    assert not np.any(donors == np.arange(n_windows))
    return donors


# ----------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    quantile_levels = parse_quantiles(args.quantiles)

    raw_frame = load_frame(args.data, args.timestamp_column, args.target_column)
    feature_columns, known = select_features(raw_frame, args)
    args.known_covariates_names = known
    if not feature_columns:
        raise SystemExit("Permutation importance needs at least one covariate")
    to_permute = feature_columns if args.features_to_permute is None else [c.strip() for c in args.features_to_permute.split(",") if c.strip()]
    unknown = set(to_permute) - set(feature_columns)
    if unknown:
        raise SystemExit(f"--features-to-permute not among selected features: {sorted(unknown)}")

    frame = prepare_modeling_frame(raw_frame, args.timestamp_column, args.target_column, feature_columns,
                                   max_rows=None, start_date=args.start_date, end_date=args.end_date)
    _, validation_frame, split_index = split_train_validation(frame, args.validation_rows, args.prediction_length)
    inputs, actuals = build_validation_windows(
        frame=frame, split_index=split_index, prediction_length=args.prediction_length, stride=args.stride,
        max_windows=args.max_windows, max_context_rows=args.max_context_rows,
        timestamp_column=args.timestamp_column, target_column=args.target_column,
        feature_columns=feature_columns, known_covariates_names=known,
    )
    n_windows = len(inputs)
    print(f"Permutation importance plan: {n_windows} windows x {len(to_permute)} features x {args.repeats} repeats; "
          f"{len(feature_columns)} covariates ({len(known)} known-future); device {args.device_map}")
    if n_windows < 2:
        raise SystemExit("Need at least two validation windows to swap features between them")
    if args.prepare_only:
        print("Prepare-only check completed; Chronos-2 was not loaded.")
        return

    from chronos import Chronos2Pipeline
    from transformers import set_seed
    set_seed(args.seed)
    source = str(args.checkpoint) if args.checkpoint else args.model_id
    print(f"Loading {source} ...")
    pipeline = Chronos2Pipeline.from_pretrained(source, device_map=args.device_map,
                                                revision=None if args.checkpoint else args.model_revision)
    if not args.checkpoint:
        args.model_revision = args.model_revision or getattr(pipeline.model.config, "_commit_hash", None)

    started = time.time()
    baseline = predict_validation_windows(pipeline, inputs, actuals, "baseline", args.prediction_length,
                                          quantile_levels, args.eval_batch_size, args.context_length)
    base_loss = window_losses(baseline, quantile_levels, args.prediction_length)
    print(f"Baseline: mean WQL {base_loss['wql'].mean():.6f} over {n_windows} windows ({time.time() - started:.1f}s)")

    rng = np.random.default_rng(args.seed)
    origins = actuals.drop_duplicates("window").set_index("window")["forecast_origin"]
    horizons = [f"h{h}" for h in range(1, args.prediction_length + 1)]
    records = []
    batch_times = []
    for feature in to_permute:
        is_known = feature in known
        for repeat in range(args.repeats):
            donors = draw_donors(rng, n_windows)
            permuted_inputs = swap_feature(inputs, feature, is_known, donors)
            t0 = time.time()
            permuted = predict_validation_windows(pipeline, permuted_inputs, actuals, f"perm:{feature}:{repeat}",
                                                  args.prediction_length, quantile_levels, args.eval_batch_size, args.context_length)
            batch_times.append(time.time() - t0)
            loss = window_losses(permuted, quantile_levels, args.prediction_length)
            for window in loss.index:
                row = {"window": int(window), "forecast_origin": pd.Timestamp(origins.loc[window]).strftime("%Y-%m-%d"),
                       "feature": feature, "known_future": int(is_known), "repeat": repeat, "donor_window": int(donors[window]),
                       "baseline_wql": float(base_loss.loc[window, "wql"]), "permuted_wql": float(loss.loc[window, "wql"]),
                       "delta_wql": float(loss.loc[window, "wql"] - base_loss.loc[window, "wql"])}
                for h in horizons:
                    row[f"delta_{h}"] = float(loss.loc[window, f"wql_{h}"] - base_loss.loc[window, f"wql_{h}"])
                records.append(row)
        mean_delta = np.mean([r["delta_wql"] for r in records if r["feature"] == feature])
        print(f"  {feature:36s} mean dWQL {mean_delta:+.5f}   ({batch_times[-1]:.1f}s per batch)")

    by_window = pd.DataFrame(records)
    per_feature = by_window.groupby("feature")
    summary = pd.DataFrame({
        "mean_delta_wql": per_feature["delta_wql"].mean(),
        "std_delta_wql": per_feature["delta_wql"].std(),
        "share_windows_positive": per_feature["delta_wql"].apply(lambda s: float((s > 0).mean())),
        "mean_relative_delta": per_feature.apply(lambda g: float((g["delta_wql"] / g["baseline_wql"]).mean())),
        "known_future": per_feature["known_future"].first(),
        **{f"mean_delta_{h}": per_feature[f"delta_{h}"].mean() for h in horizons},
    }).sort_values("mean_delta_wql", ascending=False).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    by_window.to_csv(args.output_dir / "importance_by_window.csv", index=False)
    summary.to_csv(args.output_dir / "importance_summary.csv", index=False)
    metadata = {
        **provenance(args, feature_columns, known),
        "method": "swap permutation between validation windows",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_id": args.model_id,
        "features_permuted": to_permute,
        "quantiles": quantile_levels,
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "validation_rows": args.validation_rows,
        "stride": args.stride,
        "num_windows": n_windows,
        "repeats": args.repeats,
        "validation_start": str(validation_frame[args.timestamp_column].min().date()),
        "validation_end": str(validation_frame[args.timestamp_column].max().date()),
        "baseline_mean_wql": float(base_loss["wql"].mean()),
        "seconds_per_permutation_batch": float(np.mean(batch_times)),
        "device_map": args.device_map,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"\nWrote {args.output_dir}/importance_summary.csv; mean {np.mean(batch_times):.1f}s per (feature x repeat) batch")
    print(summary[["feature", "mean_delta_wql", "share_windows_positive", "mean_relative_delta"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
