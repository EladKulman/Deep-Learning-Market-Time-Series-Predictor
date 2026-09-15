#!/usr/bin/env python3
"""Zero-shot covariate ladder: pretrained Chronos-2, no fine-tuning, several validation folds.

This is the "free experiment". Every rung gives the *pretrained* model a different set of
covariates in context and scores the same rolling five-session windows, so the effect of
each feature group is measured without any training noise. Folds are validation calendar
years (context = every session before the fold start, sample from 2017-01-03) plus the
2025H2-2026H1 window used by the fine-tuning screen.

Rungs (built from configs/news_ablation.json so they match the fine-tuning screen):
  r0_target                    no covariates
  r1_qqq                       six QQQ transforms
  r2_qqq_calendar              + six known-future calendar flags
  r3_control                   all 21 control features
  r3_control_nofuture          same 21, but calendar flags passed as past-only (value of the
                               known-future channel)
  r4_plus_uncertainty          control + EPU/EMU
  r5_plus_fed                  control + four FOMC tone features
  r6_plus_gdelt_fed_recession  control + Fed and recession GDELT share/tone
  r7_all_external              control + every external source
The zero-mean Gaussian baseline is scored once per fold as rung "gaussian".

Outputs under --output-dir: <fold>/<rung>/{predictions,metrics}.csv, ladder_metrics.csv,
timings.csv, metadata.json. Summarize with scripts/summarize_ladder.py.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronos_data import fingerprint, load_groups, validate_values  # noqa: E402
from compare_chronos2_validation import (  # noqa: E402
    build_validation_windows,
    gaussian_baseline,
    load_frame,
    parse_quantiles,
    predict_validation_windows,
    prepare_modeling_frame,
    summarize_metrics,
)

DEFAULT_DATA = Path("data/processed/daily_feature_table.csv")
DEFAULT_PLAN = Path("configs/news_ablation.json")
DEFAULT_QUANTILES = "0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99"
SAMPLE_START = "2017-01-03"
FOLDS = {
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2025H2_2026H1": ("2025-07-01", "2026-06-30"),
}
RUNG_ORDER = [
    "r0_target", "r1_qqq", "r2_qqq_calendar", "r3_control", "r3_control_nofuture",
    "r4_plus_uncertainty", "r5_plus_fed", "r6_plus_gdelt_fed_recession", "r7_all_external",
]
METRIC_COLUMNS = [
    "rows", "weighted_quantile_loss", "mae_log_return", "rmse_log_return", "directional_accuracy",
    "q10_q90_coverage", "q01_q99_coverage", "q10_q90_mean_width", "mean_prediction", "mean_actual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="news_ablation.json (feature definitions)")
    parser.add_argument("--rungs", default=",".join(RUNG_ORDER))
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--model-id", default=None, help="Defaults to the plan's model_id")
    parser.add_argument("--model-revision", default=None, help="Defaults to the plan's model_revision")
    parser.add_argument("--device-map", default=None, help="Defaults to mps if available, else cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("models/pretrained-ladder"))
    parser.add_argument("--max-windows", type=int, default=None, help="Smoke-test cap per fold")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--prediction-length", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--quantiles", default=DEFAULT_QUANTILES)
    parser.add_argument("--sample-start", default=SAMPLE_START)
    parser.add_argument("--timestamp-column", default="date")
    parser.add_argument("--target-column", default="log_return_1d")
    return parser.parse_args()


def setup_features(plan: dict, name: str) -> list[str]:
    features = list(plan["control_features"])

    def add(setup_name: str) -> None:
        setup = plan["setups"][setup_name]
        for included in setup.get("include_setups", []):
            add(included)
        features.extend(setup.get("add_features", []))

    add(name)
    return list(dict.fromkeys(features))


def build_rungs(plan: dict, groups: dict) -> dict[str, dict]:
    control = list(plan["control_features"])
    qqq = [c for c in control if groups[c]["group"] == "qqq"]
    calendar = [c for c in control if groups[c]["role"] == "known_future"]
    add = lambda *names: list(dict.fromkeys(control + [f for n in names for f in plan["setups"][n]["add_features"]]))  # noqa: E731
    fed_rec = ["gdelt_fed_news_share", "gdelt_fed_avg_tone", "gdelt_recession_news_share", "gdelt_recession_avg_tone"]
    rungs = {
        "r0_target": {"features": [], "known_future": True},
        "r1_qqq": {"features": qqq, "known_future": True},
        "r2_qqq_calendar": {"features": qqq + calendar, "known_future": True},
        "r3_control": {"features": control, "known_future": True},
        "r3_control_nofuture": {"features": control, "known_future": False},
        "r4_plus_uncertainty": {"features": add("uncertainty"), "known_future": True},
        "r5_plus_fed": {"features": add("fomc_tone"), "known_future": True},
        "r6_plus_gdelt_fed_recession": {"features": list(dict.fromkeys(control + fed_rec)), "known_future": True},
        "r7_all_external": {"features": setup_features(plan, "all_external"), "known_future": True},
    }
    for name, rung in rungs.items():
        bad = [c for c in rung["features"] if c not in groups or groups[c]["role"] not in {"past", "known_future"}]
        if bad:
            raise ValueError(f"{name}: invalid covariates {bad}")
        rung["known"] = [c for c in rung["features"] if groups[c]["role"] == "known_future"] if rung["known_future"] else []
    return rungs


def fold_frame(raw: pd.DataFrame, args, fold: str, features: list[str]):
    start, end = FOLDS[fold]
    frame = prepare_modeling_frame(raw, args.timestamp_column, args.target_column, features,
                                   max_rows=None, start_date=args.sample_start, end_date=end)
    split_index = int((frame[args.timestamp_column] >= pd.Timestamp(start)).idxmax())
    if split_index == 0 or split_index + args.prediction_length > len(frame):
        raise ValueError(f"Fold {fold} has no usable validation rows in the table")
    return frame, split_index


def package_versions() -> dict:
    out = {}
    for name in ("chronos-forecasting", "torch", "transformers", "pandas", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    model_id = args.model_id or plan["model_id"]
    revision = args.model_revision or plan.get("model_revision")
    quantile_levels = parse_quantiles(args.quantiles)
    groups = load_groups(args.data)
    rungs = build_rungs(plan, groups)
    selected_rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    selected_folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    unknown = [r for r in selected_rungs if r not in rungs] + [f for f in selected_folds if f not in FOLDS]
    if unknown:
        raise SystemExit(f"Unknown rungs/folds: {unknown}")

    raw = load_frame(args.data, args.timestamp_column, args.target_column)
    for name in selected_rungs:
        validate_values(raw, args.timestamp_column, args.target_column, rungs[name]["features"], rungs[name]["known"])

    if args.device_map is None:
        import torch
        args.device_map = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    from chronos import Chronos2Pipeline
    from transformers import set_seed
    set_seed(42)
    print(f"Loading {model_id}@{revision} on {args.device_map}")
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=args.device_map, revision=revision)
    revision = revision or getattr(pipeline.model.config, "_commit_hash", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ladder_rows: list[dict] = []
    timings: list[dict] = []
    for fold in selected_folds:
        gaussian_done = False
        for name in selected_rungs:
            rung = rungs[name]
            started = time.time()
            frame, split_index = fold_frame(raw, args, fold, rung["features"])
            inputs, actuals = build_validation_windows(
                frame=frame, split_index=split_index, prediction_length=args.prediction_length,
                stride=args.stride, max_windows=args.max_windows, max_context_rows=None,
                timestamp_column=args.timestamp_column, target_column=args.target_column,
                feature_columns=rung["features"], known_covariates_names=rung["known"],
            )
            predictions = predict_validation_windows(
                pipeline=pipeline, inputs=inputs, actuals=actuals, model_name=name,
                prediction_length=args.prediction_length, quantile_levels=quantile_levels,
                batch_size=args.eval_batch_size, context_length=args.context_length,
            )
            frames = [predictions]
            if not gaussian_done:
                frames.append(gaussian_baseline(inputs, actuals, quantile_levels).assign(model="gaussian"))
                gaussian_done = True
            all_predictions = pd.concat(frames, ignore_index=True)
            metrics = summarize_metrics(all_predictions)

            out_dir = args.output_dir / fold / name
            out_dir.mkdir(parents=True, exist_ok=True)
            all_predictions.to_csv(out_dir / "predictions.csv", index=False)
            metrics.to_csv(out_dir / "metrics.csv", index=False)
            for _, row in metrics.iterrows():
                rung_name = name if row["model"] == name else row["model"]
                ladder_rows.append({"rung": rung_name, "fold": fold, "model": row["model"], "horizon": row["horizon"],
                                    **{c: row.get(c) for c in METRIC_COLUMNS}})
            elapsed = time.time() - started
            overall = metrics[(metrics["model"] == name) & (metrics["horizon"] == "overall")].iloc[0]
            timings.append({"fold": fold, "rung": name, "windows": len(inputs), "n_features": len(rung["features"]),
                            "seconds": round(elapsed, 1)})
            print(f"[{fold}] {name:28s} windows={len(inputs):3d} feats={len(rung['features']):2d} "
                  f"WQL={overall['weighted_quantile_loss']:.6f} cov10-90={overall['q10_q90_coverage']:.3f} "
                  f"({elapsed:.1f}s)", flush=True)
            pd.DataFrame(ladder_rows).to_csv(args.output_dir / "ladder_metrics.csv", index=False)
            pd.DataFrame(timings).to_csv(args.output_dir / "timings.csv", index=False)

    metadata = {
        "model_id": model_id, "model_revision": revision, "device_map": args.device_map,
        "data_path": str(args.data), "data_sha256": fingerprint(args.data),
        "groups_sha256": fingerprint(args.data.with_name(args.data.stem + "_groups.json")),
        "plan_sha256": fingerprint(args.plan), "sample_start": args.sample_start,
        "folds": {f: FOLDS[f] for f in selected_folds},
        "rungs": {r: {"features": rungs[r]["features"], "known_covariates_names": rungs[r]["known"]} for r in selected_rungs},
        "prediction_length": args.prediction_length, "context_length": args.context_length, "stride": args.stride,
        "max_windows": args.max_windows, "quantiles": quantile_levels, "package_versions": package_versions(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.output_dir / 'ladder_metrics.csv'} ({len(ladder_rows)} rows)")


if __name__ == "__main__":
    main()
