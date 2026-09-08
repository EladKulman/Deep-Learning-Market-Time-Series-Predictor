#!/usr/bin/env python3
"""Compare base Chronos-2 against a fine-tuned Chronos-2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np
from statistics import NormalDist
from chronos_data import add_feature_args, select_features, training_inputs, forecast_input, provenance


DEFAULT_DATA = Path("data/processed/daily_feature_table.csv")
DEFAULT_OUTPUT_DIR = Path("models/chronos2-validation-comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split data chronologically, validate base Chronos-2, fine-tune on "
            "train rows, validate the fine-tuned model on the same windows, and compare."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="CSV built by scripts/build_daily_feature_table.py.",
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="Base pretrained Chronos-2 model ID or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fine-tuned checkpoint, predictions, metrics, and metadata.",
    )
    parser.add_argument(
        "--target-column",
        default="log_return_1d",
        help="Numeric target column to forecast.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="date",
        help="Timestamp column in the input CSV.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=None,
        help="Forecast horizon per validation window.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Maximum context length used for training and validation.",
    )
    parser.add_argument(
        "--validation-rows",
        type=int,
        default=None,
        help="Number of final rows reserved for validation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Rows between validation forecast origins. Defaults to prediction length.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of validation windows, starting from the earliest window.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the most recent N rows before splitting. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-context-rows",
        type=int,
        default=None,
        help="Use only the most recent N context rows for each validation window.",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional comma-separated covariates; otherwise use the selected feature profile.",
    )
    parser.add_argument(
        "--exclude-columns",
        default="open,high,low,close,adj_close,volume,return_1d",
        help="Comma-separated numeric columns to exclude from default covariates.",
    )
    parser.add_argument(
        "--quantiles",
        default="0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99",
        help="Comma-separated quantile levels to evaluate.",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=["lora", "full"],
        default="lora",
        help="Fine-tuning mode for the second model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Fine-tuning learning rate.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Fine-tuning optimizer steps.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Fine-tuning batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation prediction batch size.",
    )
    parser.add_argument(
        "--min-past",
        type=int,
        default=None,
        help="Minimum past rows required during fine-tuning.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to Chronos2Pipeline.from_pretrained.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small local defaults for a quick base-vs-fine-tuned comparison.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate data, splits, windows, and feature selection without loading Chronos-2.",
    )
    parser.add_argument("--base-only", action="store_true", help="Evaluate pretrained Chronos-2 and the Gaussian baseline without fine-tuning")
    add_feature_args(parser)
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "prediction_length": 3 if args.smoke_test else 5,
        "context_length": 64 if args.smoke_test else 512,
        "validation_rows": 60 if args.smoke_test else 252,
        "max_rows": 512 if args.smoke_test else None,
        "max_context_rows": 128 if args.smoke_test else None,
        "max_windows": 5 if args.smoke_test else None,
        "learning_rate": 1e-5,
        "num_steps": 5 if args.smoke_test else 200,
        "train_batch_size": 8 if args.smoke_test else 64,
        "eval_batch_size": 8 if args.smoke_test else 64,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.stride is None:
        args.stride = args.prediction_length
    return args




def parse_quantiles(raw_value: str) -> list[float]:
    quantiles = [float(value) for value in raw_value.split(",") if value.strip()]
    if not quantiles:
        raise ValueError("--quantiles must contain at least one value")
    if min(quantiles) <= 0 or max(quantiles) >= 1:
        raise ValueError("--quantiles must be strictly between 0 and 1")
    if not all(np.isfinite(quantiles)) or quantiles != sorted(set(quantiles)) or 0.5 not in quantiles:
        raise ValueError("--quantiles must be finite, unique, increasing, and include 0.5")
    return quantiles


def load_frame(
    data_path: Path,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run scripts/build_daily_feature_table.py first."
        )

    frame = pd.read_csv(data_path)
    required_columns = {timestamp_column, target_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {data_path}: {sorted(missing_columns)}")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if frame[[timestamp_column, target_column]].isna().any().any():
        raise ValueError("Invalid dates or missing targets would break the trading-session sequence")
    return frame.sort_values(timestamp_column).reset_index(drop=True)




def prepare_modeling_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    feature_columns: Sequence[str],
    max_rows: int | None,
) -> pd.DataFrame:
    frame = frame.copy()  # Missing covariates are masked, never used to remove trading days.
    frame = frame.sort_values(timestamp_column).reset_index(drop=True)
    if max_rows is not None:
        frame = frame.tail(max_rows).reset_index(drop=True)
    return frame


def split_train_validation(
    frame: pd.DataFrame,
    validation_rows: int,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if validation_rows < prediction_length:
        raise ValueError("--validation-rows must be at least --prediction-length")
    if validation_rows >= len(frame):
        raise ValueError("--validation-rows leaves no training rows")

    split_index = len(frame) - validation_rows
    train_frame = frame.iloc[:split_index].copy()
    validation_frame = frame.iloc[split_index:].copy()
    if len(train_frame) <= prediction_length:
        raise ValueError("Training split is too short for the selected prediction length")
    return train_frame, validation_frame, split_index


def build_validation_windows(
    frame: pd.DataFrame,
    split_index: int,
    prediction_length: int,
    stride: int,
    max_windows: int | None,
    max_context_rows: int | None,
    timestamp_column: str,
    target_column: str,
    feature_columns: Sequence[str],
    known_covariates_names: Sequence[str] = (),
) -> tuple[list[dict], pd.DataFrame]:
    if stride <= 0:
        raise ValueError("--stride must be positive")

    inputs = []
    actual_rows = []
    window_index = 0
    max_origin = len(frame) - prediction_length
    for origin in range(split_index, max_origin + 1, stride):
        context = frame.iloc[:origin].copy()
        if max_context_rows is not None:
            if max_context_rows <= prediction_length:
                raise ValueError("--max-context-rows must be greater than --prediction-length")
            context = context.tail(max_context_rows).copy()

        actual = frame.iloc[origin : origin + prediction_length].copy()
        inputs.append(forecast_input(context, actual, target_column, feature_columns, known_covariates_names))

        for horizon_index, (_, row) in enumerate(actual.iterrows(), start=1):
            actual_rows.append(
                {
                    "window": window_index,
                    "horizon": horizon_index,
                    "date": row[timestamp_column],
                    "forecast_origin": context[timestamp_column].iloc[-1],
                    "actual": row[target_column],
                }
            )
        window_index += 1
        if max_windows is not None and window_index >= max_windows:
            break

    if not inputs:
        raise ValueError("No validation windows were created")

    actuals = pd.DataFrame(actual_rows)
    return inputs, actuals


def to_chronos_training_inputs(
    frame: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_columns: Sequence[str],
    prediction_length: int,
    known_covariates_names: Sequence[str] = (),
):
    return training_inputs(frame, target_column, timestamp_column, feature_columns,
                           known_covariates_names, prediction_length)


def predict_validation_windows(
    pipeline,
    inputs: Sequence[dict],
    actuals: pd.DataFrame,
    model_name: str,
    prediction_length: int,
    quantile_levels: Sequence[float],
    batch_size: int,
    context_length: int,
) -> pd.DataFrame:
    quantiles, _ = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=prediction_length,
        quantile_levels=list(quantile_levels),
        batch_size=batch_size,
        context_length=context_length,
    )

    prediction_rows = []
    for window_index, prediction in enumerate(quantiles):
        quantile_values = prediction[0].detach().cpu().numpy()
        for horizon_index in range(prediction_length):
            row = {
                "model": model_name,
                "window": window_index,
                "horizon": horizon_index + 1,
            }
            for quantile_index, quantile_level in enumerate(quantile_levels):
                row[f"q{quantile_level:g}"] = quantile_values[horizon_index, quantile_index]
            row["prediction"] = row.get("q0.5", quantile_values[horizon_index, len(quantile_levels) // 2])
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    result = predictions.merge(actuals, on=["window", "horizon"], how="left", validate="one_to_one")
    return add_errors(result)


def gaussian_baseline(inputs, actuals, quantile_levels) -> pd.DataFrame:
    result = actuals.copy()
    result["model"] = "zero_gaussian"
    # One-day returns at each horizon: no sqrt(h) cumulative-return scaling.
    vol = {i: float(np.nanstd(x["target"][-60:], ddof=1)) for i, x in enumerate(inputs)}
    sigma = result["window"].map(vol)
    for q in quantile_levels:
        result[f"q{q:g}"] = sigma * NormalDist().inv_cdf(q)
    result["prediction"] = 0.0
    return add_errors(result)


def add_errors(result: pd.DataFrame) -> pd.DataFrame:
    result["error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    result["correct_direction"] = (
        (result["prediction"] >= 0) == (result["actual"] >= 0)
    ).astype(int)
    if "q0.1" in result.columns and "q0.9" in result.columns:
        result["interval_coverage"] = (
            (result["actual"] >= result["q0.1"]) & (result["actual"] <= result["q0.9"])
        ).astype(int)
        result["interval_width"] = result["q0.9"] - result["q0.1"]
    return result


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, horizon), group in predictions.groupby(["model", "horizon"], dropna=False):
        rows.append(metric_row(model_name, horizon, group))
    for model_name, group in predictions.groupby("model", dropna=False):
        rows.append(metric_row(model_name, "overall", group))
    return pd.DataFrame(rows)


def metric_row(model_name: str, horizon, group: pd.DataFrame) -> dict:
    row = {
        "model": model_name,
        "horizon": horizon,
        "rows": len(group),
        "mae_log_return": group["abs_error"].mean(),
        "rmse_log_return": group["squared_error"].mean() ** 0.5,
        "bias_log_return": group["error"].mean(),
        "directional_accuracy": group["correct_direction"].mean(),
        "mean_prediction": group["prediction"].mean(),
        "mean_actual": group["actual"].mean(),
    }
    if "interval_coverage" in group.columns:
        row["q10_q90_coverage"] = group["interval_coverage"].mean()
        row["q10_q90_mean_width"] = group["interval_width"].mean()
    qcols = [c for c in group if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    losses = []
    for column in qcols:
        error = group["actual"] - group[column]
        q = float(column[1:])
        losses.append(np.maximum(q * error, (q - 1) * error).to_numpy())
    row["mean_pinball_loss"] = float(np.mean(losses))
    denominator = group["actual"].abs().mean()
    row["weighted_quantile_loss"] = 2 * row["mean_pinball_loss"] / denominator if denominator else np.nan
    if "q0.01" in group and "q0.99" in group:
        row["q01_q99_coverage"] = ((group.actual >= group["q0.01"]) & (group.actual <= group["q0.99"])).mean()
    return row


def write_metadata(
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    num_windows: int,
) -> None:
    metadata = {
        **provenance(args, feature_columns, args.known_covariates_names),
        "base_only": args.base_only,
        "quantiles": parse_quantiles(args.quantiles),
        "model_id": args.model_id,
        "target_column": args.target_column,
        "feature_columns": list(feature_columns),
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "validation_rows": args.validation_rows,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "num_windows": num_windows,
        "finetune_mode": args.finetune_mode,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_rows": len(train_frame),
        "validation_rows_actual": len(validation_frame),
        "train_start": str(train_frame[args.timestamp_column].min().date()),
        "train_end": str(train_frame[args.timestamp_column].max().date()),
        "validation_start": str(validation_frame[args.timestamp_column].min().date()),
        "validation_end": str(validation_frame[args.timestamp_column].max().date()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_plan(
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    num_windows: int,
) -> None:
    print("Chronos-2 validation comparison plan")
    print(f"  model: {args.model_id}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  target: {args.target_column}")
    print(f"  features: {len(feature_columns)}")
    print(f"  train_rows: {len(train_frame)}")
    print(f"  validation_rows: {len(validation_frame)}")
    print(f"  validation_windows: {num_windows}")
    print(f"  prediction_length: {args.prediction_length}")
    print(f"  context_length: {args.context_length}")
    print(f"  max_context_rows: {args.max_context_rows}")
    print(f"  finetune_steps: {args.num_steps}")
    print(f"  train_batch_size: {args.train_batch_size}")
    print(f"  eval_batch_size: {args.eval_batch_size}")
    print(f"  device_map: {args.device_map}")


def main() -> None:
    args = apply_mode_defaults(parse_args())
    quantile_levels = parse_quantiles(args.quantiles)

    raw_frame = load_frame(args.data, args.timestamp_column, args.target_column)
    feature_columns, args.known_covariates_names = select_features(raw_frame, args)

    frame = prepare_modeling_frame(
        frame=raw_frame,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
        feature_columns=feature_columns,
        max_rows=args.max_rows,
    )
    train_frame, validation_frame, split_index = split_train_validation(
        frame=frame,
        validation_rows=args.validation_rows,
        prediction_length=args.prediction_length,
    )
    validation_inputs, actuals = build_validation_windows(
        frame=frame,
        split_index=split_index,
        prediction_length=args.prediction_length,
        stride=args.stride,
        max_windows=args.max_windows,
        max_context_rows=args.max_context_rows,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
        feature_columns=feature_columns,
        known_covariates_names=args.known_covariates_names,
    )
    print_plan(args, feature_columns, train_frame, validation_frame, len(validation_inputs))
    if args.prepare_only:
        print("Prepare-only check completed; Chronos-2 was not loaded.")
        return

    from chronos import Chronos2Pipeline
    from transformers import set_seed
    set_seed(args.seed)

    print("Loading base Chronos-2 pipeline...")
    base_pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device_map, revision=args.model_revision)
    args.model_revision = args.model_revision or getattr(base_pipeline.model.config, "_commit_hash", None)

    print("Validating base pretrained model...")
    base_predictions = predict_validation_windows(
        pipeline=base_pipeline,
        inputs=validation_inputs,
        actuals=actuals,
        model_name="base_pretrained",
        prediction_length=args.prediction_length,
        quantile_levels=quantile_levels,
        batch_size=args.eval_batch_size,
        context_length=args.context_length,
    )

    predictions = [base_predictions, gaussian_baseline(validation_inputs, actuals, quantile_levels)]
    if not args.base_only:
        predictions.append(fine_tuned_predictions(base_pipeline, args, train_frame, feature_columns,
                                                 validation_inputs, actuals, quantile_levels))
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics = summarize_metrics(all_predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args, feature_columns, train_frame, validation_frame, len(validation_inputs))
    predictions_path = args.output_dir / "validation_predictions.csv"
    metrics_path = args.output_dir / "validation_metrics.csv"
    all_predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote validation predictions to {predictions_path}")
    print(f"Wrote comparison metrics to {metrics_path}")
    print(metrics.to_string(index=False))


def fine_tuned_predictions(base_pipeline, args, train_frame, feature_columns, validation_inputs, actuals, quantile_levels):
    print("Fine-tuning on train split...")
    train_inputs = to_chronos_training_inputs(
        frame=train_frame,
        target_column=args.target_column,
        timestamp_column=args.timestamp_column,
        feature_columns=feature_columns,
        prediction_length=args.prediction_length,
        known_covariates_names=args.known_covariates_names,
    )
    finetuned_pipeline = base_pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.train_batch_size,
        context_length=args.context_length,
        min_past=args.min_past,
        output_dir=args.output_dir / "finetuned",
        remove_printer_callback=True,
        seed=args.seed,
        report_to="none",
    )

    print("Validating fine-tuned model on the same windows...")
    # Keep enough metadata beside the adapter for the standalone prediction script.
    from fine_tune_chronos2 import write_metadata as write_fine_tuning_metadata
    from copy import copy
    training_args = copy(args)
    training_args.batch_size = args.train_batch_size
    write_fine_tuning_metadata(args.output_dir / "finetuned", training_args, feature_columns, train_frame, None)
    return predict_validation_windows(
        pipeline=finetuned_pipeline,
        inputs=validation_inputs,
        actuals=actuals,
        model_name="fine_tuned",
        prediction_length=args.prediction_length,
        quantile_levels=quantile_levels,
        batch_size=args.eval_batch_size,
        context_length=args.context_length,
    )

if __name__ == "__main__":
    main()
