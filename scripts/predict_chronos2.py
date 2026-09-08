#!/usr/bin/env python3
"""Generate forecasts from a Chronos-2 fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import pandas_market_calendars as mcal
from chronos_data import add_feature_args, select_features, forecast_input


DEFAULT_CHECKPOINT = Path("models/chronos2-qqq-smoke/finetuned-ckpt")
DEFAULT_DATA = Path("data/processed/daily_feature_table.csv")
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast QQQ returns using a fine-tuned Chronos-2 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Fine-tuned Chronos-2 checkpoint or LoRA adapter directory.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="CSV built by scripts/build_daily_feature_table.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chronos2_smoke_forecast.csv"),
        help="Where to write forecast rows.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target column. Defaults to the value saved in fine_tuning_metadata.json.",
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
        help="Forecast horizon. Defaults to the value saved in fine_tuning_metadata.json.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Maximum context passed to Chronos-2. Defaults to saved metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prediction batch size.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to Chronos2Pipeline.from_pretrained.",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional comma-separated covariates. Defaults to saved metadata.",
    )
    parser.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantile levels to write.",
    )
    parser.add_argument(
        "--holdout-rows",
        type=int,
        default=0,
        help="Hold back the last N rows and compare predictions with known actual values.",
    )
    parser.add_argument(
        "--max-context-rows",
        type=int,
        default=None,
        help="Use only the most recent N rows before the forecast origin.",
    )
    parser.add_argument(
        "--date-freq",
        default="B",
        help="Frequency for future forecast dates when not using holdout rows.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs and print the forecast plan without loading Chronos-2.",
    )
    parser.add_argument("--pretrained", action="store_true", help="Forecast with the base model instead of loading a checkpoint")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--calendar", type=Path, default=Path("data/processed/event_calendar_daily.csv"))
    parser.add_argument("--exclude-columns", default="open,high,low,close,adj_close,volume,return_1d")
    add_feature_args(parser)
    return parser.parse_args()


def parse_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    return values or None


def parse_quantiles(raw_value: str) -> list[float]:
    values = [float(value) for value in raw_value.split(",") if value.strip()]
    if not values:
        return DEFAULT_QUANTILES
    if min(values) <= 0 or max(values) >= 1:
        raise ValueError("--quantiles must be strictly between 0 and 1")
    return values


def metadata_path_for_checkpoint(checkpoint: Path) -> Path | None:
    candidates = [
        checkpoint.parent / "fine_tuning_metadata.json",
        checkpoint / "fine_tuning_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_metadata(checkpoint: Path) -> dict:
    metadata_path = metadata_path_for_checkpoint(checkpoint)
    if metadata_path is None:
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_frame(
    data_path: Path,
    timestamp_column: str,
    target_column: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run scripts/build_daily_feature_table.py first."
        )

    frame = pd.read_csv(data_path)
    required_columns = {timestamp_column, target_column, *feature_columns}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {data_path}: {sorted(missing_columns)}")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if frame[[timestamp_column, target_column]].isna().any().any():
        raise ValueError("Invalid dates or missing targets would break the trading-session sequence")
    return frame.sort_values(timestamp_column).reset_index(drop=True)


def split_context_and_actuals(
    frame: pd.DataFrame,
    holdout_rows: int,
    prediction_length: int,
    max_context_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if holdout_rows < 0:
        raise ValueError("--holdout-rows cannot be negative")
    if holdout_rows:
        if holdout_rows < prediction_length:
            raise ValueError("--holdout-rows must be at least --prediction-length")
        if holdout_rows >= len(frame):
            raise ValueError("--holdout-rows leaves no context rows")
        actual_start = len(frame) - holdout_rows
        actual_end = actual_start + prediction_length
        context = frame.iloc[:actual_start].copy()
        actuals = frame.iloc[actual_start:actual_end].copy()
    else:
        context = frame.copy()
        actuals = None

    if max_context_rows is not None:
        if max_context_rows <= prediction_length:
            raise ValueError("--max-context-rows must be greater than --prediction-length")
        context = context.tail(max_context_rows).copy()

    return context.reset_index(drop=True), actuals


def make_chronos_input(
    context: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
    future: pd.DataFrame | None = None,
    known_covariates_names: Sequence[str] = (),
) -> list[dict]:
    return [forecast_input(context, future, target_column, feature_columns, known_covariates_names)]


def make_forecast_dates(
    context: pd.DataFrame,
    actuals: pd.DataFrame | None,
    timestamp_column: str,
    prediction_length: int,
    date_freq: str,
) -> pd.Series:
    if actuals is not None:
        return actuals[timestamp_column].reset_index(drop=True)

    last_timestamp = context[timestamp_column].max()
    if date_freq != "B":
        raise ValueError("QQQ forecasts use NYSE sessions; --date-freq overrides are unsupported")
    days = mcal.get_calendar("NYSE").schedule(start_date=last_timestamp + pd.Timedelta(days=1),
        end_date=last_timestamp + pd.Timedelta(days=prediction_length * 3 + 30)).index
    return pd.Series(days[:prediction_length])


def build_forecast_frame(
    forecast_dates: pd.Series,
    predictions,
    quantile_levels: Sequence[float],
    target_column: str,
    actuals: pd.DataFrame | None,
) -> pd.DataFrame:
    quantile_values = predictions[0][0].detach().cpu().numpy()
    result = pd.DataFrame(
        {
            "date": forecast_dates.dt.strftime("%Y-%m-%d"),
            "target": target_column,
        }
    )
    for quantile_index, quantile_level in enumerate(quantile_levels):
        column_name = f"q{quantile_level:g}"
        result[column_name] = quantile_values[:, quantile_index]

    median_column = "q0.5" if "q0.5" in result.columns else result.columns[-1]
    result["prediction"] = result[median_column]

    if actuals is not None:
        actual_values = actuals[target_column].reset_index(drop=True)
        result["actual"] = actual_values
        result["error"] = result["prediction"] - result["actual"]
        result["abs_error"] = result["error"].abs()

    return result


def print_plan(
    args: argparse.Namespace,
    target_column: str,
    feature_columns: Sequence[str],
    prediction_length: int,
    context_length: int | None,
    context: pd.DataFrame,
    actuals: pd.DataFrame | None,
) -> None:
    print("Chronos-2 prediction plan")
    print(f"  model: {args.model_id if args.pretrained else args.checkpoint}")
    print(f"  output: {args.output}")
    print(f"  target: {target_column}")
    print(f"  features: {len(feature_columns)}")
    print(f"  context_rows: {len(context)}")
    print(f"  holdout_rows: {0 if actuals is None else len(actuals)}")
    print(f"  prediction_length: {prediction_length}")
    print(f"  context_length: {context_length}")
    print(f"  device_map: {args.device_map}")


def main() -> None:
    args = parse_args()
    metadata = {} if args.pretrained else load_metadata(args.checkpoint)
    if not args.pretrained and metadata.get("schema_version") != 2:
        raise ValueError("Checkpoint uses the old data schema or has no metadata. Retrain on the corrected table, or use --pretrained.")
    adapter_path = args.checkpoint / "adapter_config.json"
    if not args.pretrained and adapter_path.exists() and metadata.get("model_revision"):
        adapter = json.loads(adapter_path.read_text())
        if adapter.get("revision") != metadata["model_revision"]:
            raise ValueError("Adapter base revision differs from its training metadata; pin adapter_config.json before offline reload")
    target_column = args.target_column or metadata.get("target_column", "log_return_1d")
    args.target_column = target_column
    if not args.pretrained and args.feature_columns is None:
        args.feature_columns = ",".join(metadata["feature_columns"]) or "none"
    frame = load_frame(args.data, args.timestamp_column, target_column, [])
    feature_columns, known_covariates_names = select_features(frame, args)
    if not args.pretrained and (feature_columns != metadata["feature_columns"] or known_covariates_names != metadata["known_covariates_names"]):
        raise ValueError("Selected feature names/roles differ from the checkpoint's training schema")

    prediction_length = args.prediction_length or metadata.get("prediction_length", 3)
    context_length = args.context_length or metadata.get("context_length")
    quantile_levels = parse_quantiles(args.quantiles)

    context, actuals = split_context_and_actuals(
        frame=frame,
        holdout_rows=args.holdout_rows,
        prediction_length=prediction_length,
        max_context_rows=args.max_context_rows,
    )
    print_plan(
        args=args,
        target_column=target_column,
        feature_columns=feature_columns,
        prediction_length=prediction_length,
        context_length=context_length,
        context=context,
        actuals=actuals,
    )
    forecast_dates = make_forecast_dates(context, actuals, args.timestamp_column, prediction_length, args.date_freq)
    if actuals is not None:
        future = actuals
    else:
        calendar = pd.read_csv(args.calendar, parse_dates=["date"]).set_index("date") if known_covariates_names else pd.DataFrame()
        future = calendar.reindex(forecast_dates)
    inputs = make_chronos_input(context, target_column, feature_columns, future, known_covariates_names)
    print(f"  known_future_features: {len(known_covariates_names)}")
    print(f"  forecast_dates: {[str(d.date()) for d in forecast_dates]}")
    if args.prepare_only:
        print("Prepare-only check completed; Chronos-2 was not loaded.")
        return

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id if args.pretrained else args.checkpoint,
        device_map=args.device_map, **({"revision": args.model_revision} if args.pretrained else {}))
    quantiles, _ = pipeline.predict_quantiles(
        inputs=inputs,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=args.batch_size,
        context_length=context_length,
    )
    forecast = build_forecast_frame(
        forecast_dates=forecast_dates,
        predictions=quantiles,
        quantile_levels=quantile_levels,
        target_column=target_column,
        actuals=actuals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    print(f"Wrote {len(forecast)} forecast rows to {args.output}")
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
