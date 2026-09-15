#!/usr/bin/env python3
"""Fine-tune a pretrained Chronos-2 predictor on the QQQ daily feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
from chronos_data import add_feature_args, select_features, training_inputs, provenance


DEFAULT_OUTPUT_DIR = Path("models/chronos2-qqq")
SMOKE_OUTPUT_DIR = Path("models/chronos2-qqq-smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune amazon/chronos-2 on a daily financial time-series table."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/daily_feature_table.csv"),
        help="CSV built by scripts/build_daily_feature_table.py.",
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="Pretrained Chronos-2 model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for trainer outputs and the final checkpoint.",
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
        "--item-id",
        default="QQQ",
        help="Series identifier to assign to this single-instrument dataset.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=None,
        help="Forecast horizon in rows. For daily trading data, 5 is roughly one week.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Maximum historical context length used during fine-tuning.",
    )
    parser.add_argument(
        "--min-past",
        type=int,
        default=None,
        help="Minimum past rows required for a training window. Defaults to prediction length.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Chronological validation fraction from the end of the dataset.",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=["lora", "full"],
        default="lora",
        help="LoRA is the practical default for local experimentation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Fine-tuning learning rate. Use smaller values such as 1e-6 for full fine-tuning.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--train-batch-windows",
        type=int,
        default=None,
        help=(
            "Windows per optimizer step (default 8, smoke 2). Chronos-2 counts the target plus "
            "every covariate row toward batch_size, so the batch_size passed to fit() is "
            "train_batch_windows * (1 + number of covariates)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Explicit Chronos-2 batch_size in series rows (overrides --train-batch-windows). "
            "Values below one window's variate count train on a single window per step."
        ),
    )
    parser.add_argument(
        "--select-on-validation",
        action="store_true",
        help=(
            "Pass the validation split to fit() so Chronos-2 keeps the checkpoint with the best "
            "validation loss. Off by default: that is model selection on the split you then "
            "report, which biases validation metrics. Use only for hyperparameter search on a "
            "split you will not report."
        ),
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to Chronos2Pipeline.from_pretrained, e.g. auto, cuda, cpu.",
    )
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional comma-separated covariate columns. Defaults to all numeric non-target columns.",
    )
    parser.add_argument(
        "--exclude-columns",
        default="open,high,low,close,adj_close,volume,return_1d",
        help="Comma-separated numeric columns to exclude from covariates.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation and train on all rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the most recent N rows. Useful for local smoke tests.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use tiny defaults suitable for a quick local Chronos-2 fine-tuning test.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate rows, feature selection, and splits without loading Chronos-2.",
    )
    add_feature_args(parser)
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "output_dir": SMOKE_OUTPUT_DIR if args.smoke_test else DEFAULT_OUTPUT_DIR,
        "prediction_length": 3 if args.smoke_test else 5,
        "context_length": 64 if args.smoke_test else 512,
        "validation_fraction": 0.2 if args.smoke_test else 0.15,
        "learning_rate": 1e-5,
        "num_steps": 5 if args.smoke_test else 500,
        "train_batch_windows": 2 if args.smoke_test else 8,
        "max_rows": 384 if args.smoke_test else None,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args




def effective_batch_size(args: argparse.Namespace, num_features: int) -> int:
    """Chronos-2 batch_size counts target + covariate rows; one window = 1 + num_features rows."""
    variates = 1 + num_features
    if args.batch_size is not None:
        if args.batch_size < variates:
            print(
                f"WARNING: --batch-size {args.batch_size} is smaller than one window's "
                f"{variates} variates; every optimizer step will see a single window."
            )
        return args.batch_size
    return args.train_batch_windows * variates


def load_modeling_frame(args: argparse.Namespace) -> pd.DataFrame:
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} does not exist. Run scripts/build_daily_feature_table.py first."
        )

    frame = pd.read_csv(args.data)
    required_columns = {args.timestamp_column, args.target_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {args.data}: {sorted(missing_columns)}")

    frame[args.timestamp_column] = pd.to_datetime(frame[args.timestamp_column], errors="coerce")
    if frame[[args.timestamp_column, args.target_column]].isna().any().any():
        raise ValueError("Invalid dates or missing targets would break the trading-session sequence")
    frame = frame.sort_values(args.timestamp_column).reset_index(drop=True)
    if args.max_rows is not None:
        if args.max_rows <= args.prediction_length * 3:
            raise ValueError("--max-rows must be greater than three forecast horizons")
        frame = frame.tail(args.max_rows).reset_index(drop=True)
    frame["item_id"] = args.item_id
    return frame




def chronological_split(
    frame: pd.DataFrame,
    prediction_length: int,
    validation_fraction: float,
    no_validation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if no_validation:
        return frame, None

    if not 0 < validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be greater than 0 and less than 0.5")

    min_validation_rows = prediction_length * 4
    validation_rows = max(min_validation_rows, round(len(frame) * validation_fraction))
    if validation_rows + prediction_length >= len(frame):
        return frame, None

    split_index = len(frame) - validation_rows
    train_frame = frame.iloc[:split_index].copy()
    validation_frame = frame.iloc[split_index:].copy()

    if len(validation_frame) < prediction_length * 2:
        return frame, None

    return train_frame, validation_frame


def to_chronos_inputs(
    frame: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    feature_columns: Sequence[str],
    prediction_length: int,
    known_covariates_names: Sequence[str] = (),
):
    return training_inputs(frame, target_column, timestamp_column, feature_columns,
                           known_covariates_names, prediction_length)


def print_training_plan(
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
) -> None:
    validation_rows = 0 if validation_frame is None else len(validation_frame)
    print("Chronos-2 fine-tuning plan")
    print(f"  model: {args.model_id}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  target: {args.target_column}")
    print(f"  features: {len(feature_columns)}")
    print(f"  max_rows: {args.max_rows}")
    print(f"  train_rows: {len(train_frame)}")
    print(f"  validation_rows: {validation_rows}")
    print(f"  prediction_length: {args.prediction_length}")
    print(f"  context_length: {args.context_length}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  learning_rate: {args.learning_rate}")
    variates = 1 + len(feature_columns)
    print(
        f"  effective batch: {args.effective_batch_size / variates:g} windows x {variates} variates = "
        f"{args.effective_batch_size} rows (train_batch_windows={args.train_batch_windows}, "
        f"batch_size override={args.batch_size})"
    )
    print(f"  select_on_validation: {args.select_on_validation}")
    print(f"  finetune_mode: {args.finetune_mode}")
    print(f"  device_map: {args.device_map}")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    feature_columns: Sequence[str],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
) -> None:
    metadata = {
        **provenance(args, feature_columns, args.known_covariates_names),
        "model_id": args.model_id,
        "target_column": args.target_column,
        "feature_columns": list(feature_columns),
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "finetune_mode": args.finetune_mode,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "train_batch_windows": getattr(args, "train_batch_windows", None),
        "batch_size_override": args.batch_size,
        "batch_size": getattr(args, "effective_batch_size", args.batch_size),
        "variates_per_window": 1 + len(feature_columns),
        "select_on_validation": getattr(args, "select_on_validation", False),
        "smoke_test": args.smoke_test,
        "max_rows": args.max_rows,
        "train_rows": len(train_frame),
        "validation_rows": 0 if validation_frame is None else len(validation_frame),
        "train_start": str(train_frame[args.timestamp_column].min().date()),
        "train_end": str(train_frame[args.timestamp_column].max().date()),
        "validation_start": None
        if validation_frame is None
        else str(validation_frame[args.timestamp_column].min().date()),
        "validation_end": None
        if validation_frame is None
        else str(validation_frame[args.timestamp_column].max().date()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # PEFT does not inherit the loaded base model's commit automatically. Persist it
    # inside the adapter config so offline reload works with a revision-only cache.
    adapter_path = output_dir / "finetuned-ckpt" / "adapter_config.json"
    if adapter_path.exists() and args.model_revision:
        adapter_config = json.loads(adapter_path.read_text())
        adapter_config["revision"] = args.model_revision
        adapter_path.write_text(json.dumps(adapter_config, indent=2) + "\n")
    (output_dir / "fine_tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = apply_mode_defaults(parse_args())
    frame = load_modeling_frame(args)
    resolved_features, args.known_covariates_names = select_features(frame, args)
    args.effective_batch_size = effective_batch_size(args, len(resolved_features))

    train_frame, validation_frame = chronological_split(
        frame=frame,
        prediction_length=args.prediction_length,
        validation_fraction=args.validation_fraction,
        no_validation=args.no_validation,
    )
    print_training_plan(
        args=args,
        feature_columns=resolved_features,
        train_frame=train_frame,
        validation_frame=validation_frame,
    )
    if args.prepare_only:
        print("Prepare-only check completed; Chronos-2 was not loaded.")
        return

    from transformers import set_seed
    set_seed(args.seed)

    train_inputs = to_chronos_inputs(
        train_frame,
        target_column=args.target_column,
        timestamp_column=args.timestamp_column,
        feature_columns=resolved_features,
        prediction_length=args.prediction_length,
        known_covariates_names=args.known_covariates_names,
    )
    # Only hand the validation split to fit() when explicitly requested: Chronos-2 then
    # enables load_best_model_at_end, i.e. checkpoint selection on the reported split.
    validation_inputs = (
        None
        if validation_frame is None or not args.select_on_validation
        else to_chronos_inputs(
            validation_frame,
            target_column=args.target_column,
            timestamp_column=args.timestamp_column,
            feature_columns=resolved_features,
            prediction_length=args.prediction_length,
            known_covariates_names=args.known_covariates_names,
        )
    )

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device_map, revision=args.model_revision)
    args.model_revision = args.model_revision or getattr(pipeline.model.config, "_commit_hash", None)
    pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs,
        prediction_length=args.prediction_length,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.effective_batch_size,
        context_length=args.context_length,
        min_past=args.min_past,
        output_dir=args.output_dir,
        remove_printer_callback=True,
        seed=args.seed,
        report_to="none",
    )

    write_metadata(
        output_dir=args.output_dir,
        args=args,
        feature_columns=resolved_features,
        train_frame=train_frame,
        validation_frame=validation_frame,
    )
    checkpoint_path = args.output_dir / "finetuned-ckpt"
    print(f"Fine-tuned checkpoint saved to {checkpoint_path}")
    print(f"Metadata saved to {args.output_dir / 'fine_tuning_metadata.json'}")


if __name__ == "__main__":
    main()
