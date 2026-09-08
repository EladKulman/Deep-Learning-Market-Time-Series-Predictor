#!/usr/bin/env python3
"""Summarize a Chronos-2 base-vs-fine-tuned validation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_COMPARISON_DIR = Path("models/chronos2-validation-comparison")
BASE_MODEL_NAME = "base_pretrained"
FINE_TUNED_MODEL_NAME = "fine_tuned"
LOWER_IS_BETTER = {"mae_log_return", "rmse_log_return", "abs_bias_log_return", "weighted_quantile_loss", "mean_pinball_loss"}
HIGHER_IS_BETTER = {"directional_accuracy"}
COVERAGE_TARGET = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a readable summary of Chronos-2 validation comparison outputs."
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Directory containing validation_metrics.csv and validation_predictions.csv.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional explicit path to validation_metrics.csv.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional explicit path to validation_predictions.csv.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional explicit path to comparison_metadata.json.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Where to write the human-readable summary report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write metric deltas.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best/worst fine-tuning examples to include.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    comparison_dir = args.comparison_dir
    return {
        "metrics": args.metrics or comparison_dir / "validation_metrics.csv",
        "predictions": args.predictions or comparison_dir / "validation_predictions.csv",
        "metadata": args.metadata or comparison_dir / "comparison_metadata.json",
        "markdown": args.output_markdown or comparison_dir / "summary_report.md",
        "csv": args.output_csv or comparison_dir / "summary_deltas.csv",
    }


def load_required_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_csv(path)


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_models(frame: pd.DataFrame, source_path: Path) -> None:
    models = set(frame["model"].unique())
    required = {BASE_MODEL_NAME, FINE_TUNED_MODEL_NAME}
    missing = required - models
    if missing:
        raise ValueError(f"{source_path} is missing model rows: {sorted(missing)}")


def add_derived_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["abs_bias_log_return"] = metrics["bias_log_return"].abs()
    if "q10_q90_coverage" in metrics.columns:
        metrics["q10_q90_coverage_gap"] = (
            metrics["q10_q90_coverage"] - COVERAGE_TARGET
        ).abs()
    return metrics


def compare_metric(base_value: float, fine_value: float, metric: str) -> tuple[float, float | None, str]:
    delta = fine_value - base_value
    if metric in LOWER_IS_BETTER:
        improvement = None if base_value == 0 else (base_value - fine_value) / abs(base_value) * 100
        winner = FINE_TUNED_MODEL_NAME if fine_value < base_value else BASE_MODEL_NAME
    elif metric in HIGHER_IS_BETTER:
        improvement = delta * 100
        winner = FINE_TUNED_MODEL_NAME if fine_value > base_value else BASE_MODEL_NAME
    elif metric == "q10_q90_coverage":
        base_gap = abs(base_value - COVERAGE_TARGET)
        fine_gap = abs(fine_value - COVERAGE_TARGET)
        improvement = None if base_gap == 0 else (base_gap - fine_gap) / base_gap * 100
        winner = FINE_TUNED_MODEL_NAME if fine_gap < base_gap else BASE_MODEL_NAME
    elif metric == "q10_q90_mean_width":
        # Narrower intervals are only useful when calibrated; width alone has no winner.
        improvement = None
        winner = "descriptive"
    else:
        improvement = None
        winner = "n/a"
    return delta, improvement, winner


def build_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = add_derived_metrics(metrics)
    metric_columns = [
        column
        for column in [
            "weighted_quantile_loss",
            "mean_pinball_loss",
            "mae_log_return",
            "rmse_log_return",
            "abs_bias_log_return",
            "directional_accuracy",
            "q10_q90_coverage",
            "q10_q90_mean_width",
        ]
        if column in metrics.columns
    ]

    rows = []
    for horizon in metrics["horizon"].unique():
        horizon_metrics = metrics[metrics["horizon"] == horizon]
        indexed = horizon_metrics.set_index("model")
        if BASE_MODEL_NAME not in indexed.index or FINE_TUNED_MODEL_NAME not in indexed.index:
            continue
        for metric in metric_columns:
            base_value = float(indexed.loc[BASE_MODEL_NAME, metric])
            fine_value = float(indexed.loc[FINE_TUNED_MODEL_NAME, metric])
            delta, improvement_pct, winner = compare_metric(base_value, fine_value, metric)
            rows.append(
                {
                    "horizon": horizon,
                    "metric": metric,
                    "base_pretrained": base_value,
                    "fine_tuned": fine_value,
                    "delta_fine_minus_base": delta,
                    "improvement_pct": improvement_pct,
                    "winner": winner,
                }
            )
    return pd.DataFrame(rows)


def build_prediction_delta_table(predictions: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["window", "horizon", "date", "actual"]
    base = predictions[predictions["model"] == BASE_MODEL_NAME][
        key_columns + ["prediction", "abs_error", "correct_direction"]
    ].rename(
        columns={
            "prediction": "base_prediction",
            "abs_error": "base_abs_error",
            "correct_direction": "base_correct_direction",
        }
    )
    fine = predictions[predictions["model"] == FINE_TUNED_MODEL_NAME][
        key_columns + ["prediction", "abs_error", "correct_direction"]
    ].rename(
        columns={
            "prediction": "fine_prediction",
            "abs_error": "fine_abs_error",
            "correct_direction": "fine_correct_direction",
        }
    )
    merged = base.merge(fine, on=key_columns, how="inner", validate="one_to_one")
    merged["abs_error_delta_fine_minus_base"] = (
        merged["fine_abs_error"] - merged["base_abs_error"]
    )
    merged["fine_improved_abs_error"] = merged["abs_error_delta_fine_minus_base"] < 0
    return merged


def format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No rows available."]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(format_number(row[column]) for column in columns) + " |")
    return lines


def overall_verdict(delta_table: pd.DataFrame) -> str:
    overall = delta_table[delta_table["horizon"].astype(str) == "overall"]
    if overall.empty:
        return "No overall row was found, so no overall verdict is available."

    mae = overall[overall["metric"] == "mae_log_return"]
    rmse = overall[overall["metric"] == "rmse_log_return"]
    direction = overall[overall["metric"] == "directional_accuracy"]
    parts = []
    quantile = overall[overall["metric"] == "weighted_quantile_loss"]
    if not quantile.empty:
        row = quantile.iloc[0]
        parts.append(f"Quantile-loss winner: {row['winner']} ({format_number(row['improvement_pct'])}% fine-tuning improvement).")
    if not mae.empty:
        row = mae.iloc[0]
        parts.append(
            f"MAE winner: {row['winner']} ({format_number(row['improvement_pct'])}% improvement)."
        )
    if not rmse.empty:
        row = rmse.iloc[0]
        parts.append(
            f"RMSE winner: {row['winner']} ({format_number(row['improvement_pct'])}% improvement)."
        )
    if not direction.empty:
        row = direction.iloc[0]
        parts.append(
            f"Directional accuracy delta: {format_number(row['delta_fine_minus_base'] * 100)} percentage points."
        )
    return " ".join(parts)


def write_markdown_report(
    output_path: Path,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    delta_table: pd.DataFrame,
    prediction_deltas: pd.DataFrame,
    top_n: int,
) -> None:
    lines = ["# Chronos-2 Validation Summary", ""]
    if metadata.get("num_windows", 0) <= 5:
        lines.extend(["Infrastructure smoke test: the small sample does not establish forecasting skill.", ""])
    lines.extend(["## Verdict", "", overall_verdict(delta_table), ""])

    if metadata:
        config_rows = [
            ("model_id", metadata.get("model_id")),
            ("target_column", metadata.get("target_column")),
            ("prediction_length", metadata.get("prediction_length")),
            ("context_length", metadata.get("context_length")),
            ("train_rows", metadata.get("train_rows")),
            ("validation_rows", metadata.get("validation_rows_actual")),
            ("num_windows", metadata.get("num_windows")),
            ("finetune_mode", metadata.get("finetune_mode")),
            ("num_steps", metadata.get("num_steps")),
        ]
        lines.extend(["## Run Configuration", ""])
        for key, value in config_rows:
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    overall_metrics = metrics[metrics["horizon"].astype(str) == "overall"].copy()
    lines.extend(["## Overall Metrics", ""])
    lines.extend(
        markdown_table(
            overall_metrics,
            [
                "model",
                "rows",
                *[c for c in ("weighted_quantile_loss", "mean_pinball_loss", "q01_q99_coverage") if c in overall_metrics],
                "mae_log_return",
                "rmse_log_return",
                "bias_log_return",
                "directional_accuracy",
                "q10_q90_coverage",
                "q10_q90_mean_width",
            ],
        )
    )
    lines.append("")

    overall_deltas = delta_table[delta_table["horizon"].astype(str) == "overall"].copy()
    lines.extend(["## Overall Deltas", ""])
    lines.extend(
        markdown_table(
            overall_deltas,
            [
                "metric",
                "base_pretrained",
                "fine_tuned",
                "delta_fine_minus_base",
                "improvement_pct",
                "winner",
            ],
        )
    )
    lines.append("")

    horizon_deltas = delta_table[delta_table["horizon"].astype(str) != "overall"].copy()
    lines.extend(["## Horizon Deltas", ""])
    lines.extend(
        markdown_table(
            horizon_deltas,
            [
                "horizon",
                "metric",
                "base_pretrained",
                "fine_tuned",
                "delta_fine_minus_base",
                "improvement_pct",
                "winner",
            ],
        )
    )
    lines.append("")

    improved = prediction_deltas.sort_values("abs_error_delta_fine_minus_base").head(top_n)
    regressed = prediction_deltas.sort_values("abs_error_delta_fine_minus_base", ascending=False).head(top_n)
    example_columns = [
        "date",
        "horizon",
        "actual",
        "base_prediction",
        "fine_prediction",
        "base_abs_error",
        "fine_abs_error",
        "abs_error_delta_fine_minus_base",
    ]
    lines.extend(["## Best Fine-Tuning Improvements", ""])
    lines.extend(markdown_table(improved, example_columns))
    lines.append("")
    lines.extend(["## Worst Fine-Tuning Regressions", ""])
    lines.extend(markdown_table(regressed, example_columns))
    lines.append("")
    lines.extend(
        [
            "## Metric Notes",
            "",
            "- `mae_log_return` and `rmse_log_return`: lower is better.",
            "- `bias_log_return`: closer to 0 is better.",
            "- `directional_accuracy`: higher is better.",
            f"- `q10_q90_coverage`: closer to {COVERAGE_TARGET:.1f} is better for a calibrated 10%-90% interval.",
            "- `improvement_pct`: positive means fine-tuning improved the metric according to that metric's direction.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    metrics = load_required_csv(paths["metrics"], "validation metrics")
    predictions = load_required_csv(paths["predictions"], "validation predictions")
    metadata = load_metadata(paths["metadata"])
    require_models(metrics, paths["metrics"])
    require_models(predictions, paths["predictions"])

    metrics = add_derived_metrics(metrics)
    delta_table = build_delta_table(metrics)
    prediction_deltas = build_prediction_delta_table(predictions)

    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    delta_table.to_csv(paths["csv"], index=False)
    write_markdown_report(
        output_path=paths["markdown"],
        metadata=metadata,
        metrics=metrics,
        delta_table=delta_table,
        prediction_deltas=prediction_deltas,
        top_n=args.top_n,
    )

    print(f"Wrote metric deltas to {paths['csv']}")
    print(f"Wrote summary report to {paths['markdown']}")
    print("")
    print(overall_verdict(delta_table))


if __name__ == "__main__":
    main()
