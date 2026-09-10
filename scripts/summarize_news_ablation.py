#!/usr/bin/env python3
"""Rank completed news-ablation runs and compute paired window uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-aggregate-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def run_identity(path: Path) -> tuple[str, int]:
    metadata = json.loads((path / "comparison_metadata.json").read_text())
    seed = int(metadata["seed"])
    suffix = f"-seed{seed}"
    if not path.name.endswith(suffix):
        raise ValueError(f"Run directory does not end in {suffix}: {path}")
    return path.name[: -len(suffix)], seed


def overall_row(path: Path, model: str) -> pd.Series:
    metrics = pd.read_csv(path / "validation_metrics.csv", dtype={"horizon": str})
    rows = metrics[(metrics["model"] == model) & (metrics["horizon"] == "overall")]
    if len(rows) != 1:
        raise ValueError(f"Expected one {model} overall row in {path}")
    return rows.iloc[0]


def window_losses(path: Path, model: str) -> pd.Series:
    predictions = pd.read_csv(path / "validation_predictions.csv")
    predictions = predictions[predictions["model"] == model].copy()
    qcols = [
        column for column in predictions
        if column.startswith("q") and column[1:].replace(".", "", 1).isdigit()
    ]
    losses = []
    actual = predictions["actual"].to_numpy()
    for column in qcols:
        quantile = float(column[1:])
        error = actual - predictions[column].to_numpy()
        losses.append(np.maximum(quantile * error, (quantile - 1) * error))
    predictions["mean_pinball"] = np.mean(losses, axis=0)
    denominator = float(predictions["actual"].abs().mean())
    predictions["wql_component"] = 2 * predictions["mean_pinball"] / denominator
    return predictions.groupby("window")["wql_component"].mean()


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if samples <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def collect(root: Path, control_name: str, bootstrap_samples: int) -> pd.DataFrame:
    paths = sorted(path.parent for path in root.glob("*-seed*/validation_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No completed runs found under {root}")

    identities = {run_identity(path): path for path in paths}
    controls = {seed: path for (name, seed), path in identities.items() if name == control_name}
    rows: list[dict] = []
    for (name, seed), path in identities.items():
        metadata = json.loads((path / "comparison_metadata.json").read_text())
        fine = overall_row(path, "fine_tuned")
        base = overall_row(path, "base_pretrained")
        row = {
            "setup": name,
            "seed": seed,
            "feature_count": len(metadata["feature_columns"]),
            "weighted_quantile_loss": float(fine["weighted_quantile_loss"]),
            "wql_delta_vs_own_base": float(fine["weighted_quantile_loss"] - base["weighted_quantile_loss"]),
            "mae_log_return": float(fine["mae_log_return"]),
            "rmse_log_return": float(fine["rmse_log_return"]),
            "directional_accuracy": float(fine["directional_accuracy"]),
            "q10_q90_coverage": float(fine["q10_q90_coverage"]),
            "q01_q99_coverage": float(fine["q01_q99_coverage"]),
        }
        control_path = controls.get(seed)
        if control_path is not None:
            control = overall_row(control_path, "fine_tuned")
            paired = window_losses(path, "fine_tuned").subtract(
                window_losses(control_path, "fine_tuned"), fill_value=np.nan
            ).dropna()
            low, high = bootstrap_interval(
                paired.to_numpy(), bootstrap_samples, seed=20260908 + seed
            )
            row.update({
                "wql_delta_vs_control": float(
                    fine["weighted_quantile_loss"] - control["weighted_quantile_loss"]
                ),
                "paired_window_win_rate_vs_control": float((paired < 0).mean()),
                "paired_wql_delta_ci_low": low,
                "paired_wql_delta_ci_high": high,
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["weighted_quantile_loss", "setup", "seed"])


def aggregate_runs(
    frame: pd.DataFrame,
    root: Path,
    control_name: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    aggregate = (
        frame.groupby("setup", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_weighted_quantile_loss=("weighted_quantile_loss", "mean"),
            std_weighted_quantile_loss=("weighted_quantile_loss", "std"),
            mean_wql_delta_vs_control=("wql_delta_vs_control", "mean"),
            std_wql_delta_vs_control=("wql_delta_vs_control", "std"),
            seeds_beating_control=("wql_delta_vs_control", lambda values: int((values < 0).sum())),
            mean_wql_delta_vs_own_base=("wql_delta_vs_own_base", "mean"),
            mean_mae_log_return=("mae_log_return", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_q10_q90_coverage=("q10_q90_coverage", "mean"),
        )
    )
    paired_rows = []
    for setup, group in frame.groupby("setup"):
        paired_by_seed = []
        for seed in group["seed"]:
            setup_path = root / f"{setup}-seed{seed}"
            control_path = root / f"{control_name}-seed{seed}"
            if not control_path.exists():
                continue
            paired_by_seed.append(
                window_losses(setup_path, "fine_tuned").subtract(
                    window_losses(control_path, "fine_tuned"), fill_value=np.nan
                ).rename(seed)
            )
        if not paired_by_seed:
            continue
        paired = pd.concat(paired_by_seed, axis=1).mean(axis=1).dropna()
        low, high = bootstrap_interval(
            paired.to_numpy(), bootstrap_samples, seed=20260909
        )
        paired_rows.append({
            "setup": setup,
            "aggregate_paired_window_win_rate_vs_control": float((paired < 0).mean()),
            "aggregate_paired_wql_delta_ci_low": low,
            "aggregate_paired_wql_delta_ci_high": high,
        })
    return (
        aggregate.merge(pd.DataFrame(paired_rows), on="setup", how="left")
        .sort_values(["mean_weighted_quantile_loss", "setup"])
    )


def markdown_table(frame: pd.DataFrame) -> list[str]:
    shown = frame.copy()
    for column in shown.select_dtypes(include="number"):
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    lines = [
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in shown.astype(str).to_numpy())
    return lines


def write_report(frame: pd.DataFrame, aggregate: pd.DataFrame, output: Path) -> None:
    columns = [
        "setup", "seed", "weighted_quantile_loss", "wql_delta_vs_control",
        "wql_delta_vs_own_base", "mae_log_return", "rmse_log_return",
        "directional_accuracy", "q10_q90_coverage",
        "paired_window_win_rate_vs_control", "paired_wql_delta_ci_low",
        "paired_wql_delta_ci_high",
    ]
    shown = frame[[column for column in columns if column in frame]].copy()
    lines = [
        "# News-source ablation results", "",
        "Lower weighted quantile loss is better. Paired intervals resample the 50 "
        "non-overlapping forecast windows; an interval below zero favors the setup over control.", "",
        "## Aggregate across seeds", "",
    ]
    lines.extend(markdown_table(aggregate))
    lines.extend(["", "## Individual runs", ""])
    lines.extend(markdown_table(shown))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    frame = collect(args.root, args.control, args.bootstrap_samples)
    aggregate = aggregate_runs(
        frame, args.root, args.control, args.bootstrap_samples
    )
    output_csv = args.output_csv or args.root / "news_ablation_ranking.csv"
    output_aggregate_csv = args.output_aggregate_csv or args.root / "news_ablation_aggregate.csv"
    output_markdown = args.output_markdown or args.root / "news_ablation_report.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    aggregate.to_csv(output_aggregate_csv, index=False)
    write_report(frame, aggregate, output_markdown)
    print(frame.to_string(index=False))
    print("\nAggregate:\n" + aggregate.to_string(index=False))
    print(f"Wrote {output_csv}, {output_aggregate_csv}, and {output_markdown}")


if __name__ == "__main__":
    main()
