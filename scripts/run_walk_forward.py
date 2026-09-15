#!/usr/bin/env python3
"""Run one (rung, fold, seed) cell of the walk-forward covariate ladder.

The plan is configs/walk_forward.json. A cell trains LoRA on every session before the
validation year and evaluates that year's non-overlapping five-session windows; with
--base-only it evaluates only the pretrained model and the Gaussian baseline (inference).

Examples
--------
    python scripts/run_walk_forward.py --rung r3_qqq_calendar_market --fold 2024 --seed 42
    python scripts/run_walk_forward.py --rung r7_all_external --fold 2022 --base-only
    python scripts/run_walk_forward.py --task-index 17            # Slurm array cell
    python scripts/run_walk_forward.py --list-tasks --base-only   # enumerate cells
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

DEFAULT_PLAN = Path("configs/walk_forward.json")
DEFAULT_DATA = Path("data/processed/daily_feature_table.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rung")
    parser.add_argument("--fold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-index", type=int, help="Enumerate (rung, fold[, seed]) cells and run this one")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--base-only", action="store_true", help="Pretrained + Gaussian baseline only; no fine-tuning, seeds ignored")
    parser.add_argument("--output-root", type=Path, default=Path("models/walk-forward"))
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def rung_features(plan: dict, rung: str) -> list[str]:
    if rung not in plan["rungs"]:
        raise ValueError(f"Unknown rung {rung!r}; choose one of {', '.join(plan['rungs'])}")
    features: list[str] = []
    for group in plan["rungs"][rung]["groups"]:
        features.extend(plan["feature_groups"][group])
    return list(dict.fromkeys(features))


def enumerate_tasks(plan: dict, base_only: bool) -> list[tuple[str, str, int | None]]:
    seeds = [None] if base_only else list(plan["seeds"])
    return [(rung, fold, seed) for rung in plan["rungs"] for fold in plan["folds"] for seed in seeds]


def fold_bounds(plan: dict, fold: str, data: Path) -> dict:
    spec = plan["folds"][fold]
    dates = pd.to_datetime(pd.read_csv(data, usecols=["date"])["date"])
    start, end = pd.Timestamp(plan["sample_start"]), pd.Timestamp(spec["validation_end"])
    in_sample = dates[(dates >= start) & (dates <= end)]
    validation = in_sample[in_sample >= pd.Timestamp(spec["validation_start"])]
    if validation.empty:
        raise ValueError(f"No sessions in the validation year for fold {fold}")
    return {
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": validation.max().strftime("%Y-%m-%d"),
        "validation_rows": int(len(validation)),
        "train_rows": int(len(in_sample) - len(validation)),
        "validation_first": validation.min().strftime("%Y-%m-%d"),
    }


def comparison_command(args, plan, features, bounds, seed, output: Path) -> list[str]:
    command = [
        sys.executable, "-u", "scripts/compare_chronos2_validation.py",
        "--data", str(args.data),
        "--model-id", plan["model_id"],
        "--model-revision", plan["model_revision"],
        "--feature-columns", ",".join(features) if features else "none",
        "--start-date", bounds["start_date"],
        "--end-date", bounds["end_date"],
        "--prediction-length", str(plan["prediction_length"]),
        "--context-length", str(plan["context_length"]),
        "--validation-rows", str(bounds["validation_rows"]),
        "--stride", str(plan["stride"]),
        "--num-steps", str(plan["num_steps"]),
        "--learning-rate", str(plan["learning_rate"]),
        "--train-batch-windows", str(plan["train_batch_windows"]),
        "--eval-batch-size", str(plan["eval_batch_size"]),
        "--seed", str(seed if seed is not None else plan["seeds"][0]),
        "--device-map", args.device_map,
        "--output-dir", str(output),
    ]
    if args.base_only:
        command.append("--base-only")
    if args.prepare_only:
        command.append("--prepare-only")
    return command


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = enumerate_tasks(plan, args.base_only)
    if args.list_tasks:
        for index, (rung, fold, seed) in enumerate(tasks):
            print(index, rung, fold, seed if seed is not None else "pretrained")
        print(f"{len(tasks)} tasks")
        return
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise SystemExit(f"--task-index must be in [0, {len(tasks) - 1}]")
        args.rung, args.fold, args.seed = tasks[args.task_index]
    if not args.rung or not args.fold:
        raise SystemExit("Provide --rung and --fold, or --task-index")
    if str(args.fold) not in plan["folds"]:
        raise SystemExit(f"Unknown fold {args.fold}; choose one of {', '.join(plan['folds'])}")

    features = rung_features(plan, args.rung)
    bounds = fold_bounds(plan, str(args.fold), args.data)
    seed = None if args.base_only else (args.seed if args.seed is not None else plan["seeds"][0])
    cell = f"{args.rung}-fold{args.fold}-" + ("pretrained" if args.base_only else f"seed{seed}")
    output = args.output_root / cell
    print(json.dumps({
        "rung": args.rung, "description": plan["rungs"][args.rung]["description"],
        "fold": args.fold, "seed": seed, "base_only": args.base_only,
        "features": features, "feature_count": len(features), **bounds, "output": str(output),
    }, indent=2))
    subprocess.run(comparison_command(args, plan, features, bounds, seed, output), check=True)


if __name__ == "__main__":
    main()
