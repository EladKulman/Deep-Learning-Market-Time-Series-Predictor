#!/usr/bin/env python3
"""Run one reproducible news-source setup from the ablation plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_PLAN = Path("configs/news_ablation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=Path("models/news-ablation"))
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def setup_features(plan: dict, name: str) -> list[str]:
    setups = plan["setups"]
    if name not in setups:
        raise ValueError(f"Unknown setup {name!r}; choose one of {', '.join(setups)}")
    features = list(plan["control_features"])

    def add_setup(setup_name: str) -> None:
        setup = setups[setup_name]
        for included in setup.get("include_setups", []):
            add_setup(included)
        features.extend(setup.get("add_features", []))

    add_setup(name)
    return list(dict.fromkeys(features))


def comparison_command(args: argparse.Namespace, plan: dict, features: list[str], output: Path) -> list[str]:
    seed = args.seed if args.seed is not None else plan["seed"]
    command = [
        sys.executable,
        "-u",
        "scripts/compare_chronos2_validation.py",
        "--model-id", plan["model_id"],
        "--model-revision", plan["model_revision"],
        "--feature-columns", ",".join(features),
        "--start-date", plan["start_date"],
        "--end-date", plan["end_date"],
        "--prediction-length", str(plan["prediction_length"]),
        "--context-length", str(plan["context_length"]),
        "--validation-rows", str(plan["validation_rows"]),
        "--stride", str(plan["stride"]),
        "--num-steps", str(plan["num_steps"]),
        "--learning-rate", str(plan["learning_rate"]),
        "--train-batch-size", str(plan["train_batch_size"]),
        "--eval-batch-size", str(plan["eval_batch_size"]),
        "--seed", str(seed),
        "--device-map", args.device_map,
        "--output-dir", str(output),
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    return command


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    features = setup_features(plan, args.config_name)
    seed = args.seed if args.seed is not None else plan["seed"]
    output = args.output_root / f"{args.config_name}-seed{seed}"
    command = comparison_command(args, plan, features, output)
    print(json.dumps({
        "config_name": args.config_name,
        "description": plan["setups"][args.config_name]["description"],
        "features": features,
        "feature_count": len(features),
        "sample": [plan["start_date"], plan["end_date"]],
        "seed": seed,
        "output": str(output),
    }, indent=2))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
