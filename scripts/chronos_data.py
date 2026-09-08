"""Shared data contract for training, rolling evaluation, and live forecasts.

Rows are consecutive observed NYSE sessions. Missing past covariates stay NaN;
future inputs contain only columns explicitly marked known_future in the group file.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from build_daily_feature_table import CORE_PAST, CORE_FUTURE


def add_feature_args(parser) -> None:
    parser.add_argument("--feature-profile", choices=["core", "full", "target"], default="core")
    parser.add_argument("--feature-groups", help="Optional group filter, e.g. qqq,calendar,market,rates")
    parser.add_argument("--groups-file", type=Path, help="Defaults to <data-stem>_groups.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-revision", help="Optional Hugging Face commit SHA for a reproducible base model")


def load_groups(data: Path, groups_file: Path | None = None) -> dict:
    path = groups_file or data.with_name(data.stem + "_groups.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing feature roles: {path}. Rebuild the feature table first.")
    return json.loads(path.read_text())


def select_features(frame: pd.DataFrame, args) -> tuple[list[str], list[str]]:
    groups = load_groups(args.data, args.groups_file)
    explicit = args.feature_columns
    if explicit is not None:
        features = [] if explicit.strip().lower() in {"none", "target"} else [c.strip() for c in explicit.split(",") if c.strip()]
    elif args.feature_profile == "target":
        features = []
    elif args.feature_profile == "core":
        features = list(CORE_PAST + CORE_FUTURE)
        if getattr(args, "smoke_test", False):
            small = {"parkinson_vol_1d", "vxn_log_chg", "epu_log", "fomc_net_hawkish_ewma", "is_fomc_day", "is_cpi_day"}
            features = [c for c in features if c in small]
    else:
        features = [c for c, g in groups.items() if g["role"] in {"past", "known_future"}]
    if len(features) != len(set(features)):
        raise ValueError("Feature columns must be unique")
    missing = set(features) - set(frame)
    if missing:
        raise ValueError(f"Features absent from current data: {sorted(missing)}. Old checkpoints need retraining on this schema.")
    invalid = [c for c in features if c not in groups or groups[c]["role"] not in {"past", "known_future"}]
    if invalid:
        raise ValueError(f"Target/raw/unclassified columns cannot be covariates: {invalid}")
    if args.feature_groups:
        selected_groups = set(args.feature_groups.split(","))
        unknown = selected_groups - {g["group"] for g in groups.values()}
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
        features = [c for c in features if groups[c]["group"] in selected_groups]
    excluded = set((args.exclude_columns or "").split(","))
    features = [c for c in features if c not in excluded]
    known = [c for c in features if groups[c]["role"] == "known_future"]
    validate_values(frame, args.timestamp_column, args.target_column, features, known)
    return features, known


def validate_values(frame, timestamp_column, target_column, features, known=()) -> None:
    dates = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Modeling dates must be valid, unique, and sorted")
    values = frame[[target_column, *features]].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite modeling values are not allowed")
    if frame[target_column].isna().any():
        raise ValueError("Missing target values would break the trading-session sequence")
    if known and frame[list(known)].isna().any().any():
        raise ValueError("Known-future features are incomplete in the modeling table")


def forecast_input(context, future, target_column, features, known) -> dict:
    result = {"target": context[target_column].to_numpy(dtype="float32")}
    if features:
        result["past_covariates"] = {c: context[c].to_numpy(dtype="float32") for c in features}
    if known:
        if future is None or not set(known) <= set(future):
            raise ValueError("Forecast horizon is missing required known-future columns")
        if not np.isfinite(future[list(known)].to_numpy(dtype=float)).all():
            raise ValueError("Forecast calendar is incomplete; refresh the event schedules before predicting")
        result["future_covariates"] = {c: future[c].to_numpy(dtype="float32") for c in known}
    return result


def training_inputs(frame, target_column, timestamp_column, features, known, prediction_length):
    from chronos.chronos2.preprocess import from_data_frame
    source = frame[[timestamp_column, target_column, *features]].copy()
    source["item_id"] = frame["item_id"].to_numpy() if "item_id" in frame else "QQQ"
    return from_data_frame(source, target_columns=[target_column], prediction_length=prediction_length,
                           known_covariates_names=list(known), id_column="item_id",
                           timestamp_column=timestamp_column)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(args, features, known) -> dict:
    groups_path = args.groups_file or args.data.with_name(args.data.stem + "_groups.json")
    versions = {}
    for name in ("chronos-forecasting", "torch", "transformers", "peft", "pandas", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"schema_version": 2, "feature_profile": args.feature_profile,
            "model_revision": args.model_revision,
            "feature_columns": list(features), "known_covariates_names": list(known),
            "data_path": str(args.data), "data_sha256": fingerprint(args.data),
            "groups_sha256": fingerprint(groups_path), "seed": args.seed,
            "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
            "script_sha256": {p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "package_versions": versions}
