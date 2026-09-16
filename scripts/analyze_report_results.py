#!/usr/bin/env python3
"""Audit saved forecasts and reproduce the paper's descriptive sensitivity analyses.

No model is loaded or retrained. Bootstrap intervals are exploratory, conditional
on the observed seeds and fixed full-sample WQL denominator, and unadjusted for
multiple comparisons. Circular blocks retain consecutive forecast windows.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models/news-ablation-869989"
OUT = ROOT / "output/pdf/report_analysis.json"
Q = np.array([.01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
              .55, .60, .65, .70, .75, .80, .85, .90, .95, .99])
QC = [f"q{x:g}" for x in Q]
BOOTSTRAP_SEED = 20260910
SAMPLES = 20000


def score(f):
    y = f.actual.to_numpy()
    q = f[QC].to_numpy()
    e = y[:, None] - q
    loss = np.maximum(e * Q, e * (Q - 1)).mean(axis=1)
    pred = f.prediction.to_numpy()
    return {
        "n": len(y), "wql": float(2 * loss.mean() / np.abs(y).mean()),
        "mae_pp": float(np.abs(pred - y).mean() * 100),
        "rmse_pp": float(np.sqrt(np.mean((pred - y) ** 2)) * 100),
        "direction_correct": int(((pred >= 0) == (y >= 0)).sum()),
        "predicted_down": int((pred < 0).sum()),
        "coverage80": float(((y >= q[:, 2]) & (y <= q[:, 18])).mean()),
        "coverage98": float(((y >= q[:, 0]) & (y <= q[:, 20])).mean()),
        "width80_pp": float((q[:, 18] - q[:, 2]).mean() * 100),
        "crossed_forecasts": int((np.diff(q, axis=1) < -1e-8).any(axis=1).sum()),
        "max_adjacent_crossing_pp": float(max(0, -np.diff(q, axis=1).min()) * 100),
        "calibration": (y[:, None] <= q).mean(axis=0).tolist(),
    }


def window_components(f):
    y = f.actual.to_numpy()
    e = y[:, None] - f[QC].to_numpy()
    v = np.maximum(e * Q, e * (Q - 1)).mean(axis=1) * 2 / np.abs(y).mean()
    return pd.Series(v, index=f.window).groupby(level=0).mean().to_numpy()


def interval(values, length=1, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, (SAMPLES, math.ceil(n / length)))
    indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(SAMPLES, -1)[:, :n]
    draws = np.asarray(values)[indices].mean(axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def count_parameters(path):
    # Read the safetensors header only, avoiding model loading and GPU use.
    with path.open("rb") as stream:
        n = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(n))
    return sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def analyze():
    frames, rows, hashes = {}, {}, {}
    reference = None
    expected = None
    for path in sorted(RUNS.glob("*-seed*/validation_predictions.csv")):
        name = path.parent.name
        meta = json.loads((path.parent / "comparison_metadata.json").read_text())
        contract = {k:meta[k] for k in ["model_revision", "data_sha256", "groups_sha256",
            "train_rows", "train_start", "train_end", "validation_start", "validation_end",
            "prediction_length", "context_length", "num_steps", "learning_rate", "stride",
            "train_batch_size", "eval_batch_size", "num_windows", "quantiles"]}
        if expected is None:
            expected = contract
        assert contract == expected, f"Experiment contract differs: {name}"
        data = pd.read_csv(path)
        saved = pd.read_csv(path.parent / "validation_metrics.csv", dtype={"horizon": str})
        frames[name] = {}
        rows[name] = {"features": len(meta["feature_columns"]), "seed": meta["seed"]}
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for model, frame in data.groupby("model"):
            frame = frame.sort_values(["window", "horizon"]).reset_index(drop=True)
            assert len(frame) == 250 and frame.date.nunique() == 250
            assert (frame.groupby("window").size() == 5).all()
            assert frame.window.nunique() == 50
            assert np.isfinite(frame[QC].to_numpy()).all()
            keys = frame[["window", "horizon", "date", "forecast_origin", "actual"]]
            if reference is None:
                reference = keys
            pd.testing.assert_frame_equal(reference, keys)
            metric = score(frame)
            saved_row = saved[(saved.model == model) & (saved.horizon == "overall")].iloc[0]
            assert abs(metric["wql"] - saved_row.weighted_quantile_loss) < 1e-10
            assert abs(metric["mae_pp"] / 100 - saved_row.mae_log_return) < 1e-10
            assert metric["direction_correct"] == round(saved_row.directional_accuracy * 250)
            frames[name][model] = frame
            rows[name][model] = metric
        seed = meta["seed"]
        setup = name.rsplit("-seed", 1)[0]
        cfg = json.loads((ROOT / "configs/news_ablation.json").read_text())
        add = cfg["setups"][setup].get("add_features", [])
        if setup == "all_external":
            add = sum((cfg["setups"][x]["add_features"] for x in cfg["setups"][setup]["include_setups"]), [])
        assert meta["feature_columns"] == cfg["control_features"] + add
        rows[name]["by_horizon"] = {str(h):score(f) for h, f in frames[name]["fine_tuned"].groupby("horizon")}
    assert len(rows) == 18
    for setup in ["control", "uncertainty", "all_external"]:
        for seed in [43, 44]:
            np.testing.assert_allclose(frames[f"{setup}-seed42"]["base_pretrained"][QC],
                                       frames[f"{setup}-seed{seed}"]["base_pretrained"][QC], rtol=0, atol=0)
    paired = {}
    for name in rows:
        seed = rows[name]["seed"]
        d = window_components(frames[name]["fine_tuned"]) - window_components(frames[f"control-seed{seed}"]["fine_tuned"])
        paired[name] = {"mean_delta":float(d.mean()), "winning_windows":int((d < 0).sum()),
                        "ci95": {str(b):interval(d, b) for b in [1, 2, 5, 10]},
                        "components":d.tolist()}
    direct = {}
    for other in ["gdelt_recession", "all_external", "gdelt_all"]:
        d = window_components(frames["gdelt_fed-seed42"]["fine_tuned"]) - window_components(frames[f"{other}-seed42"]["fine_tuned"])
        direct[other] = {"mean_delta":float(d.mean()), "ci95":{str(b):interval(d,b) for b in [1,2,5,10]}}
    aggregate = {}
    for setup in ["control", "uncertainty", "all_external"]:
        names = [f"{setup}-seed{s}" for s in [42,43,44]]
        wql = [rows[n]["fine_tuned"]["wql"] for n in names]
        d = np.array([paired[n]["components"] for n in names]).mean(axis=0)
        aggregate[setup] = {"mean_wql":float(np.mean(wql)), "sd_wql":float(np.std(wql, ddof=1)),
            "delta":float(d.mean()), "ci95":{str(b):interval(d,b) for b in [1,2,5,10]},
            "seeds_beating_control":sum(paired[n]["mean_delta"] < 0 for n in names)}
        adapted = np.array([window_components(frames[n]["fine_tuned"]) -
            window_components(frames[n]["base_pretrained"]) for n in names]).mean(axis=0)
        aggregate[setup]["adaptation_vs_own_pretrained"] = {
            "mean_delta":float(adapted.mean()),"ci95":{str(b):interval(adapted,b) for b in [1,2,5,10]}}
    stats = {"actual_up":int((reference.actual >= 0).sum()), "actual_down":int((reference.actual < 0).sum()),
             "return_min_pp":float(reference.actual.min()*100), "return_max_pp":float(reference.actual.max()*100),
             "first_date":reference.date.min(), "last_date":reference.date.max()}
    # Compare coarse calendar halves and all five lead times without new forecasts.
    halves = {}
    for setup in ["control", "all_external", "gdelt_fed", "gdelt_recession"]:
        f = frames[f"{setup}-seed42"]["fine_tuned"]
        halves[setup] = {label:score(f.loc[mask]) for label, mask in {
            "first_25_windows":f.window < 25, "last_25_windows":f.window >= 25}.items()}
    feature_path = ROOT / "data/processed/daily_feature_table.csv"
    feature_audit = {"sha256":hashlib.sha256(feature_path.read_bytes()).hexdigest()}
    if feature_audit["sha256"] == expected["data_sha256"]:
        table = pd.read_csv(feature_path)
        subset = table[(table.date >= "2017-01-03") & (table.date <= "2026-06-30")]
        feature_audit.update({"all_rows":len(table), "sample_rows":len(subset),
            "gdelt_missing":{c:int(subset[c].isna().sum()) for c in subset if c.startswith("gdelt_")},
            "gdelt_missing_dates":subset.loc[subset.filter(regex="^gdelt_").isna().any(axis=1),"date"].tolist()})
        train = subset[subset.date <= "2025-06-27"]
        share = train.filter(regex="^gdelt_.*news_share$")
        feature_audit["training_news_share_correlations"] = share.corr().to_dict()
        feature_audit["training_news_observed_rows"] = len(share.dropna())
        f = frames["gdelt_fed-seed42"]["fine_tuned"].merge(
            table[["date", "is_fomc_day", "is_cpi_day", "is_nfp_day"]], on="date", validate="one_to_one")
        event = f[["is_fomc_day", "is_cpi_day", "is_nfp_day"]].sum(axis=1) > 0
        feature_audit["fed_scheduled_event_diagnostic"] = {
            "scheduled":score(f[event]), "other":score(f[~event])}
    adapter = RUNS / "gdelt_fed-seed42/finetuned/finetuned-ckpt/adapter_model.safetensors"
    base = ROOT / "models/huggingface/models--amazon--chronos-2/snapshots" / expected["model_revision"] / "model.safetensors"
    result = {"method":{"bootstrap_samples":SAMPLES,"seed":BOOTSTRAP_SEED,"circular_block_lengths_windows":[1,2,5,10],
        "normalization":"Fixed mean absolute return over all 250 evaluation observations; paired differences averaged over five-day forecast windows.",
        "inference_scope":"Exploratory percentile intervals; no multiplicity adjustment; seed-aggregate intervals condition on observed seeds."},
        "contract":expected,"runs":rows,"paired":paired,"direct_fed_minus":direct,"aggregate":aggregate,
        "sample":stats,"halves":halves,"feature_audit":feature_audit,
        "parameters":{"base":count_parameters(base),"adapter":count_parameters(adapter)},
        "input_sha256":hashes}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in ["sample","aggregate","parameters"]},indent=2))
    print(f"Verified 18 runs, 54 forecast sets, common labels, feature contracts and saved metrics; quantified quantile crossings. Wrote {OUT}")
    return result


if __name__ == "__main__":
    analyze()
