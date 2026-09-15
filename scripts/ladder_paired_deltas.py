#!/usr/bin/env python3
"""Paired bootstrap deltas versus the control rung for the zero-shot ladder (fold-level normalization).

Fixes the earlier inline computation, which did not filter the `gaussian` rows stacked inside
r0_target_predictions.csv. Writes paired_deltas_vs_control.csv and paired_deltas_pooled.csv.
"""
import glob, os
import numpy as np, pandas as pd
ROOT = "docs/results/pretrained_ladder"; FOLDS = ["2022", "2023", "2024", "2025", "2025H2_2026H1"]
rng = np.random.default_rng(0)

def pinball(df):
    q = [c for c in df.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    out = {}
    for w, g in df.groupby("window"):
        out[w] = np.mean([np.maximum(float(c[1:]) * (g.actual - g[c]), (float(c[1:]) - 1) * (g.actual - g[c])).to_numpy() for c in q])
    return pd.Series(out), df.actual.abs().mean()

def load(fold, rung):
    df = pd.read_csv(f"{ROOT}/{fold}/{rung}_predictions.csv")
    if "model" in df.columns:
        df = df[~df.model.str.contains("gaussian")] if rung != "gaussian" else df[df.model.str.contains("gaussian")]
    return df

rows, pooled = [], {}
for fold in FOLDS:
    ctrl, den = pinball(load(fold, "r3_control"))
    for path in sorted(glob.glob(f"{ROOT}/{fold}/*_predictions.csv")):
        rung = os.path.basename(path).replace("_predictions.csv", "")
        if rung == "r3_control": continue
        w, _ = pinball(load(fold, rung)); d = (2 * (w - ctrl) / den).dropna().to_numpy()
        boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        rows.append(dict(fold=fold, rung=rung, mean_delta_wql=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), win_share=(d < 0).mean(), n=len(d)))
        pooled.setdefault(rung, []).append(d)
res = pd.DataFrame(rows); res["sig"] = np.where(res.ci_hi < 0, "better", np.where(res.ci_lo > 0, "worse", ""))
res.to_csv(f"{ROOT}/paired_deltas_vs_control.csv", index=False)
pl = []
for rung, ds in pooled.items():
    d = np.concatenate(ds); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    pl.append(dict(rung=rung, n=len(d), mean_delta_wql=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), win_share=(d < 0).mean()))
pl = pd.DataFrame(pl).sort_values("mean_delta_wql"); pl.to_csv(f"{ROOT}/paired_deltas_pooled.csv", index=False)
print(pl.round(4).to_string(index=False))
