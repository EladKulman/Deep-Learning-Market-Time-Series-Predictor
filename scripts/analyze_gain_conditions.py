#!/usr/bin/env python3
"""Where does fine-tuning pay? Gain by realised volatility tercile and by horizon; writes figure F."""
import glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
def spearmanr(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank(); return (np.corrcoef(a, b)[0, 1],)
rng = np.random.default_rng(0)
w = pd.read_csv("docs/results/walk_forward/window_losses.csv"); w["fold"] = w.fold.astype(str)
folds = ["2022", "2023", "2024", "2025"]
vol = {}
for f in folds:
    p = pd.read_csv(f"docs/results/pretrained_ladder/{f}/r3_control_predictions.csv"); p = p[~p.model.str.contains("gaussian")]
    vol[f] = p.groupby("window").actual.apply(lambda s: s.abs().mean())
rows = []
for rung in ["r3_qqq_calendar_market", "r4_plus_uncertainty", "r5_plus_fed", "r6_plus_gdelt_fed_recession"]:
    for f in folds:
        ft = w[(w.rung == rung) & (w.fold == f) & (w.model == "fine_tuned")].groupby("window").wql_contrib.mean()
        pt = w[(w.rung == rung) & (w.fold == f) & (w.model == "base_pretrained")].groupby("window").wql_contrib.mean()
        for win, val in (ft - pt).dropna().items():
            rows.append(dict(rung=rung, fold=f, window=win, gain=val, vol=vol[f].loc[win]))
d = pd.DataFrame(rows); d["tercile"] = pd.qcut(d.vol, 3, labels=["quiet", "mid", "volatile"])
def ci(x):
    x = np.asarray(x); b = [rng.choice(x, len(x), replace=True).mean() for _ in range(3000)]
    return pd.Series(dict(mean=x.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), n=len(x)))
t = d.groupby("tercile", observed=True).gain.apply(ci).unstack()
print("fine-tuning gain by realised-volatility tercile (rungs r3-r6 pooled):"); print(t.round(4).to_string())
print("spearman gain vs vol:", {r: round(spearmanr(g.gain, g.vol)[0], 2) for r, g in d.groupby("rung")})
hr = []
for path in glob.glob("docs/results/walk_forward/cells/*-seed4*/validation_metrics.csv"):
    cell = path.split("/")[-2]
    if "bw4" in cell: continue
    m = pd.read_csv(path); m = m[m.horizon != "overall"]
    ft = m[m.model == "fine_tuned"].set_index("horizon").weighted_quantile_loss; pt = m[m.model == "base_pretrained"].set_index("horizon").weighted_quantile_loss
    for h in ft.index: hr.append(dict(cell=cell, horizon=int(h), gain=ft[h] - pt[h]))
hz = pd.DataFrame(hr).groupby("horizon").gain.apply(ci).unstack(); print("\ngain by horizon:"); print(hz.round(4).to_string())
gm = []
for f in folds:
    ft = w[(w.rung == "r3_qqq_calendar_market") & (w.fold == f) & (w.model == "fine_tuned")].groupby("window").wql_contrib.mean()
    ga = w[(w.rung == "r3_qqq_calendar_market") & (w.fold == f) & (w.model == "zero_gaussian")].groupby("window").wql_contrib.mean()
    gm.append((ft - ga).dropna().to_numpy())
print("\nfine-tuned control minus Gaussian, pooled:", ci(np.concatenate(gm)).round(4).to_dict())
d.to_csv("docs/results/walk_forward/gain_by_window_volatility.csv", index=False); hz.to_csv("docs/results/walk_forward/gain_by_horizon.csv")

BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"; cols = {"2022": "#2a78d6", "2023": "#1baf7a", "2024": "#eda100", "2025": "#e34948"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
sub = d[d.rung == "r3_qqq_calendar_market"]
for f in folds:
    s = sub[sub.fold == f]; ax1.scatter(s.vol * 100, s.gain * 1000, s=22, color=cols[f], alpha=.75, label=f, edgecolor="white", linewidth=.5)
ax1.axhline(0, color=INK, linewidth=1)
edges = [d.vol.min(), d.vol.quantile(1 / 3), d.vol.quantile(2 / 3), d.vol.max()]
for i, name in enumerate(["quiet", "mid", "volatile"]):
    m = t.loc[name, "mean"] * 1000; ax1.hlines(m, edges[i] * 100, edges[i + 1] * 100, color=INK, linewidth=2.5)
    ax1.text((edges[i] + edges[i + 1]) / 2 * 100, m + (1.5 if m > 0 else -2.5), f"{name} weeks: {m:+.1f}", ha="center", fontsize=9, color=INK)
ax1.set_xlabel("Realised volatility of the forecast week (mean |daily return|, %)"); ax1.set_ylabel("Fine-tuned minus pretrained loss (x1000)")
ax1.set_title("Fine-tuning helps in volatile weeks and hurts in quiet ones\nmarket-control model, 200 test weeks, 3 seeds averaged; black bars = tercile means (all rungs)", loc="left", fontsize=11)
ax1.legend(frameon=False, fontsize=9, title="validation year", title_fontsize=9); ax1.grid(alpha=.3); ax1.spines[["top", "right"]].set_visible(False)
ax2.errorbar(hz.index, hz["mean"] * 1000, yerr=[(hz["mean"] - hz.lo) * 1000, (hz.hi - hz["mean"]) * 1000], fmt="o-", color=BLUE, ecolor=MUTED, capsize=3, markersize=7)
ax2.axhline(0, color=INK, linewidth=1); ax2.set_xticks([1, 2, 3, 4, 5]); ax2.set_xlabel("Days ahead"); ax2.set_ylabel("Fine-tuned minus pretrained loss (x1000)")
ax2.set_title("The gain sits at one day ahead\nall feature sets, years and seeds; 95% intervals", loc="left", fontsize=11); ax2.grid(alpha=.3); ax2.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("docs/assets/presentation/F_where_finetuning_pays.png", dpi=200); print("wrote figure F")
