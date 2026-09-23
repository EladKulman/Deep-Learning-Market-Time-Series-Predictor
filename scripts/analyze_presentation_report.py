"""Assemble the final report's evidence from saved runs, without training."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/qqq_report_support'
OUT.mkdir(parents=True, exist_ok=True)
WF = ROOT / 'docs/results/walk_forward'
OLD = ROOT / 'models/news-ablation-869989'
sources = set()

def read(p):
    p = Path(p); sources.add(p)
    return pd.read_csv(p)

def block_ci(matrix, length=5):
    """Resample circular blocks independently within each year, then average years."""
    rng = np.random.default_rng(20260916)
    draws = np.zeros(5000)
    for row in matrix:
        row = np.asarray(row); n = len(row)
        starts = rng.integers(0, n, size=(5000, int(np.ceil(n / length))))
        ix = ((starts[..., None] + np.arange(length)) % n).reshape(5000, -1)[:, :n]
        draws += row[ix].mean(axis=1) / len(matrix)
    return np.quantile(draws, [.025, .975]).tolist()

def main():
    w = read(WF/'window_losses.csv')
    keys = ['rung','fold','arm','model','window']
    assert not w.duplicated(keys).any()
    assert w.groupby(keys[:-1]).size().eq(50).all()
    error = 0.; widths = []; cov = []; count = 0
    for p in sorted((WF/'cells').glob('*/validation_metrics.csv')):
        name=p.parent.name; rung,tail=name.split('-fold'); year,arm=tail.split('-',1)
        if name.endswith('-bw4'): rung+='-bw4'
        arm=arm.split('-')[0]
        m=read(p); m=m[m.horizon.eq('overall')].set_index('model')
        for model,row in m.iterrows():
            sub=w[(w.rung==rung)&(w.fold==int(year))&(w.arm==arm)&(w.model==model)]
            if len(sub): error=max(error,abs(sub.wql_contrib.mean()-row.weighted_quantile_loss))
        if 'fine_tuned' in m.index:
            count+=1
            widths.append(m.loc['fine_tuned','q10_q90_mean_width']/m.loc['base_pretrained','q10_q90_mean_width'])
            cov.append([m.loc['base_pretrained','q10_q90_coverage'],m.loc['fine_tuned','q10_q90_coverage']])
    assert error<1e-10 and count==108
    pt=read(WF/'wql_pretrained.csv').rename(columns={'Unnamed: 0':'rung'}).set_index('rung')
    ft=read(WF/'wql_fine_tuned.csv').rename(columns={'Unnamed: 0':'rung'}).set_index('rung')
    agg=w.groupby(['rung','fold','model','window']).wql_contrib.mean()
    contrasts={}
    for key, left, lm, right, rm in [
        ('all_vs_control','r7_all_external-bw4','fine_tuned','r3_qqq_calendar_market-bw4','fine_tuned'),
        ('all_ft_gain','r7_all_external-bw4','fine_tuned','r7_all_external-bw4','base_pretrained'),
        ('control_vs_gaussian','r3_qqq_calendar_market','fine_tuned','r3_qqq_calendar_market','zero_gaussian')]:
        mat=[]
        for y in [2022,2023,2024,2025]: mat.append((agg.loc[left,y,lm]-agg.loc[right,y,rm]).to_numpy())
        contrasts[key]={'mean':float(np.mean(mat)), 'ci_block1':block_ci(mat,1),'ci_block5':block_ci(mat,5)}
    old={}
    for p in sorted(OLD.glob('*/validation_metrics.csv')):
        m=read(p); old[p.parent.name]=m[m.horizon.eq('overall')].set_index('model').to_dict('index')
    permutations={}
    for label in ['permutation_pretrained','permutation_control21']:
        d=read(ROOT/'docs/results'/label/'importance_by_window.csv')
        a=d.groupby(['feature','window']).delta_wql.mean()
        stats={}
        for f in a.index.get_level_values(0).unique():
            v=a.loc[f].to_numpy(); stats[f]={'mean':float(v.mean()),'ci':block_ci([v],1),'median':float(np.median(v))}
        permutations[label]=stats
    pairs=[]
    for p in sorted((ROOT/'docs/results/permutation_finetuned/runs').glob('*-finetuned/importance_by_window.csv')):
        q=Path(str(p).replace('-finetuned/','-pretrained/'))
        a=read(p).groupby('feature').delta_wql.mean(); b=read(q).groupby('feature').delta_wql.mean()
        pairs.append({'cell':p.parent.name,'correlation':float(a.corr(b)),'pre':b.to_dict(),'ft':a.to_dict()})
    vol=read(WF/'gain_by_window_volatility.csv')
    # Average repeated rungs for each observed week before describing terciles.
    vol=vol.groupby(['fold','window']).agg(gain=('gain','mean'),vol=('vol','first')).reset_index()
    vol['group']=pd.qcut(vol.vol,3,labels=['Quiet','Middle','Volatile'])
    vstats=vol.groupby('group',observed=True).agg(mean=('gain','mean'),n=('gain','size')).reset_index().to_dict('records')
    result={'old':old,'new_pt':pt.to_dict('index'),'new_ft':ft.to_dict('index'),
       'new_gain':read(WF/'finetune_gain.csv').query("fold == 'pooled'").to_dict('records'),
       'contrasts':contrasts,'permutations':permutations,'pairs':pairs,'volatility':vstats,
       'coverage_all108':np.mean(cov,axis=0).tolist(),'width_ratio_all108':float(np.mean(widths)),
       'coverage_by_year':read(WF/'coverage_direction.csv').to_dict('records'),
       'audit':{'fits_old':len(old),'fits_new':count,'metric_max_error':error,'new_window_rows':len(w),
          'ci_draws':5000,'ci_seed':20260916,'ci_note':'Seeds averaged before resampling; blocks drawn within each calendar year; no multiplicity adjustment.'}}
    sources.update([ROOT/'refrences/project presentation.pptx', ROOT/'refrences/Example Project.pdf', ROOT/'configs/walk_forward.json', ROOT/'configs/news_ablation.json'])
    result['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    (OUT/'evidence.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False))
    print(json.dumps({'audit':result['audit'],'contrasts':contrasts,'volatility':vstats},indent=2))

if __name__=='__main__':main()
