"""Explicitly refresh report figures from archived evidence; never run during PDF compilation."""
from pathlib import Path
import json
import build_report  # Reuse local dependency discovery and cache settings.
import presentation_report_layout as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
E=json.loads((ROOT/'output/pdf/qqq_report_support/evidence.json').read_text())
r.TMP=ROOT/'report/figures'
r.TMP.mkdir(parents=True,exist_ok=True)
r.plot_settings()

def charts():
    paths={}
    # Four strictly non-overlapping calendar-year evaluations, using the same cumulative ladder.
    names=['r0_target','r1_qqq','r2_qqq_calendar','r3_qqq_calendar_market','r4_plus_uncertainty','r5_plus_fed','r6_plus_gdelt_fed_recession']
    f,axs=plt.subplots(2,2,figsize=(6.5,3.5),sharex=True,sharey=True)
    for ax,y in zip(axs.flat,['2022','2023','2024','2025']):
        ax.plot(range(7),[E['new_pt'][n][y] for n in names],':o',color=r.GRAY,ms=3,label='Pretrained')
        ax.plot(range(7),[E['new_ft'][n][y] for n in names],'-o',color=r.TEAL,ms=3,label='LoRA, mean of 3 seeds')
        ax.set_title(y,loc='left');ax.set_xticks(range(7),['Target','QQQ','Cal.','Market','EPU','FOMC','GDELT'],rotation=30)
        ax.set_ylim(.675,.725);ax.grid(axis='y');ax.set_ylabel('WQL')
    axs[0,0].legend(fontsize=6.5,frameon=False);f.tight_layout(pad=.5)
    paths['ladder']=r.savefig(f,'ladder')
    f,ax=plt.subplots(figsize=(6.5,2.45))
    select=[('vxn_log_chg','VXN change'),('overnight_gap','Overnight gap'),('d_dfii10_bp','Real-yield change'),('tlt_log_ret','TLT return'),('gdelt_fed_avg_tone','GDELT Fed tone'),('gdelt_recession_news_share','Recession news share'),('gdelt_ai_avg_tone','GDELT AI tone')]
    for i,(key,label) in enumerate(select):
        d=E['permutations']['permutation_pretrained'][key];m=d['mean']*1000;lo,hi=np.array(d['ci'])*1000
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=r.BLUE,capsize=3,ms=4)
    ax.axvline(0,color=r.GRAY,ls='--',lw=.8);ax.set_yticks(range(len(select)),[v for k,v in select]);ax.invert_yaxis();ax.set_xlabel('Scrambled minus original WQL x 1,000');ax.grid(axis='x');f.tight_layout(pad=.3)
    paths['perm']=r.savefig(f,'permutation')
    f,(ax,bx)=plt.subplots(1,2,figsize=(6.5,2.4))
    cov=pd.DataFrame(E['coverage_by_year']);cov=cov[cov.rung.str.match(r'^r[0-6]_')&~cov.rung.str.endswith('-bw4')]
    for model,color,label in [('base_pretrained',r.GRAY,'Pretrained'),('fine_tuned',r.TEAL,'LoRA')]:
        vals=cov[cov.model.eq(model)].groupby('fold').coverage_10_90.mean()
        ax.plot(range(4),vals*100,'o-',color=color,label=label,ms=4)
    ax.axhline(80,color=r.GOLD,ls='--',lw=.8);ax.set_xticks(range(4),['2022','2023','2024','2025']);ax.set_ylim(72,85);ax.set_ylabel('80% interval coverage (%)');ax.legend(frameon=False,fontsize=7);ax.grid(axis='y')
    vals=[v['mean']*1000 for v in E['volatility']];bx.bar(['Quiet','Middle','Volatile'],vals,color=[r.GOLD,r.GRAY,r.TEAL]);bx.axhline(0,color=r.GRAY,lw=.8);bx.set_ylabel('WQL difference x 1,000');bx.set_xlabel('Realized forecast-week movement');bx.grid(axis='y');f.tight_layout(pad=.4,w_pad=2)
    paths['cal']=r.savefig(f,'calibration')
    f,ax=plt.subplots(figsize=(6.5,2.15))
    p=pd.read_csv(ROOT/'models/news-ablation-869989/gdelt_fed-seed42/validation_predictions.csv');p=p[p.model.eq('fine_tuned')&p.window.eq(0)]
    xx=np.arange(5);ax.fill_between(xx,p['q0.1']*100,p['q0.9']*100,color='#DCEBE9',label='10%-90% interval');ax.plot(xx,p.prediction*100,'o-',color=r.TEAL,label='Median');ax.plot(xx,p.actual*100,'o-',color=r.NAVY,label='Realized');ax.axhline(0,color=r.GRAY,lw=.6);ax.set_xticks(xx,[pd.Timestamp(v).strftime('%b %d') for v in p.date]);ax.set_ylabel('Daily log return (%)');ax.set_title('All five forecasts issued after the June 27, 2025 close',fontsize=9,loc='left');ax.legend(frameon=False,ncol=3,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.1));f.tight_layout(pad=.4)
    paths['example']=r.savefig(f,'forecast_example')
    pair=next(p for p in E['pairs'] if p['cell'].startswith('r7_all_external-fold2025'))
    selected=[('vxn_log_chg','VXN change'),('overnight_gap','Overnight gap'),('tlt_log_ret','TLT return'),('d_dgs2_bp','2-year yield change'),('epu_log','Policy uncertainty'),('fomc_net_hawkish_ewma','FOMC smoothed tone'),('gdelt_fed_avg_tone','GDELT Fed tone'),('gdelt_ai_avg_tone','GDELT AI tone'),('gdelt_recession_news_share','Recession news share'),('sec_total_filings','SEC filing count')]
    f,ax=plt.subplots(figsize=(6.5,3.35)); yy=np.arange(len(selected))
    ax.barh(yy-.17,[pair['pre'][k]*1000 for k,l in selected],height=.31,color=r.GRAY,label='Pretrained')
    ax.barh(yy+.17,[pair['ft'][k]*1000 for k,l in selected],height=.31,color=r.TEAL,label='LoRA')
    ax.set_yticks(yy,[l for k,l in selected]);ax.invert_yaxis();ax.axvline(0,color=r.NAVY,lw=.8)
    ax.set_xlabel('Scrambled minus original WQL x 1,000');ax.grid(axis='x');ax.legend(frameon=False,loc='lower left');f.tight_layout(pad=.4)
    paths['paired']=r.savefig(f,'paired_permutation_2025')
    return paths

if __name__ == "__main__":
    for path in charts().values():
        print(path)
