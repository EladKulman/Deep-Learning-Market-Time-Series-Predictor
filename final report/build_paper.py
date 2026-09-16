"""Build the final workshop paper from the two completed experiment rounds."""
from pathlib import Path
import ast, json, re, sys
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_project_report as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import BaseDocTemplate,Frame,PageTemplate,Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

E=json.loads((ROOT/'final report/evidence.json').read_text())
r.OUT=ROOT/'output/pdf/qqq_chronos2_final_report.pdf'
r.TMP=ROOT/'tmp/pdfs/final_report'
r.STORY=[]
r.TMP.mkdir(parents=True,exist_ok=True)
r.fonts();r.S=r.styles();r.plot_settings()
TEXT=[]
def add(t,style='body'):
    r.add(t,style);TEXT.append(re.sub('<[^>]+>','',t))
def h(t): add(t,'h1')
def sub(t):add(t,'h2')
def page():r.page();TEXT.append('\n---\n')
def tab(rows,widths,caption,font=9):
    r.table(rows,widths,caption,font=font)
    TEXT.append('\n'+'\n'.join(' | '.join(map(str,row)) for row in rows)+'\n'+caption)
def fig(p,caption):r.figure(p,caption);TEXT.append(caption)
def ref(n):return r.ref(n)
def num(x):return f'{x:.6f}'
def ci(x):return f'[{x[0]:+.5f}, {x[1]:+.5f}]'

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
    select=[('vxn_log_chg','VXN change'),('overnight_gap','Overnight gap'),('d_dfii10_bp','Real-yield change'),('tlt_log_ret','TLT return'),('days_to_fomc','Days to FOMC')]
    for i,(key,label) in enumerate(select):
        d=E['permutations']['permutation_control21'][key];m=d['mean']*1000;lo,hi=np.array(d['ci'])*1000
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
    return paths

def manuscript(P):
    r.STORY.append(Spacer(1,9*mm))
    add('Can News Improve QQQ Forecasts?','title')
    add('Covariate Ablations, LoRA Adaptation and<br/>Four-Year Evaluation with Chronos-2','subtitle')
    add('Elad Kulman and Tom Weitman','author')
    add('Group 13 - Workshop on Deep Learning - Tel Aviv University<br/>September 2026','meta')
    add('<b>Abstract.</b> We study whether public news-derived time series improve probabilistic forecasts of QQQ daily log returns. A publication-aware data pipeline combines market, macroeconomic, calendar, uncertainty, disclosure and news-topic variables. An initial 18-fit source screen yields small candidate improvements over a market control, including a 1.28% weighted quantile loss reduction for Federal Reserve news in one seed. A subsequent study evaluates 108 LoRA fits across four calendar years with larger, explicitly measured training batches. In that study, the all-external model improves from 0.703724 pretrained WQL to 0.698957 after adaptation, but does not outperform its identically trained control (0.697701). Its paired difference is +0.001257 with a confidence interval spanning zero. Across the trained configurations, intervals widen by about 7.8%, while permutation-importance profiles remain broadly similar before and after training. These results do not establish reliable incremental news value under the tested recipe. They also do not establish that news is intrinsically uninformative or that Chronos-2 cannot use it. We provide reproducible comparisons, investigate the limits of input perturbation, and distinguish development findings from an untouched prospective evaluation.','abstract')
    h('1  Introduction')
    add('Index-level returns aggregate responses to many firms, policy announcements and changing market conditions. The project asks whether broad news narratives supply information that is not already present in market prices and scheduled events. We use QQQ, an exchange-traded fund tracking the Nasdaq-100, as a single, consistently observed target. The objective is to evaluate daily return distributions and source contributions, rather than to attribute an index move to one headline.')
    add('The research questions are: <b>RQ1</b>, do news-derived features add value beyond a fixed market and calendar control? <b>RQ2</b>, does LoRA adaptation improve the same model supplied with the same features? <b>RQ3</b>, do input perturbations reveal a change in predictive reliance after adaptation? An accurate return distribution can represent uncertainty even when its median offers little directional discrimination.')
    add('Our contribution is an empirical study with explicit information-availability rules, a first-round source screen, a larger chronological replication, and diagnostic analysis of the saved forecasts. We use an existing foundation model and adaptation method. The original proposal considered Bitcoin and a Temporal Fusion Transformer; the implemented project retains the narrative-comparison question while using QQQ and Chronos-2 for a common pretrained baseline.')

    page();h('2  Data and information availability')
    add('The archived feature table contains 5,201 exchange sessions from January 2006 through September 4, 2026, with 2005 prices supplying indicator warm-up. Its 77 columns include the date and target, 55 past-covariate fields, 14 eligible future-calendar fields, and six raw fields. This is the available feature inventory, not the number of inputs in every experiment. GDELT availability limits all news-training comparisons to January 2017 onward. The selected control has 21 covariates and the all-external setup has 42, both excluding the target.')
    tab([['Source family','Selected representation','Role'],['QQQ and related markets','Returns, gaps, range volatility, volume, momentum, VXN, TLT, HYG','Past only'],['Rates and macroeconomics','Rate changes, curve slope, real yields, CPI growth','Past only'],['Event calendar','FOMC, CPI, payroll dates, month-end and option expiry','Known future'],['EPU and EMU','Log trailing seven-day uncertainty averages','Past only'],['FOMC statements','Classifier tone, change, smoothed state, elapsed time','Past only'],['SEC disclosures','Accepted filings and past earnings-event counts','Past only'],['GDELT topics','Normalized news share and average tone per topic','Past only']],[38,100,28],'Source families in the executed study. Appendix B gives the exact selected fields and topic definitions.',8.8)
    sub('2.1  Publication time determines the usable date')
    add('Every feature is assigned to the first permitted exchange close after its information becomes available. A July CPI observation is not known on July 1: its release arrives later. The pipeline uses ALFRED release vintages and computes CPI growth within the available vintage. This prevents the reference month from exposing an unreleased value to a historical forecast.')
    add('H.15 series use the next publication-business-day 16:15 release and then the next eligible exchange close. SEC filings use UTC acceptance timestamps, including early closes. Complete daily news aggregates are shifted by a calendar day, then rolled to exchange sessions; weekend observations are averaged. CBOE-derived inputs receive a conservative one-session lag. Only scheduled calendar information enters the future-covariate channel. Future news, prices and realized returns are masked.')
    sub('2.2  News construction and missing data')
    add(f'GDELT queries cover AI, semiconductors, the Federal Reserve, inflation, big-tech earnings and recession. Each topic contributes news share and average tone {ref(6)}. These are numeric summaries, not raw text embeddings or headlines. Queries overlap, including Nvidia in both AI and semiconductor searches. The original common sample has 13 missing sessions in each selected GDELT column (0.55%). Missing values remain masked rather than becoming zero news.')
    add(f'The statement scorer uses a pinned public RoBERTa FOMC classifier related to the task introduced by Shah, Paturi and Chava {ref(4)}. It summarizes 170 statements. Publication alignment does not solve every vintage problem: revised prices, some macro series, reconstructed schedules and retrospectively selected SEC firms remain limitations. The two experiment rounds record different feature-table hashes.')

    page();h('3  Model, adaptation and forecast protocol')
    sub('3.1  Foundation forecasting and related work')
    add(f'Chronos-2 supports forecasting with related series and covariates through group attention {ref(1)}. Its pretrained capability provides an inference-only reference before any QQQ adaptation. LoRA freezes base weights and trains low-rank updates {ref(2)}. Financial evaluations of time-series foundation models motivate testing domain-specific behavior directly rather than assuming that general forecasting performance transfers to daily equity returns {ref(3)}. Our single-ETF experiment is narrower than those benchmarks.')
    add('The pinned checkpoint contains 119,477,664 base parameters. Its configuration has 12 layers, width 768, 12 attention heads and 16-step patches. Each supplied numeric series is scaled over observed context and transformed with asinh. The model returns 21 quantiles at levels 0.01, 0.05, 0.10 through 0.95 in 0.05 increments, and 0.99. We cap historical context at 512 exchange sessions.')
    sub('3.2  LoRA and the training batch')
    r.equation(r'$W=W_0+(\alpha/r)BA,\quad r=8,\quad\alpha=16$',1)
    add('The adapters target attention query, key, value and output projections and the output patch projection. They contain 1,206,912 trainable parameters, approximately 1% of the combined model, with adapter dropout zero. Each run starts from the same original checkpoint, samples historical training slices, and optimizes quantile loss. We do not continue training a previous control adapter. No validation examples are supplied to fit(), and the final scheduled checkpoint is used without validation-based early stopping.')
    add('The Chronos training batch counts target and covariate series. In the first screen, a batch argument of eight admitted effectively one complete QQQ window with its covariates. The later runner requests a number of windows explicitly: eight windows with 21 covariates requires 176 series rows. This corrects an under-specified training budget. It does not by itself prove that the earlier performance differences were caused by batch size.')
    sub('3.3  What a five-day forecast means')
    r.equation(r'$r_t=\log(P_t^{adj}/P_{t-1}^{adj}),\qquad \widehat q_{t+h,\tau},\ h=1,\ldots,5$',2)
    add('At an origin t, the model receives observed history through t and forecasts five separate daily log returns, not one cumulative weekly return. Every lead uses the same information cutoff. After five exchange sessions, we refresh history with all five actual observations and issue the next forecast. Historical inputs overlap substantially; the five-day blocks of scored dates do not overlap. No predicted return replaces a realized observation. We freeze adapter weights throughout each evaluation year.')
    fig(P['example'],'One first-round Fed-topic forecast. The July 4 holiday shifts the fifth scored session to July 7. Later leads do not observe intervening news or returns. Inclusion inside a broad interval and a correct directional prediction are different events.')

    page();h('4  Experimental design and scoring')
    tab([['Property','Round A: source screen','Round B: chronological ladder'],['Training starts','January 3, 2017','January 3, 2017'],['Training ends','June 27, 2025','Last session before each test year'],['Evaluation','June 2025-June 2026','Calendar years 2022, 2023, 2024, 2025'],['Scored days','250 in 50 windows','250 per year, 1,000 distinct days'],['Completed LoRA fits','18','108 including matched controls'],['Updates and windows','200 updates, about 1 window each','500 x 8; all-external and matched control 1,000 x 4'],['Seeds','42; 43/44 for three source setups','42, 43, 44 for every trained setup']],[34,58,74],'The rounds answer related questions but do not isolate one training change. Both use context 512, five-step horizon, stride five and learning rate 1e-5.',8.7)
    add('Round A adds one source family or one GDELT topic to the same 21-feature control. Round B follows a cumulative ladder: target only; six QQQ transforms; six calendar flags; nine market/rate fields; EPU/EMU; FOMC tone; GDELT Fed and recession; then all remaining external fields. The resulting counts are 0, 6, 12, 21, 23, 27, 31 and 42 covariates. Seven rungs use eight windows for 500 steps (84 fits). The 42-feature rung and a matched control use four windows for 1,000 steps (24 fits), because eight all-external windows exceed the GPU memory budget.')
    add('A separate pretrained-only exploration includes four calendar years plus the original mid-2025 to mid-2026 interval. That fifth interval overlaps calendar 2025. Its intermediate source arms are not all cumulative. We use the four-year walk-forward arm for the main paired pretrained/LoRA comparison, avoiding both mixed feature definitions and treating overlapping intervals as independent years.')
    sub('4.1  Weighted quantile loss')
    r.equation(r'$\rho_\tau(u)=\max(\tau u,(\tau-1)u)$',3)
    r.equation(r'$\mathrm{WQL}=\frac{2}{|Q|\sum_{t=1}^{N}|r_t|}\sum_{\tau\in Q}\sum_{t=1}^{N}\rho_\tau(r_t-\widehat q_{t,\tau})$',4)
    add(f'WQL averages the asymmetric quantile loss across the 21 levels and normalizes by realized absolute returns. Lower is better; 0.70 does not mean 70% accuracy. Quantile scoring evaluates forecast distributions {ref(5)}. We also report median MAE, direction, and the coverage and width of the nominal 80% and 98% intervals. A zero-mean Gaussian with trailing 60-session sample standard deviation supplies quantiles without news or training. It forecasts daily, not cumulative, uncertainty.')
    sub('4.2  Pairing and uncertainty')
    add('We pair forecasts on the same realized dates and average seeds before comparing windows. Four-year estimates average fold-normalized losses, giving each calendar year equal weight. Published ladder intervals use 5,000 window-bootstrap draws. For three headline contrasts we additionally resample circular blocks of five consecutive forecast windows separately within each year (5,000 draws, seed 20260916). These intervals remain exploratory and unadjusted for multiple comparisons; they condition on observed years and seeds.')

    page();h('5  Initial news-source screen')
    add('The first round established end-to-end GPU execution and generated candidate source rankings. A separate five-update smoke test produced only 15 outcomes; it verified training and checkpoint reload, not forecasting skill. Table 3 summarizes the substantive single-topic comparison. All configurations retain the common market/calendar control.')
    rows=[['Added news','Fine-tuned WQL','Reduction vs control','Correct direction']]
    old=E['old'];con=old['control-seed42']['fine_tuned']['weighted_quantile_loss']
    for key,label in [('control','None'),('gdelt_fed','Federal Reserve'),('gdelt_recession','Recession'),('gdelt_semiconductor','Semiconductors'),('gdelt_ai','AI'),('gdelt_inflation','Inflation'),('all_external','All external sources'),('gdelt_big_tech_earnings','Big-tech earnings')]:
        m=old[key+'-seed42']['fine_tuned'];loss=m['weighted_quantile_loss']
        rows.append([label,num(loss),'--' if key=='control' else f'{100*(1-loss/con):.2f}%',f"{round(m['directional_accuracy']*250)}/250"])
    tab(rows,[51,35,43,37],'Round A, seed 42. The all-external arm includes non-GDELT sources. All-GDELT alone is a different 33-feature arm (WQL 0.692348).',9)
    add('Fed has the lowest point estimate, 1.28% below the fine-tuned control; recession differs by only 0.000269 WQL. A paired interval for their difference includes zero. The six topics each have one seed, so their apparent ranking does not demonstrate a uniquely informative narrative. Longer blocks in the first-round audit weaken several topic-versus-control comparisons as well.')
    sub('5.1  Replication changes the interpretation')
    tab([['Source setup','Mean LoRA WQL','Seed SD','Seeds beating control'],['Control','0.693582','0.001656','--'],['EPU / EMU','0.693473','0.005212','1 / 3'],['All external','0.689439','0.003583','2 / 3']],[51,40,35,40],'Round A seed repetitions. Each seed evaluates the same 250 market outcomes; these are not 750 independent days.',9)
    add('All-external improves mean loss by 0.60% relative to control, but its paired interval includes no gain. Its own pretrained WQL is 0.690730, so LoRA adds only a 0.19% relative improvement. EPU/EMU looks strong in the first seed and loses that advantage in repetitions. These observations motivated a larger training budget and evaluation across different years.')
    add('Direction is particularly weak evidence of skill in this interval: 143 of 250 actual returns are non-negative. Always predicting up scores 57.2%, versus 57.6% for Fed and 58.0% for all-external at seed 42. The Gaussian median is zero, which the scoring code counts as up. The result therefore supports candidate probabilistic comparisons, not a dependable trading rule.')
    sub('5.2  Why the later study is not a batch-only experiment')
    add('The original API batch argument gave much less training than intended. Round B increases both windows per update and update count, evaluates different periods, changes the feature snapshot and uses updated Transformers/PEFT versions. Its evidence can overturn confidence in the original ranking, but cannot identify a batch-size defect as the sole causal explanation. A controlled rerun on the original snapshot and dates would be required for that claim.')

    page();h('6  Four-year results')
    fig(P['ladder'],'Round B cumulative ladder. Dotted curves use the original checkpoint; solid curves average three LoRA seeds. Each year scores the same 250 dates across rungs. The all-external arm is shown separately in Table 5 because its training recipe requires a matched control.')
    add('Across the seven standard rungs, pooled fine-tuned-minus-pretrained WQL ranges from -0.001813 to +0.000371. Every reported pooled interval includes zero. This is insufficient evidence of a reliable adaptation benefit at these settings, rather than an equivalence test proving an exactly zero effect. Small changes in individual years are exploratory among many examined contrasts.')
    rows=[['Model and recipe','2022','2023','2024','2025','Mean']]
    for group,key,label in [('new_pt','r7_all_external-bw4','Pretrained, all external'),('new_ft','r7_all_external-bw4','LoRA, all external, 4 x 1,000'),('new_ft','r3_qqq_calendar_market-bw4','LoRA, control, 4 x 1,000'),('new_ft','r3_qqq_calendar_market','LoRA, control, 8 x 500')]:
        v=E[group][key];rows.append([label]+[f'{v[y]:.4f}' for y in ['2022','2023','2024','2025','mean']])
    tab(rows,[66,20,20,20,20,20],'Matched all-external comparison. The four-window and eight-window recipes each sample 4,000 training windows, but differ in optimizer-update granularity.',8.6)
    a=E['contrasts']['all_vs_control'];g=E['contrasts']['all_ft_gain']
    add(f"All-external improves by {g['mean']:.6f} WQL relative to its own pretrained arm. Yet it remains {a['mean']:+.6f} above the identically trained control, with a five-window-block 95% interval {ci(a['ci_block5'])}. Its advantage from adaptation also remains unresolved under that block analysis: {ci(g['ci_block5'])}. Thus the larger input set has not established incremental predictive value over the market control.")
    add('The first-round mid-2025 to mid-2026 interval favored the pretrained all-external setup. Most additional calendar-year comparisons show the opposite direction. Period dependence and a small initial training budget make the original improvement a development observation, not a general result. The later cumulative GDELT arm combines Fed and recession with other sources; it does not independently replicate all six single-topic fits.')

    page();h('7  Input perturbation and predictive reliance')
    add('The swap test replaces one feature\'s context with the same feature from another evaluation window and re-forecasts without retraining. For known-future calendar inputs it also replaces the future calendar block. Three donor assignments are averaged. The loss difference is scrambled minus original: a positive value indicates that the original alignment was helpful under this intervention; a negative value indicates that the swap improved the score.')
    add('This is an offline diagnostic. Donor windows can be later than the recipient origin, so the perturbed series is not a feasible historical trading input. Its history retains temporal structure but loses its original relationship to the target and other covariates. It is not literal feature removal, a causal intervention on the market, or a direct measure of internal semantic understanding.')
    fig(P['perm'],'Selected pretrained control features in the original evaluation interval. Means and exploratory 95% window-bootstrap intervals use locally normalized window WQL, as in the saved swap experiment. These values are not directly interchangeable with the fold-normalized ladder deltas.')
    add('Swapping VXN change lowers mean locally normalized WQL by 0.0134 in the control. Its median effect is much smaller (-0.00234), and five windows account for approximately 94% of the signed total. In the 42-feature setup, the mean effect is -0.0138. A same-session VXN variant has a similar effect, which weakens the hypothesis that a one-session lag alone explains the result. The perturbation may alter forecast dispersion ahead of shocks; the evidence does not establish that the model reads volatility backwards.')
    sub('7.1  News effects and adaptation')
    add('The 42-feature pretrained test has one small positive news result: GDELT Fed average tone, +0.00175 WQL, with an unadjusted interval excluding zero. Forty-two feature tests create a multiple-comparison problem. This is also a different variable from the FOMC-statement classifier used in another ladder arm, so those two findings do not independently confirm the same signal. Correlated features can share information and make individual perturbations appear weak.')
    add('Six fine-tuned checkpoints cover the control, FOMC rung and all-external rung in 2022 and 2025, using seed 42. Their matched pretrained/LoRA importance-vector correlations range from 0.908 to 0.978. Dominant effects persist, but profiles are not identical: the two-year yield change moves toward zero in both tested 2022 control/FOMC cells. The supported conclusion is limited change in the measured importance pattern at this budget, rather than proof that training cannot alter covariate use.')

    page();h('8  Calibration, regimes and baseline performance')
    fig(P['cal'],'Left: average nominal 80% coverage for the seven standard rungs, with equal rung weight. Right: post-hoc mean adaptation effect across rungs 3-6 after averaging their repeated measurements per observed week. The terciles contain 67, 66 and 67 distinct weeks, not 800 independent market observations.')
    add('Across all 108 trained cells, including the matched-recipe reruns, mean 80% coverage increases from 77.69% to 80.09%, and mean interval width increases by 7.80%. Those cells repeatedly evaluate the same market dates, so this aggregate describes model behavior rather than providing an independent-observation count. The standard-rung plot shows a rise in every year, but higher coverage is not always closer to 80%: in 2025 it moves from approximately 80.3% to 81.8%. Tail coverage also remains imperfect.')
    add('Widening is consistent with the post-hoc loss pattern. After averaging seeds and rungs 3-6 within each observed week, fine-tuning raises WQL by 0.0090 in the quiet third and lowers it by 0.0103 in the volatile third. The group assignment uses realized future absolute returns, so this analysis explains realized performance; it cannot select a trading action in advance. We do not attach independence-based intervals to repeated rung observations.')
    add('These findings support a dispersion-change hypothesis, not a complete identification of an unconditional scale mechanism. A forecast-time regime model or a simple calibrated widening baseline, fitted only on training data, would be needed to test whether LoRA adds more than interval rescaling. In particular, improvement in one realized-volatility group and deterioration in another can occur without revealing what information the network used.')
    sub('8.1  Comparison with a simple forecasting solution')
    c=E['contrasts']['control_vs_gaussian']
    tab([['Contrast','Mean WQL difference','Five-window-block 95% interval'],['All-external LoRA minus matched control',f"{E['contrasts']['all_vs_control']['mean']:+.6f}",ci(E['contrasts']['all_vs_control']['ci_block5'])],['All-external LoRA minus own pretrained',f"{E['contrasts']['all_ft_gain']['mean']:+.6f}",ci(E['contrasts']['all_ft_gain']['ci_block5'])],['Control LoRA minus Gaussian',f"{c['mean']:+.6f}",ci(c['ci_block5'])]],[68,39,59],'New report sensitivity analysis: seeds averaged, equal year weights, circular blocks within years, 5,000 draws. Negative differences favor the first model. Intervals are exploratory and not multiplicity-adjusted.',8.6)
    add('The fine-tuned market control beats the Gaussian on the pooled four-year score, while its 2022 point estimate is worse. Therefore, absence of demonstrated incremental news benefit does not mean that the complete model has no forecasting skill. This baseline is deliberately simple; no supervised quantile regression, tree model, or learned volatility baseline was executed in this study. Comparisons with those solutions remain future work.')

    page();h('9  Discussion, limitations and future work')
    sub('9.1  What the experiments answer')
    add('The initial screen suggested small source-specific improvements. The broader study does not establish an incremental benefit from the tested news groups across four calendar years under a common recipe. All-external adaptation reduces its own pretrained loss but does not surpass a matched market control. Permutation tests show broadly persistent input-importance patterns, and interval widening is a clearer measured effect than an improvement in overall WQL. These findings answer RQ1 and RQ2 with limited positive evidence and RQ3 with a qualified observation of similarity.')
    add('The relevant claim is about this representation, target, evaluation protocol and training budget. Numeric topic averages may discard event timing, novelty or firm-specific relevance that richer text representations retain. Shared topics may be redundant with market features. The model may require a different adaptation procedure. None of those explanations has been isolated, and a nonsignificant contrast is not proof of no possible news information.')
    sub('9.2  Limits on generalization and causal interpretation')
    add('The study uses one ETF, a five-day direct forecast, limited training seeds and a fixed learning rate/rank. Calendar-year folds provide distinct evaluation outcomes, but still form one market history. The first-round evaluation overlaps part of the later development history. All examined periods informed model choices, so none is an untouched final test. The October 2025 publication date of Chronos-2 follows most historical fold dates; the LoRA split does not rule out overlap with foundation-model pretraining or establish a historically deployable backtest.')
    add('Point-in-time feature alignment limits look-ahead in the data pipeline but cannot recover unavailable historical vintages. The FOMC classifier is retrospective, SEC company selection reflects current leaders, and some series may contain revisions. Missing-news intervals remain masked. The second round also changes the feature snapshot and some library versions, which prevents attributing the change from the first round to the batch correction alone.')
    add('Uncertainty intervals depend on normalization, pairing and resampling. We distinguish fold-normalized ladder deltas from locally normalized permutation deltas, average seed replicas before resampling, and show a longer-block sensitivity check for headline comparisons. Unadjusted exploration over many rungs, years, features and conditions can produce chance discoveries. The positive Fed-tone permutation result and single-year adaptation gains therefore require replication.')
    sub('9.3  Next experiments')
    add('First, repeat all six individual topics with explicit window batches on the same snapshot and evaluation dates to separate recipe changes from period changes. Second, align training and evaluation to a one-day horizon with daily origins if the operational goal is tomorrow\'s return. Compare share-only, tone-only and a small topic combination against matched controls. Third, compare LoRA with supervised quantile and calibrated-scale baselines, using an inner chronological validation split to select a limited budget of training settings. Finally, freeze the selected specification and record forecasts before outcomes arrive in a new prospective block. Event-conditioned calibration should use conditions available at the forecast origin.')
    sub('9.4  Conclusion')
    add('Our completed experiments do not demonstrate reliable incremental news value for QQQ under the tested Chronos-2 LoRA recipe. They do demonstrate why an apparent source ranking needs replication, why training budgets must count actual forecast windows, and why aggregate loss should be read alongside calibration and perturbation diagnostics. These findings support a bounded empirical conclusion; they do not establish that foundation models cannot use financial news.')

    page();h('References')
    refs=[
      ('Ansari, A. F., et al. (2025). <i>Chronos-2: From Univariate to Universal Forecasting.</i> arXiv:2510.15821.','https://arxiv.org/abs/2510.15821'),
      ('Hu, E. J., et al. (2021). <i>LoRA: Low-Rank Adaptation of Large Language Models.</i> arXiv:2106.09685.','https://arxiv.org/abs/2106.09685'),
      ('Rahimikia, E., Ni, H., and Wang, W. (2025). <i>Re(Visiting) Time Series Foundation Models in Finance.</i> arXiv:2511.18578.','https://arxiv.org/abs/2511.18578'),
      ('Shah, A., Paturi, S., and Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task &amp; Market Analysis. <i>ACL</i>, 6664-6679.','https://aclanthology.org/2023.acl-long.368/'),
      ('Gneiting, T., and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. <i>JASA</i>, 102(477), 359-378.','https://doi.org/10.1198/016214506000001437'),
      ('GDELT Project (2017). <i>GDELT DOC 2.0 API Debuts.</i> API documentation.','https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/'),
      ('Baker, S. R., Bloom, N., and Davis, S. J. (2016). Measuring Economic Policy Uncertainty. <i>QJE</i>, 131(4), 1593-1636.','https://www.policyuncertainty.com/research.html'),
      ('Amazon Science. <i>Chronos Forecasting</i>, version 2.3.2, source implementation.','https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2')]
    for i,(t,url) in enumerate(refs,1):add(f'<a name="ref{i}"/>[{i}] {t} <link href="{url}" color="{r.BLUE}">Source</link>.','reference')
    h('Appendix A  Execution record and reproducibility')
    tab([['Round','Slurm jobs','Completed work'],['Infrastructure','869927','Five LoRA updates, 15 outcomes, adapter reload'],['A: source screen','869989; 871482; 874192','18 fits, 200 updates per fit'],['B: ladder','896510; 896540; 896541','84 fits, seven rungs x four years x three seeds'],['B: matched recipe','896751; 896752','24 fits, all-external and control, 4 x 1,000'],['B: diagnostics','896501; 896762','32 pretrained cells; six paired checkpoint perturbation tests']],[34,49,83],'The 126 substantive LoRA fits comprise 18 first-round fits plus 108 later fits. Additional pretrained exploration and perturbation inference ran on MPS; these are not additional training seeds.',8.3)
    add('Both rounds pin model revision <font name="PaperMono" size="7.7">29ec3766d36d6f73f0696f85560a422f50e8498c</font>. Round A data SHA-256 is <font name="PaperMono" size="7.2">e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037</font>; Round B is <font name="PaperMono" size="7.2">b3beb046933b15f1f75c098bfa6df30af4106395302ea4d8b68711322a262e34</font>. Both GPU runtimes record Chronos 2.3.2 and PyTorch 2.11.0+cu128. Transformers/PEFT change from 5.16.1/0.20.0 to 5.17.0/0.21.0. Later metadata records dirty working trees, making source hashes material to reproduction.','note')
    add('Round A predictions and adapters reside in <font name="PaperMono" size="8">models/news-ablation-869989/</font>. Round B local exports in <font name="PaperMono" size="8">docs/results/walk_forward/</font> include per-window loss summaries and 140 cell metadata/metric pairs (108 trained, 32 pretrained-only). Full later adapters and per-day trained predictions are not in that local export. The final report analyzer checks cell metrics against window means to 3.4e-16 and records source hashes in <font name="PaperMono" size="8">final report/evidence.json</font>. No new model was trained to prepare this report.','note')

    page();h('Appendix B  Exact inputs and source definitions')
    cfg=json.loads((ROOT/'configs/news_ablation.json').read_text());c=cfg['control_features']
    rows=[['Control family','Exact fields']]
    for label,cols in [('QQQ (6)',c[:6]),('Market (5)',c[6:11]),('Rates / macro (4)',c[11:15]),('Calendar (6)',c[15:])]:rows.append([label,', '.join(cols)])
    tab(rows,[32,134],'The control contains 15 past-only and six known-future fields. Every experiment also includes the target history.',8.5)
    tree=ast.parse((ROOT/'scripts/fetch_gdelt_news.py').read_text())
    topics=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOPICS' for t in n.targets))
    rows=[['GDELT topic','Query before the English-language filter']]
    for k,v in topics.items():rows.append([k.replace('_',' '),escape(v)])
    tab(rows,[34,132],'Each topic adds gdelt_TOPIC_news_share and gdelt_TOPIC_avg_tone. Queries use sourcelang:eng and no timeline smoothing. Article counts contribute to share construction but are not selected model inputs.',8.5)
    rows=[['Other external family','Exact fields']]
    for k,label in [('uncertainty','EPU / EMU (2)'),('fomc_tone','FOMC (4)'),('sec_disclosures','SEC (3)')]:rows.append([label,', '.join(cfg['setups'][k]['add_features'])])
    tab(rows,[34,132],'All external combines nine non-GDELT fields and 12 GDELT fields with the 21 controls.',8.5)
    add(f'EPU and EMU derive from the policy uncertainty data family {ref(7)}. The FOMC model used in the pipeline is <link href="https://huggingface.co/LorenzoAleCon29/roberta-large-fomc-hawkish-dovish" color="{r.BLUE}">LorenzoAleCon29/roberta-large-fomc-hawkish-dovish</link>, revision f4759d4ad3f1182f81d87e47ba603261740d36cf. Missing observations use the model\'s observation mask. The array-based Chronos interface preserves exchange-session order without fabricating holiday rows {ref(8)}.','note')

def main():
    P=charts();manuscript(P)
    doc=BaseDocTemplate(str(r.OUT),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,topMargin=23*mm,bottomMargin=23*mm,title='Can News Improve QQQ Forecasts?',author='Elad Kulman and Tom Weitman')
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='paper',frames=frame,onPage=r.footer));doc.build(r.STORY)
    (ROOT/'final report/paper_text.md').write_text('\n\n'.join(TEXT))
    print(r.OUT,'figures',r.FIGURE,'tables',r.TABLE)

if __name__=='__main__':main()
