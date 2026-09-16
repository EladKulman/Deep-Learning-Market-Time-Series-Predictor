import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const RUNTIME='/Users/eladkulman/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
const SKILL='/Users/eladkulman/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const TEMPLATE='/Users/eladkulman/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const req=createRequire(path.join(RUNTIME,'node/node_modules/anchor.cjs'));
const {PresentationFile,FileBlob}=await import(pathToFileURL(req.resolve('@oai/artifact-tool')));
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')));
const TMP=path.join(ROOT,'tmp/final_report_deck');
await fs.mkdir(TMP,{recursive:true});
const E=JSON.parse(await fs.readFile(path.join(ROOT,'final report/evidence.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(TEMPLATE));
const originals=[...p.slides.items];
const coverSource=originals[0], contentSource=originals[3];
const FONT='Helvetica Neue',INK='#000000',BLUE='#3D8DFF',GRAY='#7B8188',TEAL='#168B79',GOLD='#BF8717';
const slides=[],timing=[],speakerText=[];

function tx(sl,text,x,y,w,h,size=30,color=INK,bold=false){
 const sh=sl.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=text;sh.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0},verticalAlignment:'top'};
 return sh;
}
async function slide(title,seconds,presenter,spoken,sources=[],cover=false){
 const s=(cover?coverSource:contentSource).duplicate();s.moveTo(p.slides.items.length-1);
 const layout=JSON.parse(await (await s.export({format:'layout'})).text());
 if(!cover){
  for(const e of layout.elements){
   if(e.kind!=='shape')continue;const sh=p.resolve(e.aid),[x,y,w,h]=e.bbox;
   if(y<150){sh.text=title;sh.text.style={typeface:FONT,fontSize:38.67,color:INK,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};}
   else if(y>640){sh.text=x>1000?String(slides.length+1):'';}
   else sh.delete();
  }
 }
 slides.push(s);timing.push({slide:slides.length,title,seconds,presenter,backup:seconds===0});
 const notes=`${seconds?`Main presentation. ${presenter}. ${seconds} seconds.`:'Backup slide. Use only for questions.'}\n\n${spoken}\n\nSources:\n${sources.join('\n')}`;
 s.speakerNotes.textFrame.setText(notes);
 speakerText.push(`## ${slides.length}. ${title}\n\n${seconds?`${presenter} - ${seconds} seconds`:'Backup'}\n\n${spoken}\n\nSources: ${sources.join('; ')}`);
 return s;
}
function table(s,values,{x=41.33,y=195,w=1197.33,h=350,widths=null,size=26}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,...(widths?{columnWidths:widths}:{})});
 t.borders.assign({fill:'#D1D5DA',width:.7,style:'solid'});
 for(let i=0;i<values.length;i++)for(let j=0;j<values[i].length;j++){
  const c=t.getCell(i,j);c.fill=i===0?'#F2F3F4':'#FFFFFF';c.text.style={typeface:FONT,fontSize:size,color:INK,bold:i===0,autoFit:'none'};
 }
 return t;
}
function chart(s,type,categories,series,{x=41.33,y=175,w=820,h=420,min,max,fmt='0.000',labels=false,horizontal=false,legend=true}={}){
 const c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,
  series:series.map((z,i)=>({name:z.name,values:z.values.map(v=>Number(v.toPrecision(12))),fill:z.color||[GRAY,BLUE,TEAL,GOLD][i],line:{fill:z.color||[GRAY,BLUE,TEAL,GOLD][i],width:2.5,style:z.dotted?'dotted':'solid'},marker:{symbol:'circle',size:6},valuesFormatCode:fmt})),
  hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{typeface:FONT,fontSize:21}},
  barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:90},
  lineOptions:{smooth:false},
  xAxis:{textStyle:{typeface:FONT,fontSize:21},majorGridlines:null,tickLabelPosition:'low'},
  yAxis:{min,max,numberFormatCode:fmt,tickLabelPosition:'low',textStyle:{typeface:FONT,fontSize:21},majorGridlines:{fill:'#E1E4E8',width:.7}},
  dataLabels:{showValue:labels,position:'outEnd',textStyle:{typeface:FONT,fontSize:20}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
 applyPresentationChartFont(c,{fontFamily:FONT});return c;
}
const SRC='docs/results/walk_forward/';
const primary=['https://arxiv.org/abs/2510.15821','https://arxiv.org/abs/2106.09685'];

let s=await slide('Can news improve QQQ forecasts?',45,'Elad',
 'Our project asks whether broad news narratives add information about QQQ beyond market prices and scheduled events. QQQ tracks the Nasdaq-100, so our unit of prediction is an index-level daily return, rather than the reaction of a single company to one headline. We chose a pretrained time-series model, Chronos-2, and compared the same model with different inputs, before and after adaptation. The first experiments suggested small gains. The larger study asks whether those gains persist across years and what the model changes during training. Our conclusion is deliberately scoped to the features and training budget that we tested.',
 [...primary,'docs/REPORT_DECISIONS.md'],true);
let l=JSON.parse(await (await s.export({format:'layout'})).text());
for(const e of l.elements){if(e.kind!=='shape')continue;const sh=p.resolve(e.aid);if(e.bbox[1]<120){sh.text='Workshop on Deep Learning';sh.text.style={typeface:FONT,fontSize:26,color:INK};}else if(e.bbox[1]<450){sh.text='Can news improve\nQQQ forecasts?';sh.text.style={typeface:FONT,fontSize:80,color:INK,autoFit:'none'};}else {sh.text='Elad Kulman and Tom Weitman\nGroup 13, Tel Aviv University\nSeptember 2026';sh.text.style={typeface:FONT,fontSize:30,color:INK,autoFit:'none'};}}

s=await slide('Daily returns and forecast uncertainty',75,'Elad',
 'A time-series forecaster uses an ordered history to predict future values. We supply up to 512 trading sessions and ask for five separate daily log returns at once. This is not one cumulative weekly return. After five sessions, we insert all five actual observations and issue the next forecast. The scored blocks do not overlap. Each day has 21 predicted quantiles, including a median and the 10th and 90th percentiles. The central band should contain about 80 percent of outcomes over repeated forecasts. This real example was issued after June 27, 2025. All five forecasts share that cutoff. We evaluate the quantiles with weighted quantile loss, or WQL. Lower is better. WQL measures a forecast distribution, so a value of 0.69 is not 69 percent accuracy. We separately report direction and coverage. Prior financial evaluations motivate this distinction between general forecasting capability and reliable equity-return prediction.',
 ['models/news-ablation-869989/gdelt_fed-seed42/validation_predictions.csv','scripts/compare_chronos2_validation.py','https://arxiv.org/abs/2511.18578']);
chart(s,'line',['Jun 30','Jul 1','Jul 2','Jul 3','Jul 7'],[
 {name:'Actual',values:[.6456,-.8465,.6941,.9792,-.7562],color:INK},
 {name:'Median',values:[.1548,.1311,.1701,.1693,.1779],color:BLUE},
 {name:'10th percentile',values:[-1.0729,-1.1233,-1.1152,-1.1137,-1.1259],color:GRAY,dotted:true},
 {name:'90th percentile',values:[1.3081,1.3258,1.3636,1.3600,1.3910],color:GRAY,dotted:true}],{w:780,h:400,fmt:'0.0',min:-1.5,max:1.5});
tx(s,'21 quantiles per day\n\nWQL scores the distribution\n\nLower WQL is better',885,195,350,350,30);
tx(s,'Daily log return (%)  •  Forecast origin: June 27, 2025'.replace('  •  ','; '),41,610,1130,35,23,GRAY);

s=await slide('Chronos-2',60,'Tom',
 'Chronos-2 is a pretrained time-series foundation model. The checkpoint has about 119.5 million base parameters, conventionally rounded to 120 million. We did not pretrain that model ourselves. It takes numeric time series, not the raw headlines. Each covariate becomes another series in a group with the QQQ target. Time attention processes history and group attention exchanges information across those series. The model scales observed context, forms patches of 16 observations, and produces quantile forecasts. Our context is capped at 512 sessions. Known future calendar values can be supplied, while future news and prices remain masked. The published model learns covariate relationships in pretraining. Whether those relationships transfer to financial news is our empirical question, not an assumption that we can settle from the architecture.',
 [primary[0],'scripts/build_project_report.py','configs/walk_forward.json']);
tx(s,'119.5M base parameters\n\n512-session context\n\n21 forecast quantiles',41,213,580,370,38);
tx(s,'Numeric target and covariates\n\nAttention across time and series\n\nFuture news stays masked',658,213,580,370,32);

s=await slide('LoRA adaptation and training windows',60,'Tom',
 'LoRA learns small low-rank weight updates while freezing the base checkpoint. Here that means roughly 1.2 million trainable adapter parameters, rank eight, and a learning rate of ten to the minus five. Every experiment starts from the same original weights. During training, the model sees sampled historical windows and their future return targets. Test observations never enter the fitting call. There is an important batching detail: Chronos counts target and covariate series, not independent forecast windows. Our first batch argument of eight yielded effectively one complete window per update. The later runner explicitly requests eight windows, which is 176 series rows for the 21-feature control. The larger study uses 4,000 sampled windows per fit. We freeze the adapter during evaluation. Because other settings and periods also changed, we cannot attribute every difference between rounds to batching alone.',
 [primary[1],'scripts/compare_chronos2_validation.py','configs/walk_forward.json']);
table(s,[['Recipe','Windows per update','Updates'],['Initial source screen','About 1','200'],['Later standard rungs','8','500'],['All external + matched control','4','1,000']],{h:270,widths:[590,350,257],size:29});
tx(s,'1.2M trainable parameters  /  rank 8\nSame original checkpoint for every fit',41,505,1197,95,32);

s=await slide('Public data and the availability rule',60,'Elad',
 'The archived data inventory spans about twenty years and contains 77 columns, including the date, target and auxiliary raw fields. We select 21 control covariates or 42 in the full external setup. The sources include QQQ and related markets, rates, release calendars, uncertainty indices, FOMC statements, SEC activity, and six GDELT topics. GDELT provides news share and average tone. It limits the comparable news-training history to 2017 onward. The central data rule is availability, not the period a number describes. A July CPI observation cannot be used on July first if it is published in August. We align CPI vintages to release dates and apply source-specific delays to other inputs. Missing news stays masked. These rules reduce look-ahead, although revised series and incomplete historical vintages remain limitations.',
 ['docs/DATA_NOTES.md','configs/news_ablation.json','https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/']);
tx(s,'Markets and macroeconomics\nScheduled events\nEPU / EMU uncertainty\nFOMC statements and SEC\nSix GDELT topic pairs',41,213,580,355,32);
tx(s,'CPI example\n\nJuly observation\nAugust publication\n\nUsable only after release',658,213,580,355,32);
tx(s,'77-column inventory. News comparisons begin in 2017.',41,610,1160,35,24,GRAY);

s=await slide('Two completed experiment rounds',60,'Elad',
 'The first round adds one source or one topic to a fixed control and evaluates 250 days from mid-2025 to mid-2026. It contains eighteen fits, with seed repetitions only for selected configurations. The later round expands to four calendar years, 2022 through 2025. Each year trains on earlier data beginning in 2017, then scores fifty five-day windows. The ladder starts with the target and progressively adds QQQ transforms, calendars, markets, uncertainty, statement tone and news. Three seeds cover every trained configuration. There are 108 fits including the matched four-window controls. These are 1,000 distinct test days, not three times as many observations because there are three seeds. We compare original and adapted weights on identical dates. A separate swap test exchanges one feature history between evaluation windows and measures the resulting change in loss.',
 ['configs/news_ablation.json','configs/walk_forward.json','scripts/permutation_importance.py']);
table(s,[['','Initial screen','Four-year study'],['Test period','Jun 2025-Jun 2026','2022, 2023, 2024, 2025'],['Scored daily returns','250','1,000'],['LoRA fits','18','108'],['Feature design','One source added to control','Cumulative groups'],['Repetitions','Selected configurations','Three seeds throughout']],{h:375,widths:[370,390,437],size:27});
tx(s,'Same dates within each comparison. Five-day horizon throughout.',41,610,1197,35,24,GRAY);

s=await slide('The initial source gains were small',30,'Elad',
 'The first screen identified candidates rather than a reliable winner. Fed news improved WQL by 1.28 percent against the fine-tuned control in seed 42. Across three seeds, all-external improved the mean by 0.60 percent, but its uncertainty interval included no gain. EPU and EMU lost their first-seed advantage on repetition. The larger study was designed to test whether those small effects carried across years with a better specified training budget.',
 ['docs/GDELT_TOPIC_RESULTS.md','docs/NEWS_ABLATION_RESULTS.md']);
table(s,[['Round A finding','Observed change','Qualification'],['Fed topic, seed 42','1.28% lower WQL','One training seed'],['All external, three seeds','0.60% lower mean WQL','Interval includes zero'],['Fed direction','144 / 250 correct','Always-up: 143 / 250']],{h:285,widths:[420,370,407],size:29});
tx(s,'Candidate improvements motivated the larger evaluation',41,535,1160,70,34);

const rn=['r0_target','r1_qqq','r2_qqq_calendar','r3_qqq_calendar_market','r4_plus_uncertainty','r5_plus_fed','r6_plus_gdelt_fed_recession'];
s=await slide('Overall adaptation gains remain unresolved',45,'Tom',
 'This chart uses the cumulative ladder from the four-year study. Each point averages the four calendar-year WQL scores, and the trained points also average three seeds. Across the seven standard rungs, the gain from LoRA is small. Every pooled confidence interval for adapted minus pretrained loss includes zero. That does not prove the effects are exactly zero. It means this experiment does not establish a reliable aggregate advantage. We keep the 42-feature model separate on the next slide because it requires a different batch recipe and an identically trained control.',
 [SRC+'wql_pretrained.csv',SRC+'wql_fine_tuned.csv',SRC+'finetune_gain.csv']);
chart(s,'line',['Target','QQQ','Calendar','Market','EPU','FOMC','GDELT'],[{name:'Pretrained',values:rn.map(n=>E.new_pt[n].mean),color:GRAY,dotted:true},{name:'LoRA',values:rn.map(n=>E.new_ft[n].mean),color:BLUE}],{w:850,h:420,min:.694,max:.702,fmt:'0.000'});
tx(s,'Four calendar years\nThree seeds\n\nEvery pooled adaptation interval includes zero',940,210,290,345,30);
tx(s,'WQL, lower is better. Cumulative feature groups. Axis shows a narrow range.',41,610,1190,35,23,GRAY);

s=await slide('All external does not outperform matched control',35,'Tom',
 'The full 42-feature model uses four windows per step and a thousand steps. We trained a control with exactly that recipe. All-external improves from about 0.704 to 0.699 with adaptation, but the matched control is about 0.698. The difference is positive 0.00126, and the longer-block confidence interval spans zero. The result is no demonstrated extra value from the external features. It is not evidence that training has no effect at all: training improves the full model relative to its own starting point.',
 [SRC+'wql_pretrained.csv',SRC+'wql_fine_tuned.csv','final report/evidence.json']);
chart(s,'line',['2022','2023','2024','2025'],[
 {name:'All external, pretrained',values:['2022','2023','2024','2025'].map(y=>E.new_pt['r7_all_external-bw4'][y]),color:GRAY,dotted:true},
 {name:'All external, LoRA',values:['2022','2023','2024','2025'].map(y=>E.new_ft['r7_all_external-bw4'][y]),color:BLUE},
 {name:'Matched control, LoRA',values:['2022','2023','2024','2025'].map(y=>E.new_ft['r3_qqq_calendar_market-bw4'][y]),color:TEAL}],{w:830,min:.68,max:.725,fmt:'0.000'});
tx(s,'+0.00126',920,210,315,75,54,BLUE);
tx(s,'All external minus control\n\n95% interval\n[-0.00095, +0.00369]',920,305,315,250,29);
tx(s,'WQL. Matched recipe. Circular blocks of five forecast windows within each year.',41,610,1190,35,23,GRAY);

const pair=E.pairs.find(x=>x.cell.startsWith('r7_all_external-fold2025'));
const fspec=[['vxn_log_chg','VXN\nchange'],['overnight_gap','Overnight\ngap'],['tlt_log_ret','TLT\nreturn'],['epu_log','EPU'],['fomc_net_hawkish_ewma','FOMC\ntone'],['gdelt_fed_avg_tone','GDELT\nFed tone']];
s=await slide('Input-importance patterns change relatively little',35,'Tom',
 'These are paired swap effects for the all-external model in calendar 2025. A negative bar means that exchanging the feature history improved the score. The major patterns persist after adaptation. Across all six checkpoint comparisons, the importance-profile correlation lies between 0.91 and 0.98. Most news effects are small. We should not call the profiles identical, or conclude that the network literally reads an input backwards. This is a prediction perturbation test, and correlated features and unusual donor histories limit its interpretation.',
 ['docs/results/permutation_finetuned/runs/','scripts/permutation_importance.py','final report/evidence.json']);
chart(s,'bar',fspec.map(x=>x[1]),[{name:'Pretrained',values:fspec.map(x=>pair.pre[x[0]]*1000),color:GRAY},{name:'LoRA',values:fspec.map(x=>pair.ft[x[0]]*1000),color:BLUE}],{w:850,h:420,min:-12,max:3,fmt:'0.0'});
tx(s,'0.91-0.98',940,220,290,75,52,BLUE);
tx(s,'Profile correlations\nacross six paired tests\n\nSmall changes remain',940,315,290,250,29);
tx(s,'Scrambled minus original WQL x 1,000. Mean effects, seed 42. Local window normalization.',41,610,1190,35,22,GRAY);

s=await slide('Fine-tuning widens the forecast intervals',35,'Elad',
 'The clearest change is forecast dispersion. Across all 108 trained cells, intervals widen by about 7.8 percent and aggregate 80 percent coverage rises from 77.7 to 80.1 percent. The chart shows the seven standard rungs by year. Coverage increases in every year, although increasing coverage does not always improve calibration: 2025 was already near the target. Post-hoc analysis finds that adaptation helps in volatile weeks and hurts in quiet weeks. That supports a widening explanation, but it does not prove that widening is the only thing learned.',
 [SRC+'coverage_direction.csv','final report/evidence.json','docs/FINETUNED_LADDER_RESULTS.md']);
const cvs=E.coverage_by_year.filter(x=>/^r[0-6]_/.test(x.rung)&&!x.rung.endsWith('-bw4'));
function cm(y,m){let a=cvs.filter(x=>x.fold===y&&x.model===m);return a.reduce((v,x)=>v+x.coverage_10_90,0)/a.length;}
chart(s,'line',['2022','2023','2024','2025'],[{name:'Pretrained',values:[2022,2023,2024,2025].map(y=>cm(y,'base_pretrained')),color:GRAY},{name:'LoRA',values:[2022,2023,2024,2025].map(y=>cm(y,'fine_tuned')),color:BLUE},{name:'80% target',values:[.8,.8,.8,.8],color:GOLD,dotted:true}],{w:850,min:.72,max:.84,fmt:'0%'});
tx(s,'+7.8%',940,215,290,85,60,BLUE);
tx(s,'Mean interval width\n\n77.7% to 80.1%\nAggregate coverage',940,325,290,240,30);
tx(s,'Chart: seven standard rungs. Callouts: all 108 trained cells, including matched controls.',41,610,1190,35,22,GRAY);

s=await slide('Conclusions and the next experiments',60,'Elad',
 'Across four years, the tested news groups have not established reliable incremental value beyond the market control. At this LoRA budget, forecast intervals widen while the main input-importance patterns remain broadly similar. The lesson is to measure training windows explicitly and to evaluate small source gains across periods, seeds, and uncertainty analyses. This is a bounded result about one ETF, numeric news summaries, and this recipe. It is not proof that news contains no information or that all foundation models fail. The next experiments are a controlled rerun of individual topics, daily one-step forecasts, and a calibrated-scale or supervised quantile baseline. We should select settings on development data and then freeze the model for an untouched prospective test. The model also beats our simple Gaussian baseline on pooled loss, so lack of a demonstrated news benefit should not be confused with lack of all predictive skill.',
 ['final report/evidence.json','docs/results/independent_review_2026-09-15.md']);
tx(s,'No reliable incremental news benefit\nunder the tested recipe\n\nWider intervals after adaptation\nwith broadly similar input effects',41,213,580,380,32);
tx(s,'Next experiments\n\nDaily one-step forecasts\nMatched topic reruns\nStronger quantile / scale baselines\nUntouched prospective test',658,213,580,380,30);

// Backup slides are outside the ten-minute script.
s=await slide('Backup: the original topic comparison',0,'',
 'All values are from the original five-day, seed-42 screen. The all-external model includes sources beyond GDELT. The later cumulative ladder does not replace a matched rerun of each individual topic. Direction must be compared with the 143/250 always-up rate.',
 ['docs/GDELT_TOPIC_RESULTS.md']);
const topicNames=[['control','Control'],['gdelt_fed','Fed'],['gdelt_recession','Recession'],['gdelt_semiconductor','Semiconductors'],['gdelt_ai','AI'],['gdelt_inflation','Inflation'],['all_external','All external'],['gdelt_big_tech_earnings','Big-tech earnings']];
table(s,[['Added source','WQL','Direction correct'],...topicNames.map(([k,label])=>{let m=E.old[k+'-seed42'].fine_tuned;return[label,m.weighted_quantile_loss.toFixed(6),`${Math.round(m.directional_accuracy*250)} / 250`];})],{y:165,h:450,widths:[580,310,307],size:26});

s=await slide('Backup: batch correction and comparison limits',0,'',
 'For the 21-feature control, the newer effective batch has 8 times 22 equals 176 series rows. For 42 external features, the four-window recipe has 4 times 43 equals 172 rows. The old argument of eight resulted in effectively one full group per update. The later runs differ in more than batch size: training steps, test periods, feature snapshot, and Transformers/PEFT versions also change. Therefore the statement that the batch bug caused all original improvements is not isolated by these experiments.',
 ['scripts/compare_chronos2_validation.py','configs/walk_forward.json','docs/results/walk_forward/cells/']);
table(s,[['','Original','Later control','Later all external'],['Series rows / update','8 requested','176','172'],['Windows / update','About 1','8','4'],['Updates','200','500','1,000'],['Sampled windows','About 200','4,000','4,000']],{h:350,widths:[385,240,285,287],size:27});
tx(s,'The rounds also change dates, data snapshot and some library versions',41,590,1190,60,28);

s=await slide('Backup: gains depend on realized volatility',0,'',
 'Each observed week first averages seeds and rungs 3 through 6. We then divide the 200 distinct weeks into terciles using realized mean absolute daily returns. The quiet group has 67 weeks, the middle group 66, and the volatile group 67. Average WQL change is positive 0.0090, negative 0.0021, and negative 0.0103. The analysis is post-hoc and uses future realized movement, so it is not a forecast-time decision rule. It is consistent with wider intervals helping during volatile outcomes and hurting during quiet outcomes.',
 [SRC+'gain_by_window_volatility.csv','final report/evidence.json']);
chart(s,'bar',E.volatility.map(x=>x.group),[{name:'LoRA minus pretrained',values:E.volatility.map(x=>x.mean*1000),color:BLUE}],{w:820,min:-15,max:15,fmt:'0.0',labels:true,legend:false});
tx(s,'200 distinct weeks\n\n67 / 66 / 67 per group\n\nPost-hoc grouping',925,210,310,330,31);
tx(s,'WQL difference x 1,000. Negative favors LoRA. Repeated rungs averaged per week.',41,610,1190,35,23,GRAY);

s=await slide('Backup: evidence limits and Gaussian baseline',0,'',
 'The final paper recomputes three contrasts with five-window circular blocks drawn within each year and 5,000 bootstrap replicates. The fine-tuned market control beats the Gaussian pooled by 0.01088 WQL, with interval negative 0.02269 to negative 0.00106. Its 2022 point estimate is worse. The all-external versus matched control interval includes zero. All these intervals are exploratory and not multiplicity-adjusted. We have one ETF and no untouched prospective test. The base checkpoint publication follows most historical folds, so the adaptation split does not establish absence of pretraining overlap. The original VXN scramble effect is concentrated in five windows and cannot identify the model mechanism.',
 ['final report/evidence.json','docs/results/independent_review_2026-09-15.md',...primary,'https://arxiv.org/abs/2511.18578']);
tx(s,'Control LoRA minus Gaussian\n\n-0.01088 WQL\n95% interval [-0.02269, -0.00106]\n\nPooled advantage, with a worse\n2022 point estimate',41,213,580,390,30);
tx(s,'One ETF and four test years\n\nExploratory comparisons\n\nNo untouched prospective test\n\nPretraining overlap unresolved',658,213,580,390,30);

for(const o of originals)o.delete();
slides.forEach((x,i)=>x.moveTo(i));
if(timing.reduce((a,x)=>a+x.seconds,0)!==600)throw Error('Main talk is not 600 seconds');
await fs.writeFile(path.join(ROOT,'final report/presenter_notes.md'),'# Ten-minute presentation\n\nSlides 1-12 total 10:00. Slides 13-16 are optional backups. Times are rehearsal targets.\n\n'+speakerText.join('\n\n'));
await fs.writeFile(path.join(ROOT,'final report/presentation_timing.json'),JSON.stringify(timing,null,2));
const candidate=path.join(TMP,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<slides.length;i++){
 await fs.writeFile(path.join(TMP,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await (await slides[i].export({format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(TMP,`slide-${i+1}.layout.json`),await (await slides[i].export({format:'layout'})).text());
}
const finalName=process.env.DECK_FINAL_NAME||'qqq_chronos2_presentation.pptx';
const finalPath=path.join(ROOT,'final report',finalName);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,
 pythonExecutable:path.join(RUNTIME,'python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[4,6,7,13,14].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[4,6,7,13,14],requiredNativeChartOwnerSlides:[2,8,9,10,11,15],
 materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[FONT],referencePath:TEMPLATE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(TEMPLATE)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(TMP,finalName+'.validation.json')});
console.log(JSON.stringify(result));
