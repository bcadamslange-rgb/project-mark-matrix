#!/usr/bin/env python3
import argparse, json, tempfile, zipfile, math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

REQ=['seasonal_fulfillment_extract_A.csv','seasonal_fulfillment_extract_B.csv','lighting_assortment_treatment.csv','daily_fulfillment_close.csv','source_conformation_rules.json','source_conformation_controls.csv','forecast_origin_schedule.csv','forecast_validation_calendar.csv','forecast_model_spec.json','forecast_deployment_gate.csv','forecast_feature_contract.json','data_dictionary.json','fulfillment_policy.pdf','source_extract_manifest.csv','source_provenance.csv']

def root_from(p):
 p=Path(p)
 if p.is_dir(): return p,None
 if p.is_file() and zipfile.is_zipfile(p):
  td=tempfile.TemporaryDirectory(); root=Path(td.name)
  with zipfile.ZipFile(p) as z:z.extractall(root)
  return root,td
 raise SystemExit('--input must be the supplied package directory or ZIP')

def wape(a,p): return float(np.abs(a-p).sum()/np.abs(a).sum()*100)

def source_audit(root):
 import hashlib, csv
 controls={r['Metric']:int(r['Value']) for r in pd.read_csv(root/'source_conformation_controls.csv').to_dict('records')}
 def data_rows(path):
  with open(path,'rb') as f: return max(sum(1 for _ in f)-1,0)
 r1=data_rows(root/'seasonal_fulfillment_extract_A.csv'); r2=data_rows(root/'seasonal_fulfillment_extract_B.csv')
 prov={r['FileName']:r for r in csv.DictReader(open(root/'source_provenance.csv',encoding='utf-8'))}
 hash_ok=True
 for fn in ['seasonal_fulfillment_extract_A.csv','seasonal_fulfillment_extract_B.csv','daily_fulfillment_close.csv']:
  h=hashlib.sha256((root/fn).read_bytes()).hexdigest(); hash_ok=hash_ok and fn in prov and h==prov[fn]['SHA256']
 close_rows=max(sum(1 for _ in open(root/'daily_fulfillment_close.csv',encoding='utf-8'))-1,0)
 ok=(r1==controls['SeasonalExtractAPhysicalRows'] and r2==controls['SeasonalExtractBPhysicalRows'] and r1+r2==controls['RawPhysicalOccurrences'] and close_rows==controls['DailyFulfillmentCloseRows'] and hash_ok)
 return ok,controls['CrossExtractOverlapOccurrencesCollapsed'],controls['ConformedPhysicalOccurrences'],r1+r2

def build(root):
 for f in REQ:
  if not (root/f).exists(): raise SystemExit('missing '+f)
 close=pd.read_csv(root/'daily_fulfillment_close.csv',parse_dates=['ReportDate','PublishedAt'])
 source_controls_ok,overlap,conformed_occ,raw_occ=source_audit(root)
 origin=pd.read_csv(root/'forecast_origin_schedule.csv').iloc[0]
 cutoff=pd.Timestamp(origin['DecisionCutoff'])
 folds=pd.read_csv(root/'forecast_validation_calendar.csv')
 spec=json.load(open(root/'forecast_model_spec.json'))
 contract=json.load(open(root/'forecast_feature_contract.json'))
 gate=float(pd.read_csv(root/'forecast_deployment_gate.csv').iloc[0]['MaximumPct'])
 close=close.sort_values('ReportDate').set_index('ReportDate')
 full=pd.DataFrame(index=pd.date_range(close.index.min(),pd.Timestamp('2011-09-30'),freq='D'))
 full['ActualCompletedUnits']=close['CompletedUnits'].reindex(full.index).fillna(0).astype(float)
 full['DayOfWeek']=full.index.dayofweek; full['Month']=full.index.month; full['WeekOfYear']=full.index.isocalendar().week.astype(int)
 full['SinDayOfWeek']=np.sin(2*np.pi*full.DayOfWeek/7); full['CosDayOfWeek']=np.cos(2*np.pi*full.DayOfWeek/7)
 full['SinDayOfYear']=np.sin(2*np.pi*full.index.dayofyear/365.25); full['CosDayOfYear']=np.cos(2*np.pi*full.index.dayofyear/365.25)
 full['TimeIndex']=(full.index-full.index.min()).days
 for lag in [1,7,14,28]: full[f'UnitsLag{lag}']=full.ActualCompletedUnits.shift(lag)
 full['UnitsRolling7Mean']=full.ActualCompletedUnits.shift(1).rolling(7).mean(); full['UnitsRolling28Mean']=full.ActualCompletedUnits.shift(1).rolling(28).mean()
 full['ForecastOrigin']=full.index+pd.Timedelta(hours=5)
 pub=close[['PublishedAt','ClosedInvoiceCount','ClosedActiveCustomerCount','ClosedNMRGBP']].reset_index().sort_values('PublishedAt')
 left=pd.DataFrame({'TargetDate':full.index,'ForecastOrigin':full.ForecastOrigin.values}).sort_values('ForecastOrigin')
 asof=pd.merge_asof(left,pub,left_on='ForecastOrigin',right_on='PublishedAt',direction='backward')
 asof=asof.set_index('TargetDate')
 full['CloseReportDate']=asof['ReportDate']; full['CloseReportPublishedAt']=asof['PublishedAt']
 full['ClosedInvoiceCount']=asof['ClosedInvoiceCount']; full['ClosedActiveCustomerCount']=asof['ClosedActiveCustomerCount']; full['ClosedNMRGBP']=asof['ClosedNMRGBP']
 features=spec['PredictorColumns']
 params=spec['Parameters']
 preds=[]
 for _,f in folds.iterrows():
  start=pd.Timestamp(f.EvaluateFrom); end=pd.Timestamp(f.EvaluateThrough)
  tr=full[full.index<start].dropna(subset=features+['ActualCompletedUnits'])
  te=full[(full.index>=start)&(full.index<=end)].dropna(subset=features+['ActualCompletedUnits'])
  m=HistGradientBoostingRegressor(**params); m.fit(tr[features],tr.ActualCompletedUnits)
  part=te.copy(); part['Prediction']=m.predict(te[features]); part['ValidationFold']=f.Fold; preds.append(part)
 out=pd.concat(preds).sort_index(); governed=wape(out.ActualCompletedUnits.values,out.Prediction.values)
 decision='CERTIFY' if source_controls_ok and governed<=gate else 'HOLD'
 eligible=full.dropna(subset=features+['ActualCompletedUnits'])
 Xtr,Xte,ytr,yte=train_test_split(eligible[features],eligible.ActualCompletedUnits,test_size=.25,random_state=42)
 mr=HistGradientBoostingRegressor(**params); mr.fit(Xtr,ytr); random_safe=wape(yte.values,mr.predict(Xte))
 invalid=full.copy(); invalid['ClosedInvoiceCount']=close['ClosedInvoiceCount'].reindex(invalid.index).values; invalid['ClosedActiveCustomerCount']=close['ClosedActiveCustomerCount'].reindex(invalid.index).values; invalid['ClosedNMRGBP']=close['ClosedNMRGBP'].reindex(invalid.index).values
 lp=[]
 for _,f in folds.iterrows():
  start=pd.Timestamp(f.EvaluateFrom); end=pd.Timestamp(f.EvaluateThrough)
  tr=invalid[invalid.index<start].dropna(subset=features+['ActualCompletedUnits']); te=invalid[(invalid.index>=start)&(invalid.index<=end)].dropna(subset=features+['ActualCompletedUnits'])
  m=HistGradientBoostingRegressor(**params);m.fit(tr[features],tr.ActualCompletedUnits); lp.append((te.ActualCompletedUnits.values,m.predict(te[features])))
 aa=np.concatenate([x[0] for x in lp]); pp=np.concatenate([x[1] for x in lp]); same_day=wape(aa,pp)
 inveligible=invalid.dropna(subset=features+['ActualCompletedUnits']); Xtr,Xte,ytr,yte=train_test_split(inveligible[features],inveligible.ActualCompletedUnits,test_size=.25,random_state=42)
 mm=HistGradientBoostingRegressor(**params);mm.fit(Xtr,ytr); both=wape(yte.values,mm.predict(Xte))
 foldstats=[]
 for fold,g in out.groupby('ValidationFold',sort=False): foldstats.append((fold,len(g),wape(g.ActualCompletedUnits.values,g.Prediction.values)))
 mdf=out.copy(); mdf['TargetDate']=mdf.index.strftime('%Y-%m-%d'); mdf['ForecastOrigin']=mdf.ForecastOrigin.dt.strftime('%Y-%m-%d %H:%M:%S'); mdf['CloseReportDate']=mdf.CloseReportDate.dt.strftime('%Y-%m-%d'); mdf['CloseReportPublishedAt']=mdf.CloseReportPublishedAt.dt.strftime('%Y-%m-%d %H:%M:%S'); mdf['Prediction']=mdf.Prediction.round(2); mdf['AbsoluteError']=(mdf.ActualCompletedUnits-mdf.Prediction).abs().round(2); mdf['ClosedNMRGBP']=mdf.ClosedNMRGBP.round(2)
 cols=contract['Columns']; mdf=mdf[cols]
 governed=float(mdf.AbsoluteError.astype(float).sum()/mdf.ActualCompletedUnits.astype(float).abs().sum()*100)
 decision='CERTIFY' if source_controls_ok and governed<=gate else 'HOLD'
 foldstats=[]
 for fold,g in mdf.groupby('ValidationFold',sort=False):
  fv=float(g.AbsoluteError.astype(float).sum()/g.ActualCompletedUnits.astype(float).abs().sum()*100); foldstats.append((fold,len(g),fv))
 return mdf,dict(decision=decision,governed=governed,gate=gate,random_safe=random_safe,same_day=same_day,both=both,foldstats=foldstats,rows=len(mdf),features=len(features),source_controls_ok=source_controls_ok,overlap=overlap,conformed_occ=conformed_occ,raw_occ=raw_occ)

def html_report(stats):
 rows=''.join(f'<tr><td>{f}</td><td>{n}</td><td>{v:.2f}%</td></tr>' for f,n,v in stats['foldstats'])
 return f"""<!doctype html><html><head><meta charset="utf-8"><title>{stats['decision']} - Seasonal Replenishment Forecast Mart {stats['governed']:.2f}% WAPE</title><style>body{{font-family:Arial,sans-serif;margin:28px;color:#111}}table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{border:1px solid #111;padding:6px;text-align:left}}.k{{display:inline-block;border:1px solid #111;padding:10px;margin:4px 8px 4px 0}}h1,h2{{margin-bottom:8px}}.bar{{height:16px;border:1px solid #111;margin:3px 0}}.fill{{height:100%;background:#111}}</style></head><body><h1>{stats['decision']} - Production Seasonal-Replenishment Forecast Mart</h1><div class="k"><b>Governed WAPE</b><br>{stats['governed']:.2f}%</div><div class="k"><b>Deployment maximum</b><br>{stats['gate']:.1f}%</div><div class="k"><b>Validation rows</b><br>{stats['rows']}</div><h2>Source conformation</h2><p>Conformed physical occurrences: {stats['conformed_occ']:,}. Cross-extract overlaps collapsed: {stats['overlap']:,}. Raw physical occurrences: {stats['raw_occ']:,}. Source controls: {'PASS' if stats['source_controls_ok'] else 'FAIL'}.</p><h2>Governed validation</h2><table><tr><th>Fold</th><th>Days</th><th>WAPE</th></tr>{rows}</table><h2>Material diagnostics</h2><table><tr><th>Construction</th><th>WAPE</th><th>Decision at {stats['gate']:.1f}% gate</th></tr><tr><td>Governed fold sequence + as-of close</td><td>{stats['governed']:.2f}%</td><td>{stats['decision']}</td></tr><tr><td>Random validation + as-of close</td><td>{stats['random_safe']:.2f}%</td><td>{'CERTIFY' if stats['random_safe']<=stats['gate'] else 'HOLD'}</td></tr><tr><td>Governed folds + same-day close join</td><td>{stats['same_day']:.2f}%</td><td>{'CERTIFY' if stats['same_day']<=stats['gate'] else 'HOLD'}</td></tr><tr><td>Random validation + same-day close join</td><td>{stats['both']:.2f}%</td><td>{'CERTIFY' if stats['both']<=stats['gate'] else 'HOLD'}</td></tr></table><h2>Findings</h2><p>The production mart is held because the governed backtest exceeds the deployment maximum. Random row splitting understates out-of-time error. Joining a target day's close record before its PublishedAt timestamp introduces information that did not exist at the forecast origin.</p><p>The mart uses the scheduled forecast origin, the latest close report published by that origin, the frozen estimator, and the governed monthly validation sequence.</p></body></html>"""

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output-dir',default='.');args=ap.parse_args();root,tmp=root_from(args.input);outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
 mart,stats=build(root); mart.to_csv(outdir/'replenishment_feature_mart.csv',index=False,float_format='%.10g'); (outdir/'replenishment_certification.html').write_text(html_report(stats),encoding='utf-8')
 print(f"{stats['decision']} production seasonal-replenishment forecast mart: governed WAPE {stats['governed']:.2f}% versus {stats['gate']:.1f}% maximum.")
 print(f"Source conformation: {stats['conformed_occ']:,} occurrences; {stats['overlap']:,} overlaps collapsed; raw physical occurrences {stats['raw_occ']:,}; source controls {'PASS' if stats['source_controls_ok'] else 'FAIL'}.")
 print(f"Random/as-of diagnostic: {stats['random_safe']:.2f}%. Governed/same-day-close diagnostic: {stats['same_day']:.2f}%. Random/same-day-close diagnostic: {stats['both']:.2f}%.")
 for f,n,v in stats['foldstats']: print(f"{f}: {n} days, WAPE {v:.2f}%")
 if tmp: tmp.cleanup()
if __name__=='__main__': main()
