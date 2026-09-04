#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def run(convention, root):
    parts=sorted(root.glob('daily_close_part*.csv'))
    if not parts:
        raise SystemExit('No daily_close_part*.csv files found')
    close=pd.concat([pd.read_csv(p,parse_dates=['ReportDate','PublishedAt']) for p in parts],ignore_index=True).sort_values('ReportDate').set_index('ReportDate')
    folds=pd.read_csv(root/'forecast_validation_calendar.csv')
    spec=json.load(open(root/'forecast_model_spec.json'))
    contract=json.load(open(root/'forecast_feature_contract.json'))
    gate=float(pd.read_csv(root/'forecast_deployment_gate.csv').iloc[0]['MaximumPct'])
    full=pd.DataFrame(index=pd.date_range(close.index.min(),pd.Timestamp('2011-09-30'),freq='D'))
    full['ActualCompletedUnits']=close['CompletedUnits'].reindex(full.index).fillna(0).astype(float)
    full['DayOfWeek']=full.index.dayofweek
    full['Month']=full.index.month
    if convention=='ISO': full['WeekOfYear']=full.index.isocalendar().week.astype(int)
    elif convention=='Sunday_%U': full['WeekOfYear']=full.index.strftime('%U').astype(int)
    elif convention=='Monday_%W': full['WeekOfYear']=full.index.strftime('%W').astype(int)
    else: raise ValueError(convention)
    full['SinDayOfWeek']=np.sin(2*np.pi*full.DayOfWeek/7); full['CosDayOfWeek']=np.cos(2*np.pi*full.DayOfWeek/7)
    full['SinDayOfYear']=np.sin(2*np.pi*full.index.dayofyear/365.25); full['CosDayOfYear']=np.cos(2*np.pi*full.index.dayofyear/365.25)
    full['TimeIndex']=(full.index-full.index.min()).days
    for lag in [1,7,14,28]: full[f'UnitsLag{lag}']=full.ActualCompletedUnits.shift(lag)
    full['UnitsRolling7Mean']=full.ActualCompletedUnits.shift(1).rolling(7).mean(); full['UnitsRolling28Mean']=full.ActualCompletedUnits.shift(1).rolling(28).mean()
    full['ForecastOrigin']=full.index+pd.Timedelta(hours=5)
    pub=close[['PublishedAt','ClosedInvoiceCount','ClosedActiveCustomerCount','ClosedNMRGBP']].reset_index().sort_values('PublishedAt')
    left=pd.DataFrame({'TargetDate':full.index,'ForecastOrigin':full.ForecastOrigin.values}).sort_values('ForecastOrigin')
    asof=pd.merge_asof(left,pub,left_on='ForecastOrigin',right_on='PublishedAt',direction='backward').set_index('TargetDate')
    full['CloseReportDate']=asof['ReportDate']; full['CloseReportPublishedAt']=asof['PublishedAt']
    full['ClosedInvoiceCount']=asof['ClosedInvoiceCount']; full['ClosedActiveCustomerCount']=asof['ClosedActiveCustomerCount']; full['ClosedNMRGBP']=asof['ClosedNMRGBP']
    features=spec['PredictorColumns']; params=spec['Parameters']; preds=[]
    for _,f in folds.iterrows():
        start=pd.Timestamp(f.EvaluateFrom); end=pd.Timestamp(f.EvaluateThrough)
        tr=full[full.index<start].dropna(subset=features+['ActualCompletedUnits'])
        te=full[(full.index>=start)&(full.index<=end)].dropna(subset=features+['ActualCompletedUnits'])
        m=HistGradientBoostingRegressor(**params); m.fit(tr[features],tr.ActualCompletedUnits)
        part=te.copy(); part['Prediction']=m.predict(te[features]); part['ValidationFold']=f.Fold; preds.append(part)
    out=pd.concat(preds).sort_index()
    mdf=out.copy(); mdf['TargetDate']=mdf.index.strftime('%Y-%m-%d'); mdf['ForecastOrigin']=mdf.ForecastOrigin.dt.strftime('%Y-%m-%d %H:%M:%S'); mdf['CloseReportDate']=mdf.CloseReportDate.dt.strftime('%Y-%m-%d'); mdf['CloseReportPublishedAt']=mdf.CloseReportPublishedAt.dt.strftime('%Y-%m-%d %H:%M:%S'); mdf['Prediction']=mdf.Prediction.round(2); mdf['AbsoluteError']=(mdf.ActualCompletedUnits-mdf.Prediction).abs().round(2); mdf['ClosedNMRGBP']=mdf.ClosedNMRGBP.round(2)
    mdf=mdf[contract['Columns']]
    wape=float(mdf.AbsoluteError.astype(float).sum()/mdf.ActualCompletedUnits.astype(float).abs().sum()*100)
    return wape, gate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='matrix_input'); ap.add_argument('--out',default='week_matrix_results.csv'); args=ap.parse_args()
    import sklearn, scipy
    root=Path(args.root)
    env={'python':sys.version.split()[0],'scikit_learn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    rows=[]
    for c in ['ISO','Sunday_%U','Monday_%W']:
        w,g=run(c,root); rows.append({**env,'week_convention':c,'wape_pct':w,'decision_at_51pct':'CERTIFY' if w<=51 else 'HOLD'})
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False); print(df.to_string(index=False))
if __name__=='__main__': main()
