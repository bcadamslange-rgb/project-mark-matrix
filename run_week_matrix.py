#!/usr/bin/env python3
"""Run the exact candidate under the current Python/sklearn wheel with three plausible WeekOfYear constructions."""
from pathlib import Path
import argparse, subprocess, sys, tempfile, os, json, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='Q4LightingReplenishment_InputPack.zip')
    ap.add_argument('--builder',default='replenishment_mart_build.py')
    ap.add_argument('--out',default='week_matrix_results.csv')
    args=ap.parse_args()
    import sklearn, numpy, pandas, scipy
    envdesc={'python':sys.version.split()[0],'scikit_learn':sklearn.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__}
    src=Path(args.builder).read_text(encoding='utf-8')
    needle="full.index.isocalendar().week.astype(int)"
    variants={'ISO':needle,'Sunday_%U':"full.index.strftime('%U').astype(int)",'Monday_%W':"full.index.strftime('%W').astype(int)"}
    rows=[]
    for name,expr in variants.items():
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); script=td/'builder.py'; script.write_text(src.replace(needle,expr),encoding='utf-8'); od=td/'out'; od.mkdir()
            r=subprocess.run([sys.executable,str(script),'--input',str(Path(args.input).resolve()),'--output-dir',str(od)],capture_output=True,text=True,check=True)
            mart=pd.read_csv(od/'replenishment_feature_mart.csv')
            wape=float(mart.AbsoluteError.astype(float).sum()/mart.ActualCompletedUnits.astype(float).abs().sum()*100)
            rows.append({**envdesc,'week_convention':name,'wape_pct':wape,'decision_at_51pct':'CERTIFY' if wape<=51 else 'HOLD'})
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False); print(df.to_string(index=False))
if __name__=='__main__': main()
