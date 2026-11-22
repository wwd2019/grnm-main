#!/usr/bin/env python3
"""Execute the R tutorial's code cells sequentially with Rscript."""
import argparse, json, os, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--rscript',default=os.environ.get('RSCRIPT','Rscript'));a=p.parse_args()
n=json.loads((root/'notebooks/R/01_pando_workflow.ipynb').read_text())
out=root/'results/generated/pando_tutorial_validation.R';out.parent.mkdir(parents=True,exist_ok=True)
code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
out.write_text(code)
subprocess.run([a.rscript,str(out)],cwd=root,check=True)
