#!/usr/bin/env python3
"""Re-render the three circular trees from the archived selected edge tables."""
from pathlib import Path
import os, shutil, subprocess
root = Path(__file__).resolve().parents[2]
source = root / 'results/model/phase4_trees'
out = root / 'results/generated/phase4_trees'
out.mkdir(parents=True, exist_ok=True)
rscript = os.environ.get('RSCRIPT') or shutil.which('Rscript')
if not rscript:
    raise SystemExit('Rscript not found; set RSCRIPT to an Rscript executable with required packages.')
watch = 'ZIC3,ZNF608,CUX1,HES1,JARID2,MAF,PBX1,SIX6,SOX11,EMX2,TOX,ZIC2,SIX3,SMAD3,SOX4,SALL4,DACH1,RAX,SOX6,ZNF521,PAX6,CREB5,DMRTA1,HES4,HES5'
for key, n1, hubs in [('grnm',80,6),('pando_metacells',50,6),('grnboost2',81,8)]:
    for ext in ('pdf','png'):
        target = out / f'{key}_tree.{ext}'
        cmd = [rscript, str(root/'src/visualization'/f'{key}_renderer.R'), '--edges', str(source/f'{key}_edges.csv'), '--root','GLI3','--output',str(target),'--n1',str(n1),'--nhubs',str(hubs),'--n2','5','--highlight',watch,'--label-all']
        subprocess.run(cmd, check=True)
        if not target.is_file() or target.stat().st_size < 1000:
            raise RuntimeError(f'Invalid output: {target}')
        print('Saved', target, flush=True)
