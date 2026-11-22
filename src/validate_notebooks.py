#!/usr/bin/env python3
"""Validate notebook schema and execute code cells without a Jupyter kernel.

Local sandbox blocks the sockets used by Jupyter kernels. This executes the same
Python cell source sequentially; it is not a test of a Jupyter frontend/kernel.
"""
from pathlib import Path
import json,os,sys,traceback
os.environ.setdefault('MPLBACKEND','Agg')
root=Path(__file__).resolve().parents[1];os.chdir(root)
paths = sorted((root/'notebooks/python').glob('*.ipynb')) + sorted((root/'notebooks/planarian').glob('*.ipynb'))
for path in paths:
 notebook=json.loads(path.read_text());assert notebook['nbformat']==4 and notebook['nbformat_minor']>=5
 scope={'__name__':'__main__'}
 print('NOTEBOOK',path.name,flush=True)
 for i,cell in enumerate(notebook['cells']):
  assert cell['cell_type'] in {'code','markdown'} and isinstance(cell['source'],list)
  if cell['cell_type']=='code':
   try: exec(compile(''.join(cell['source']),f'{path.name}:cell_{i}','exec'),scope)
   except Exception:
    print('FAILED:',path.name,'cell',i,flush=True);traceback.print_exc();sys.exit(1)
 print('PASSED',path.name,flush=True)
print('All notebook cell sources passed direct execution; Jupyter kernel not tested.')
