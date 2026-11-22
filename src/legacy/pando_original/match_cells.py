"""Run the authors' MCMF implementation; only adapt SciPy's renamed keyword."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

root = Path('/home/wwd/py_code/mhr/final/new_grn')
original = root / 'reference/integration/matching.py'
source = original.read_text()
assert source.count('n_jobs=knn_n_jobs') == 1
adapted = source.replace('n_jobs=knn_n_jobs', 'workers=knn_n_jobs')
module = importlib.util.module_from_spec(importlib.util.spec_from_loader('paper_matching', loader=None))
exec(compile(adapted, str(original), 'exec'), module.__dict__)
out = root / 'matches'
out.mkdir(exist_ok=True)
results = []
for file in sorted((root / 'cca').glob('*.tsv')):
    target_file = out / file.name
    if target_file.exists():
        print('RESUME', file.stem, flush=True)
        continue
    frame = pd.read_csv(file, sep='\t', index_col='cell')
    modalities = frame.pop('modality')
    first = modalities.value_counts().idxmax()
    second = 'ATAC' if first == 'RNA' else 'RNA'
    source_group = frame.loc[modalities == first]
    target_group = frame.loc[modalities == second]
    print('MATCH', file.stem, first, len(source_group), second, len(target_group), flush=True)
    graph = module.get_cost_knn_graph(source_group, target_group, knn_k=10,
        knn_n_jobs=4, null_cost_percentile=99, capacity_method='uniform')
    i, j = module.mcmf(graph)
    good = np.isfinite(i) & np.isfinite(j)
    pairs = pd.DataFrame({first: source_group.index[i[good].astype(int)],
                          second: target_group.index[j[good].astype(int)]})
    pairs['group'] = file.stem
    pairs.to_csv(str(target_file) + '.partial', sep='\t', index=False)
    Path(str(target_file) + '.partial').replace(target_file)
    results.append({'group': file.stem, 'source': len(source_group),
                    'target': len(target_group), 'matched': int(good.sum()),
                    'unmatched_null': int((~good).sum())})
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
files = sorted(f for f in out.glob('*.tsv') if f.name != 'all_pairs.tsv')
assert len(files) == len(list((root / 'cca').glob('*.tsv'))) > 0
pairs = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
assert pairs[['RNA', 'ATAC']].notna().all().all()
pairs.to_csv(out / 'all_pairs.tsv', sep='\t', index=False)
print('MCMF_COMPLETE', len(pairs), flush=True)
