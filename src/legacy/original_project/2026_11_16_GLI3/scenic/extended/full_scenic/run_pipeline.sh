#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/home/wwd/miniconda3/envs/scprint/bin/python
D=scenic/extended/full_scenic
if [ ! -s "$D/modules.pkl" ]; then "$P" "$D/prepare_modules.py" > "$D/prepare_modules.log" 2>&1; fi
"$P" "$D/prune_batches.py" > "$D/prune_batches.log" 2>&1
while [ ! -s "$D/rna_all_genes.txt" ]; do sleep 30; done
"$P" "$D/run_aucell.py" > "$D/aucell.log" 2>&1
