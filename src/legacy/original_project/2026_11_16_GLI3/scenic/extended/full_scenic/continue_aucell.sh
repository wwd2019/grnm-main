#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
D=scenic/extended/full_scenic
for ((i=0;i<2880;i++)); do
 if [ -s "$D/pruning_summary.json" ] && [ -s "$D/rna_all_genes.txt" ]; then
  /home/wwd/miniconda3/envs/scprint/bin/python "$D/run_aucell.py" > "$D/aucell.log" 2>&1
  exit $?
 fi
 if rg -q 'Traceback \(most recent call last\)' "$D/prune_batches.log"; then echo 'Motif pruning failed; AUCell not started'; exit 1; fi
 sleep 30
done
exit 1
