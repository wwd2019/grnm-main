#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
P=/home/wwd/miniconda3/envs/scprint/bin/python
for ((i=0;i<2880;i++)); do
  if rg -q 'Writing results to file' run_pyscenic/grn.log && [ -s run_pyscenic/adjacencies.tsv ]; then
    size=$(stat -c %s run_pyscenic/adjacencies.tsv)
    sleep 60
    if [ "$size" = "$(stat -c %s run_pyscenic/adjacencies.tsv)" ]; then break; fi
  fi
  if rg -q 'Traceback \(most recent call last\)' run_pyscenic/grn.log; then echo 'GRN inference failed'; exit 1; fi
  sleep 30
done
rg -q 'Writing results to file' run_pyscenic/grn.log
"$P" run_pyscenic/cli.py ctx run_pyscenic/adjacencies.tsv resources/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather resources/hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather --annotations_fname resources/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl --expression_mtx_fname run_pyscenic/expression.loom --num_workers 4 --output run_pyscenic/motifs.csv
"$P" run_pyscenic/compare.py
