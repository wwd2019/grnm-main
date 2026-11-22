#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for ((i=0;i<2880;i++)); do
  if rg -q '^Complete$' run_pyscenic_restart/grn.log; then
    /home/wwd/miniconda3/envs/scprint/bin/python run_pyscenic_restart/prune.py
    exit $?
  fi
  if rg -q 'Traceback \(most recent call last\)' run_pyscenic_restart/grn.log; then echo 'GRN failed'; exit 1; fi
  sleep 30
done
exit 1
