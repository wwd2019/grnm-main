#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SCENIC_WORKERS=8
P=/home/wwd/miniconda3/envs/scprint/bin/python
"$P" extended/run/grn.py > extended/run/grn.log 2>&1
"$P" extended/run/prune.py > extended/run/prune.log 2>&1
