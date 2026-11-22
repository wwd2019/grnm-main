#!/usr/bin/env bash
set -uo pipefail
cd /home/wwd/py_code/mhr/final/new_grn || exit 1
source /home/wwd/miniconda3/etc/profile.d/conda.sh
conda activate spatialz || exit 1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stamp=$(date +%Y%m%d_%H%M%S)
snapshot="runs/fit_reverse_workers_${stamp}.R"
cp fit_reverse_workers.R "$snapshot"
/home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript "$snapshot" > logs/fit_reverse.stdout 2> logs/fit_reverse.stderr
rc=$?
printf '%s\n' "$rc" > logs/fit_reverse.exitcode
tail -15 logs/fit_reverse.stdout
tail -15 logs/fit_reverse.stderr >&2
exit "$rc"
