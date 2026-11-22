#!/usr/bin/env bash
set -uo pipefail
cd /home/wwd/py_code/mhr/final/new_grn || exit 1
source /home/wwd/miniconda3/etc/profile.d/conda.sh
conda activate spatialz || exit 1
stage=${1:?Specify audit_inputs, audit_resources, or prepare_resources}
case "$stage" in
  audit_inputs|audit_resources|prepare_resources|prepare_inputs|prepare_cca|train_glycolysis_qc|coarse_grain|scan_motifs|fit_grn|export_grn) ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
mkdir -p logs
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
rscript=/home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript
"$rscript" "${stage}.R" > "logs/${stage}.stdout" 2> "logs/${stage}.stderr"
rc=$?
printf '%s\n' "$rc" > "logs/${stage}.exitcode"
cat "logs/${stage}.stdout"
cat "logs/${stage}.stderr" >&2
exit "$rc"
