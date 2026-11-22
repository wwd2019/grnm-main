#!/usr/bin/env bash
set -uo pipefail
cd /home/wwd/py_code/mhr/final/new_grn || exit 1
source /home/wwd/miniconda3/etc/profile.d/conda.sh
conda activate spatialz || exit 1
stage=${1:?Specify a stage}
case "$stage" in
  train_glycolysis_qc|prepare_metacells|coarse_grain|scan_motifs|fit_grn|export_grn) language=R ;;
  match_cells) language=python ;;
  *) echo 'Unsupported stage' >&2; exit 2 ;;
esac
mkdir -p logs runs
rm -f "logs/${stage}.exitcode"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stamp=$(date +%Y%m%d_%H%M%S)
if [ "$language" = R ]; then
  snapshot="runs/${stage}_${stamp}.R"
  cp "${stage}.R" "$snapshot"
  /home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript "$snapshot" > "logs/${stage}.stdout" 2> "logs/${stage}.stderr"
else
  snapshot="runs/${stage}_${stamp}.py"
  cp "${stage}.py" "$snapshot"
  python -u "$snapshot" > "logs/${stage}.stdout" 2> "logs/${stage}.stderr"
fi
rc=$?
printf '%s\n' "$rc" > "logs/${stage}.exitcode"
tail -30 "logs/${stage}.stdout"
tail -30 "logs/${stage}.stderr" >&2
exit "$rc"
