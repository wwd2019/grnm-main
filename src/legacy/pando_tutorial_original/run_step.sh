#!/usr/bin/env bash
# Run one documented step on l40; keep tutorial logs separate from historical logs.
set -euo pipefail
work=/home/wwd/py_code/mhr/final/new_grn
step=${1:-list}
case "$step" in
  list) printf '%s\n' 'resources features fragments activity cca matching qc pseudocells background motifs glm export inspect'; exit 0 ;;
  resources) script=prepare_resources.R ;;
  features) script=prepare_feature_sets.R ;;
  fragments) script=extract_fragments.py ;;
  activity) script=prepare_inputs.R ;;
  cca) script=prepare_cca.R ;;
  matching) script=match_cells.py ;;
  qc) script=train_glycolysis_qc.R ;;
  pseudocells) script=coarse_grain.R ;;
  background) script=prepare_motif_background.R ;;
  motifs) script=scan_motifs.R ;;
  glm) script=fit_grn.R ;;
  export) script=export_grn.R ;;
  inspect) script=tutorials/read_results.R ;;
  *) printf 'Unknown step: %s\n' "$step" >&2; exit 2 ;;
esac
cd "$work"
source /home/wwd/miniconda3/etc/profile.d/conda.sh
conda activate spatialz
export PANDO_DATADIR=/home/wwd/py_code/mhr/final
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stamp=$(date +%Y%m%d_%H%M%S)
logdir="logs/tutorial_${step}_${stamp}_$$"
mkdir -p "$logdir"
snapshot="$logdir/$(basename "$script")"
cp "$script" "$snapshot"
printf 'STEP=%s\nLOGDIR=%s/%s\n' "$step" "$work" "$logdir"
set +e
if [[ "$script" == *.R ]]; then
  /home/wwd/miniconda3/envs/seurat4_env_restored/bin/Rscript "$snapshot" > "$logdir/stdout" 2> "$logdir/stderr"
else
  python -u "$snapshot" > "$logdir/stdout" 2> "$logdir/stderr"
fi
rc=$?
set -e
printf '%s\n' "$rc" > "$logdir/exitcode"
tail -40 "$logdir/stdout"
tail -40 "$logdir/stderr" >&2
printf 'EXIT_CODE=%s\n' "$rc"
exit "$rc"
