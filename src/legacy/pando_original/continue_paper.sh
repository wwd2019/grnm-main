#!/usr/bin/env bash
set -euo pipefail
cd /home/wwd/py_code/mhr/final/new_grn
run_if_needed() {
  local stage=$1
  if [ -f "logs/${stage}.exitcode" ] && [ "$(cat "logs/${stage}.exitcode")" = 0 ]; then
    printf 'RESUME %s\n' "$stage"
  else
    bash run_paper_stage.sh "$stage"
  fi
}
test "$(cat logs/prepare_cca.exitcode)" = 0
run_if_needed match_cells
run_if_needed train_glycolysis_qc
test -s qc/nowakowski_rank_lasso.rds
run_if_needed coarse_grain
run_if_needed scan_motifs
run_if_needed fit_grn
run_if_needed export_grn
printf 'PAPER_GRN_PIPELINE_COMPLETE\n'
