#!/bin/sh
# Run one or more TDC ADMET tasks under the official protocol.
#
#   ./run.sh caco2_wang                 # one task, log to logs/<task>.log
#   ./run.sh ppbr_az vdss_lombardo      # several tasks in one process
#
# LightGBM needs libomp; run setup_libomp.sh once first (see that script).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$(dirname "$HERE")
# QV_VENV picks the interpreter: the default .venv is the round-1 environment,
# and .venv-tabpfn adds torch and TabPFN on top of identical core versions.
PY="$BASE/${QV_VENV:-.venv}/bin/python"

# Keep each process within the agreed footprint: 5 compute threads, so two
# concurrent processes use 10 of the machine's cores.
export ADMET_THREADS="${ADMET_THREADS:-5}"
export OMP_NUM_THREADS="$ADMET_THREADS"
export MKL_NUM_THREADS="$ADMET_THREADS"
export OPENBLAS_NUM_THREADS="$ADMET_THREADS"

mkdir -p "$BASE/logs" "$BASE/results"
# Name the log after the task arguments only, ignoring flags and their values.
# Everything from the first flag onwards is options, and some of them take several
# values (--tune-seeds takes fifteen), so only the leading task names become the log
# name. The old version skipped exactly one word after a flag and turned the rest into
# filename parts, which produced a 90-character log name.
LOGNAME=""
for a in "$@"; do
  case "$a" in
    --*) break ;;
    *) LOGNAME="${LOGNAME:+$LOGNAME+}$a" ;;
  esac
done
LOGNAME="${LOGNAME:-run}"

cd "$HERE"
echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) tasks: $* ===" >> "$BASE/logs/$LOGNAME.log"
nice -n 10 /usr/bin/time -l "$PY" -W ignore run_task.py "$@" 2>&1 \
  | tee -a "$BASE/logs/$LOGNAME.log"
echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) tasks: $* ===" >> "$BASE/logs/$LOGNAME.log"
