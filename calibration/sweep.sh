#!/bin/bash
# Usage: sweep.sh <results.jsonl> <parallel> <case>...
#   case  = sim:layout:op:dtype:tpc:knobs   (e.g. craq-b702fd90:ddd:add:bf16:128:R4,C4,W2,B8,T8,E16)
#   sim   = a directory under $QPERF_SIM_DIR holding env.sh (sets TT_METAL_SIMULATOR etc.)
#   knobs = R/C/W threads, B dm_batch, T tiles_per_cycle, E entries_per_thread (any subset) or -
#   data-movement ops (tilize, untilize, tilize_with_val_padding, untilize_with_unpadding):
#   case = sim:mem:op:dtype:HxW:multicore, mem = dram | l1 | hs (e.g. craq-b702fd90:dram:tilize:bf16:256x512:1)
# Env: TREE (tt-metal worktree with its own build), QPERF_SIM_DIR, QPERF_RUNS (logs/cache, default ./runs).
# One process per case, so the program cache never blends configurations.
OUT=$1; PAR=$2; shift 2
: ${QPERF_SIM_DIR:?set QPERF_SIM_DIR}
export QPERF_SIM_DIR
export H=${QPERF_RUNS:-$PWD/runs}
export TREE=${TREE:?set TREE to a tt-metal worktree}
export HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $H
run_one() {
    IFS=: read -r sim layout op dtype tpc knobs <<< "$1"
    source $QPERF_SIM_DIR/$sim/env.sh >/dev/null 2>&1
    ulimit -c 0
    export TT_METAL_DEVICE_PROFILER=1
    local tag=$sim-$layout-$op-$dtype-$tpc-${knobs//,/}
    export TT_METAL_CACHE=$H/cache/$sim
    export TT_METAL_LOGS_PATH=$H/runs/$tag
    export TT_METAL_PROFILER_DIR=$TT_METAL_LOGS_PATH/generated/profiler
    rm -rf $TT_METAL_LOGS_PATH; mkdir -p $TT_METAL_LOGS_PATH $TT_METAL_CACHE
    if [ "$knobs" != "-" ] && [[ "$knobs" != [01] ]]; then
        IFS=, read -ra kv <<< "$knobs"
        for k in "${kv[@]}"; do
            v=${k:1}
            case ${k:0:1} in
                R) export TTNN_QSR_READER_THREADS=$v ;; C) export TTNN_QSR_COMPUTE_THREADS=$v ;;
                W) export TTNN_QSR_WRITER_THREADS=$v ;; B) export TTNN_QSR_DM_BATCH=$v ;;
                T) export TTNN_QSR_TILES_PER_CYCLE=$v ;; E) export TTNN_QSR_ENTRIES_PER_THREAD=$v ;;
                S) export TT_METAL_PROFILER_SUM=1 ;;
            esac
        done
    fi
    export TT_METAL_HOME=$TREE PYTHONPATH=$TREE:$TREE/ttnn:$TREE/tools${PYTHONPATH:+:$PYTHONPATH}
    cd $TREE
    local t_start=$(date +%s)
    if [ -f $HERE/runners/$op.py ]; then  # an op with its own runner: same arguments as run_dm_case.py
        timeout ${QPERF_TIMEOUT:-5400} python $HERE/runners/$op.py $op $dtype ${tpc%x*} ${tpc#*x} $layout $knobs > $TT_METAL_LOGS_PATH/stdout.log 2> $TT_METAL_LOGS_PATH/stderr.log
    else
    case $op in
        tilize|untilize|tilize_with_val_padding|untilize_with_unpadding|interleaved_to_sharded|sharded_to_interleaved|move)
            # data-movement ops: layout = mem (or shard spec, e.g. hs8), tpc = HxW, knobs = multicore
            timeout ${QPERF_TIMEOUT:-5400} python $HERE/run_dm_case.py $op $dtype ${tpc%x*} ${tpc#*x} $layout $knobs > $TT_METAL_LOGS_PATH/stdout.log 2> $TT_METAL_LOGS_PATH/stderr.log ;;
        *)
            timeout ${QPERF_TIMEOUT:-5400} python $HERE/run_case.py $op $dtype $tpc $layout > $TT_METAL_LOGS_PATH/stdout.log 2> $TT_METAL_LOGS_PATH/stderr.log ;;
    esac
    fi
    local line=$(grep '^JSON ' $TT_METAL_LOGS_PATH/stdout.log)
    local wall=$(( $(date +%s) - t_start ))
    local commit=$(git -C $TREE rev-parse --short=11 HEAD)
    if [ -n "$line" ]; then
        grep -ho 'binary_ng Quasar-native.*' $TT_METAL_LOGS_PATH/stdout.log $TT_METAL_LOGS_PATH/stderr.log > $TT_METAL_LOGS_PATH/factory.log
        echo "${line#JSON }" | SIM=$sim KN="$knobs" WALL=$wall COMMIT=$commit FL=$TT_METAL_LOGS_PATH/factory.log python3 -c "
import json, os, sys
d = json.load(sys.stdin)
d.update(sim=os.environ['SIM'], knobspec=os.environ['KN'], wall_s=int(os.environ['WALL']), commit=os.environ['COMMIT'],
         factory_log=open(os.environ['FL']).read().splitlines()[:6])
print(json.dumps(d))" >> $OUT
    else
        echo "{\"sim\": \"$sim\", \"layout\": \"$layout\", \"op\": \"$op\", \"dtype\": \"$dtype\", \"tpc\": \"$tpc\", \"knobspec\": \"$knobs\", \"error\": true, \"wall_s\": $wall, \"log\": \"$TT_METAL_LOGS_PATH\"}" >> $OUT
    fi
}
export -f run_one; export OUT
printf '%s\n' "$@" | xargs -P $PAR -I{} bash -c 'run_one {}'
