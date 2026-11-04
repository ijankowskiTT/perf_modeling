#!/bin/bash
# Creates /localdev/$USER/sim/{craq-b702fd90,perf} (env.sh + SoC descriptor) and checks out the simulator sources at the
# calibration commits. Then run calibration/setup/build_sims.sh.
set -e
L=/localdev/$USER; S=$L/sim; TREE=${TREE:-$L/wt-perf-main}
for n in craq-b702fd90 perf; do
    mkdir -p $S/$n
    cat > $S/$n/env.sh <<EOS
# Quasar simulator for qperf calibration ($n): source in a fresh shell.
export TT_METAL_HOME=$TREE
source $L/tt-metal-venv/bin/activate
export TT_METAL_SIMULATOR=$S/$n/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_CACHE=$S/$n-cache
unset ARCH_NAME CHIP_ARCH
export TT_METAL_LOGS_PATH=$S/$n-logs
mkdir -p \$TT_METAL_LOGS_PATH
export TT_METAL_PROFILER_DIR=\$TT_METAL_LOGS_PATH/generated/profiler
EOS
    cp $TREE/tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml $S/$n/soc_descriptor.yaml
done
[ -d $L/craq-sim-src ] || git clone -q /proj_sw/user_dev/gagandeepsingh/tt-blaze/craq-sim $L/craq-sim-src
[ -d $L/craq-sim-q ] || git -C $L/craq-sim-src worktree add -q --detach $L/craq-sim-q b702fd90
[ -d $L/ttsim-perf/src-repo ] || git clone -q git@github.com:tenstorrent/ttsim-private.git $L/ttsim-perf/src-repo
git -C $L/ttsim-perf/src-repo checkout -q 99f9de358abe36c5af7bce910ec09b93db772667
echo "ready: now run calibration/setup/build_sims.sh"
