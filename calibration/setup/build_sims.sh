#!/bin/bash
# Builds the two calibration simulators and installs them into /localdev/$USER/sim/<name>/libttsim.so.
set -e
S=/localdev/$USER/sim
echo "== craq-sim b702fd90"
cd /localdev/$USER/craq-sim-q/src && ../make.py --env TTSIM_LTO=0 _out/release_qsr/libttsim.so 2>&1 | tail -3
cp _out/release_qsr/libttsim.so $S/craq-b702fd90/
echo "== ttsim-perf 99f9de358abe (ENABLE_PERF=1)"
cd /localdev/$USER/ttsim-perf/src-repo/src && ../make.py --env ENABLE_PERF=1 _out/release_qsr/libttsim.so 2>&1 | tail -3
cp _out/release_qsr/libttsim.so $S/perf/
ls -la $S/craq-b702fd90/libttsim.so $S/perf/libttsim.so
