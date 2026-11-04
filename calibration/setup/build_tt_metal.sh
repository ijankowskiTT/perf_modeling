#!/bin/bash
set -o pipefail
mkdir -p /localdev/$USER/rebuild /localdev/$USER/ccache
cd ${TREE:-/localdev/$USER/wt-perf-main}
export CCACHE_DIR=/localdev/$USER/ccache TT_METAL_HOME=$PWD
echo "== submodules $(date)"; git submodule update --init --recursive -q || exit 1
echo "== build $(date)"; ./build_metal.sh > /localdev/$USER/rebuild/build.log 2>&1 || { echo BUILD FAILED; tail -30 /localdev/$USER/rebuild/build.log; exit 1; }
echo "== venv $(date)"; PYTHON_ENV_DIR=/localdev/$USER/tt-metal-venv VENV_PYTHON_VERSION=3.10 ./create_venv.sh > /localdev/$USER/rebuild/venv.log 2>&1 || { echo VENV FAILED; tail -30 /localdev/$USER/rebuild/venv.log; exit 1; }
echo "== done $(date)"; ls ttnn/ttnn/_ttnn*.so build/lib 2>&1 | head
