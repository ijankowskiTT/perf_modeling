# Calibration machine setup

Recreates the simulator and tt-metal setup the costs were fitted on, on any machine with `/localdev`.

1. tt-metal worktree at the calibration commit, with its own build and venv:
   ```bash
   git -C <tt-metal clone> worktree add --detach /localdev/$USER/wt-perf-main 0fff850f3c6
   calibration/setup/build_tt_metal.sh            # submodules, build_metal.sh, create_venv.sh (~40 min)
   ```
2. Simulators (craq-sim `b702fd90`, ttsim-perf `99f9de358abe`) and their env files:
   ```bash
   calibration/setup/make_sim_dirs.sh             # sim/<name>/env.sh + soc_descriptor.yaml, clones the sources
   calibration/setup/build_sims.sh                # builds both libttsim.so (run it yourself)
   ```
3. Check: re-run a few cases from `calibration/data/*.jsonl` with `calibration/sweep.sh`; they must match exactly.

`~/.cache` links that point into `/localdev` (pip, uv, tt-metal-cache) need their target folders on each machine
(`mkdir -p /localdev/$USER/home-cache/{tt-metal-cache,pip,pre-commit,torch,uv}`), or the build fails.
