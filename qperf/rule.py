"""Mirror of the native factory's configuration rule and work split (kernel profile 'binary-ng-native').

Source of truth: binary_ng_utils.cpp resolve_native_config / native_max_tiles_per_cycle and
binary_ng_quasar_native_factory.cpp (tiles_per_compute_thread, batch_pays), tt_metal/common/work_split.cpp
split_work_to_cores. Each function names the C++ it follows; when the C++ changes, the kernel profile's
pinned commit must change with it.
"""

from __future__ import annotations

import math

# Ops whose bf16 path the factory sends to the SFPU kernel (is_binary_sfpu_op, binary_ng_device_operation.cpp:26).
# Only the ops the model has a compute profile for are listed; others are reported as unsupported.
_SFPU_ALWAYS = {"multiply", "divide"}  # MUL/DIV are SFPU unless fast_and_approximate_mode
_FPU_BF16 = {"add", "subtract", "rsub"}  # SFPU only for fp32/int dtypes


def is_sfpu(op, dtype):
    if op in _SFPU_ALWAYS:
        return True
    if op in _FPU_BF16:
        return dtype != "bf16"
    return None  # unknown routing: the caller must say


def dst_tiles_per_section(dtype_dst32):
    # get_dest_reg_count: 16 tiles full DST at 16-bit, half-sync -> 8; fp32_dest_acc_en halves it again.
    return 4 if dtype_dst32 else 8


def max_tiles_per_cycle(sfpu, dst32):
    # native_max_tiles_per_cycle: a DST section, the SFPU kernel takes 2 slots per tile.
    return dst_tiles_per_section(dst32) // (2 if sfpu else 1)


def bit_floor(x):
    return 1 << (x.bit_length() - 1) if x > 0 else 0


def split_work_to_cores(num_cores, units):
    """[(tiles_per_core, n_cores), ...] as split_work_to_cores gives them (group 1 first)."""
    if units == 0:
        return []
    target = min(num_cores, units)
    q, r = divmod(units, target)
    if r == 0:
        return [(q, target)]
    return [(q + 1, r), (q, target - r)]


def resolve_native_config(moves, knobs, user_dm_cores=6):
    """resolve_native_config (binary_ng_utils.cpp, HEAD c86eb2dc). moves: inputs, output, dram, sfpu, dst32,
    tiles_per_compute_thread, batch_pays, dram_batch. knobs: dict of set values (None = rule)."""
    rule_dm_batch = 8
    C = knobs.get("compute_threads") or 4
    R = (knobs.get("reader_threads") or C) if moves["inputs"] else 1
    writer_room = user_dm_cores - R if R < user_dm_cores else 1
    W = (knobs.get("writer_threads") or min(C, writer_room)) if moves["output"] else 1
    one_counter = R <= C and W <= C
    mtpc = max_tiles_per_cycle(moves["sfpu"], moves["dst32"])
    share = moves.get("tiles_per_compute_thread", 0)
    share_batches = moves.get("dram_batch", True) and moves.get("batch_pays", True) and share >= 4 and one_counter
    dram_batch = bit_floor(min(mtpc, share)) if share_batches else 1
    fpu_batches = R == 1 and C == 1 and W == 1
    l1_compute_batch = mtpc if (one_counter and (moves["sfpu"] or fpu_batches)) else 1
    dm_batch = knobs.get("dm_batch") or (dram_batch if moves["dram"] else rule_dm_batch)
    tpc = knobs.get("tiles_per_cycle") or (dram_batch if moves["dram"] else l1_compute_batch)
    lcm = dm_batch * tpc // math.gcd(dm_batch, tpc)
    two = 2 * max(dm_batch, tpc)
    ept = knobs.get("entries_per_thread") or (-(-two // lcm) * lcm)
    return {"R": R, "C": C, "W": W, "dm_batch": dm_batch, "tiles_per_cycle": tpc, "entries_per_thread": ept}
