"""Configuration validity: the native factory's admission rules plus the cluster L1 budget.

Every rule names the check it mirrors. A violation is returned as a reason, never silently "fixed".
"""

from __future__ import annotations

from .rule import max_tiles_per_cycle


def check(cfg, routing, arch, kernel):
    """Return (errors, warnings, l1) for a resolved config. routing: in_moves [a,b], out_moves, sfpu, dst32."""
    errors, warnings = [], []
    R, C, W = cfg["R"], cfg["C"], cfg["W"]
    B, BC, K = cfg["dm_batch"], cfg["tiles_per_cycle"], cfg["entries_per_thread"]
    lim = arch["limits"]
    kl = dict(kernel.get("knob_limits", {}))
    kl.update(routing.get("knob_limits_override") or {})
    for name, key in (("dm_batch", "dm_batch"), ("tiles_per_cycle", "tiles_per_cycle"),
                      ("entries_per_thread", "entries_per_thread"), ("reader_threads", "R"),
                      ("writer_threads", "W"), ("compute_threads", "C")):
        if name in kl and cfg[key] > kl[name]:
            errors.append(f"{name}={cfg[key]} exceeds the TTNN_QSR knob ceiling {kl[name]} (binary_ng_utils.cpp "
                          f"read_knob); a hypothetical.knob_limits override models a factory change")

    if C not in lim["compute_threads_allowed"]:
        errors.append(f"C={C} not in {lim['compute_threads_allowed']} (native_tuning TT_FATAL)")
    if R < 1 or W < 1:
        errors.append("R and W must be >= 1")
    if R + W > lim["user_dm_cores"]:
        errors.append(f"R+W={R + W} exceeds {lim['user_dm_cores']} user DM cores (native_tuning TT_FATAL)")
    for a, b, name in ((R, C, "in (R,C)"), (C, W, "out (C,W)")):
        if max(a, b) % min(a, b):
            errors.append(f"STRIDED ratio: max%min != 0 for {name}={a},{b} (calculate_num_tile_counters TT_FATAL)")
    mtpc = max_tiles_per_cycle(routing["sfpu"], routing["dst32"])
    if BC > mtpc:
        errors.append(f"tiles_per_cycle={BC} exceeds DST section capacity {mtpc} for this kernel (factory TT_FATAL)")
    all_borrowed = not any(routing["in_moves"]) and not routing["out_moves"]
    if not all_borrowed:
        for label, b in (("tiles_per_cycle", BC), ("dm_batch", B)):
            if K < 2 * b:
                errors.append(f"{label}={b} needs entries_per_thread >= {2 * b} (factory double-buffer TT_FATAL)")
            if K % b:
                errors.append(f"{label}={b} must divide entries_per_thread={K} (factory ring-end TT_FATAL)")
        if BC != 1 and not (R <= C and W <= C):
            errors.append(f"tiles_per_cycle={BC} needs R<=C and W<=C (factory TT_FATAL; one counter per thread)")
    if R > C or W > C:
        warnings.append("R>C or W>C: STRIDED rings with several counters per compute thread; historically "
                        "produced wrong output (tt-metal#55276) - check correctness before trusting timing")
    in_entries = K * max(R, C)
    out_entries = K * max(C, W)
    cap = lim["max_ring_entries"]
    if any(routing["in_moves"]) and in_entries > cap:
        errors.append(f"in ring num_entries={in_entries} > {cap} (unguarded uint8 threshold, factory TT_FATAL)")
    if routing["out_moves"] and out_entries > cap:
        errors.append(f"out ring num_entries={out_entries} > {cap} (factory TT_FATAL)")
    if routing.get("tail_tiles"):
        warnings.append("borrowed shard with tail rings: tail-ring cost (~1000 cyc fixed) is not modeled")

    # L1 budget: owned rings only (a borrowed ring is the tensor's own L1).
    tb = routing["tile_bytes"]
    ring_bytes = 0
    if routing["in_moves"][0]:
        ring_bytes += in_entries * tb[0]
    if routing["in_moves"][1]:
        ring_bytes += in_entries * tb[1]
    if routing["out_moves"]:
        ring_bytes += out_entries * tb[2]
    resident = routing.get("resident_l1_bytes", 0)
    avail = arch["l1"]["bytes_per_cluster"] - arch["l1"]["reserved_bytes"]
    l1 = {"ring_bytes": ring_bytes, "resident_tensor_bytes": resident, "available_bytes": avail,
          "fits": ring_bytes + resident <= avail}
    if not l1["fits"]:
        errors.append(f"L1: rings {ring_bytes} B + resident tensors {resident} B > available {avail} B")
    return errors, warnings, l1
