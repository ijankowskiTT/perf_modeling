// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_pages = get_arg(args::num_pages);
    const auto start_id = get_arg(args::start_id);

    DataflowBuffer cb(dfb::out);
    const uint32_t page_bytes = cb.get_entry_size();
    Noc noc;

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else


    const auto s = TensorAccessor(tensor::dst);

    constexpr uint32_t BATCH = 8;  // qperf validation: tiles per writer flush (4 and 8 were run)
    for (uint32_t i = start_id; i < start_id + num_pages; i += BATCH) {
        const uint32_t n = (start_id + num_pages - i) < BATCH ? (start_id + num_pages - i) : BATCH;
        cb.wait_front(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_write(cb, s, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = i + k});
        }
        noc.async_writes_flushed();
        cb.pop_front(n);
    }
    noc.async_write_barrier();
#endif
}
