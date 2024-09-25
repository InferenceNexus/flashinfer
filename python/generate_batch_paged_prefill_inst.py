"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import re
import itertools
from literal_map import (
    mask_mode_literal,
    pos_encoding_mode_literal,
    warp_layout_literal,
    dtype_literal,
    idtype_literal,
)
from pathlib import Path


def get_cu_file_str(
    head_dim,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    mask_mode,
    dtype_q,
    dtype_kv,
    dtype_out,
    idtype,
):
    warp_layout_choice = [0, 1, 2]
    insts = "\n".join(
        [
            """template cudaError_t BatchPrefillWithPagedKVCacheDispatched<{warp_layout}, {head_dim}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {mask_mode}, AttentionVariant>(
    typename AttentionVariant::ParamsT params,
    typename AttentionVariant::DTypeO* tmp_v,
    float* tmp_s, cudaStream_t stream);
    """.format(
                warp_layout=warp_layout_literal[warp_layout],
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                allow_fp16_qk_reduction=allow_fp16_qk_reduction,
                mask_mode=mask_mode_literal[int(mask_mode)],
            )
            for warp_layout in warp_layout_choice
        ]
    )

    use_custom_mask = "true" if int(mask_mode) == 2 else "false"
    dtype_q = dtype_literal[dtype_q]
    dtype_kv = dtype_literal[dtype_kv]
    dtype_out = dtype_literal[dtype_out]
    idtype = idtype_literal[idtype]

    content = f"""#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

using ParamsT = BatchPrefillPagedParams<{dtype_q}, {dtype_kv}, {dtype_out}, {idtype}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code({use_custom_mask}, false, false, false)>;

{insts}

}}"""
    return content


if __name__ == "__main__":
    pattern = (
        r"batch_paged_prefill_head_([0-9]+)_posenc_([0-9]+)_"
        r"fp16qkred_([a-z]+)_mask_([0-9]+)_dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)_idtype_([a-z0-9]+)\.cu"
    )
    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
