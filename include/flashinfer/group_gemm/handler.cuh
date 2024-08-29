/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GROUP_GEMM_HANDLER_CUH_
#define FLASHINFER_GROUP_GEMM_HANDLER_CUH_

#include <cstddef>

#include "../allocator.h"
#include "../utils.cuh"
#include "group_gemm_cutlass.cuh"
#include "group_gemm_lora.cuh"
#include "group_gemv.cuh"

namespace flashinfer {

namespace group_gemm {

enum class GroupGEMMKernelConfig {
  kGeneral,  // large d_in, d_out
  kShrink,   // large d_in, small d_out
  kExpand,   // small d_in, large d_out
};

class CutlassSegmentGEMMHandler {
 public:
  void RegisterWorkspace(void* float_buffer, void* int_buffer, size_t float_buffer_size,
                         size_t int_buffer_size) {
    float_buffer_ = float_buffer;
    int_buffer_ = int_buffer;
    float_workspace_size_in_bytes_ = float_buffer_size;
    int_workspace_size_in_bytes_ = int_buffer_size;
  }

  void* GetFloatWorkspace() const { return float_buffer_; }

  void* GetIntWorkspace() const { return int_buffer_; }

  size_t GetFloatWorkspaceSizeInBytes() const { return float_workspace_size_in_bytes_; }

  size_t GetIntWorkspaceSizeInBytes() const { return int_workspace_size_in_bytes_; }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  CutlassSegmentGEMMHandler() {}

  ~CutlassSegmentGEMMHandler() {}

 private:
  void *float_buffer_, *int_buffer_;
  size_t float_workspace_size_in_bytes_, int_workspace_size_in_bytes_;
  cudaStream_t stream_;
};

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_HANDLER_CUH_
