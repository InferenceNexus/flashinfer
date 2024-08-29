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
#ifndef FLASHINFER_GROUP_GEMM_WRAPPER_CUH_
#define FLASHINFER_GROUP_GEMM_WRAPPER_CUH_

#include <sstream>

#include "../allocator.h"
#include "../utils.cuh"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "handler.cuh"

namespace flashinfer {

namespace group_gemm {


using namespace cute;

#define DISPATCH_WEIGHT_LAYOUT(is_column_major, WEIGHT_LAYOUT, ...) \
  if (is_column_major) {                                            \
    using WEIGHT_LAYOUT = cutlass::layout::ColumnMajor;             \
    __VA_ARGS__                                                     \
  } else {                                                          \
    using WEIGHT_LAYOUT = cutlass::layout::RowMajor;                \
    __VA_ARGS__                                                     \
  }

template <typename DType>
cudaError_t CutlassSegmentGEMMWrapper(CutlassSegmentGEMMHandler* handler, DType* x, DType* w,
                                      DType* y, int64_t* xy_indptr_d, int64_t* w_indices_d,
                                      unsigned int batch_size, unsigned int d_in,
                                      unsigned int d_out, bool weight_column_major,
                                      cudaStream_t stream) {
  auto compute_capacity = GetCudaComputeCapability();
  if (compute_capacity.first < 8) {
    std::cerr << "CutlassSegmentGEMMWrapper requires compute capability of at least 8.0"
              << std::endl;
    return cudaErrorNotSupported;
  }
  if (compute_capacity.first == 8) {
    if (sizeof(DType) != 2) {
      std::cerr << "CutlassSegmentGEMMWrapper requires fp16/bf16 data type for compute capability 8.x"
                << std::endl;
      return cudaErrorNotSupported;
    }
    // SM80 grouped gemm
    AlignedAllocator allocator(handler->GetWorkspace(), handler->GetWorkspaceSizeInBytes());
    cutlass::gemm::GemmCoord* problem_sizes_device =
        allocator.aligned_alloc<cutlass::gemm::GemmCoord>(
            batch_size * sizeof(cutlass::gemm::GemmCoord), 16, "problem_sizes_device");
    DType** x_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "x_data");
    DType** w_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "w_data");
    DType** y_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "y_data");
    int64_t* ld_x = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_x");
    int64_t* ld_w = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_w");
    int64_t* ld_y = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_y");

    // NOTE(Zihao): I didn't successfully launch the kernel with cudaLaunchKernel API,
    // so I just use the kernel function directly, need to investigate more.
    auto compute_args_kernel = compute_sm80_cutlass_group_gemm_args<DType>;
    compute_args_kernel<<<batch_size, 1, 0, stream>>>(
        problem_sizes_device, x_data, w_data, y_data, ld_x, ld_w, ld_y, (DType*)x, (DType*)w,
        (DType*)y, xy_indptr_d, w_indices_d, d_in, d_out, weight_column_major);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
      return err;
    }

    using cutlass::epilogue::thread::LinearCombination;
    using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
    DISPATCH_WEIGHT_LAYOUT(weight_column_major, WEIGHT_LAYOUT, {
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          DType,                                   // Element A
          cutlass::layout::RowMajor,               // Layout A
          cutlass::ComplexTransform::kNone,        //
          8,                                       // Granularity A
          DType,                                   // Element B
          WEIGHT_LAYOUT,                           // Layout B
          cutlass::ComplexTransform::kNone,        //
          8,                                       // Granularity B
          DType,                                   // Element C&D
          cutlass::layout::RowMajor,               // Layout C&D
          float,                                   // Element Accumulator
          cutlass::arch::OpClassTensorOp,          // Operator Class Tag
          cutlass::arch::Sm80,                     // Architecture
          cutlass::gemm::GemmShape<128, 128, 32>,  // Thread Block Shape
          cutlass::gemm::GemmShape<64, 64, 32>,    // Warp Shape
          cutlass::gemm::GemmShape<16, 8, 16>,     // Instruction Shape
          cutlass::epilogue::thread::LinearCombination<DType, 8, float, float>,  // Epilogue
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,  // Swizzling Operator
          8                                                                   // Stages
          >::GemmKernel;

      using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
      typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);
      using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
      typename GemmGrouped::Arguments args(problem_sizes_device, batch_size, 4, epilogue_op, x_data,
                                           w_data, y_data, y_data, ld_x, ld_w, ld_y, ld_y);

      GemmGrouped gemm;
      auto status = gemm.initialize(args, nullptr, stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "cutlass group_gemm.initialize failed: " << cutlassGetStatusString(status);
        throw std::runtime_error(err_msg.str());
      }
      status = gemm.run(stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "cutlass group_gemm.run failed: " << cutlassGetStatusString(status);
        throw std::runtime_error(err_msg.str());
      }
    });
  } else {
    // Compute capability >= 9.0
    // Reference implementation
    // - https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
    using ProblemShape =
        cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
    // TODO(Zihao): dispatch
    using ElementA = cutlass::float_e4m3_t;  // Element type for A matrix operand
    using ElementB = cutlass::float_e4m3_t;  // Element type for B matrix operand
    using ElementC = cutlass::half_t;        // Element type for C and D matrix operands

    using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
    constexpr int AlignmentA =
        128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of elements
                                                      // (up to 16 bytes)

    // B matrix configuration
    using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
    constexpr int AlignmentB =
        128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of elements
                                                      // (up to 16 bytes)

    // C/D matrix configuration
    using LayoutC = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
    constexpr int AlignmentC =
        128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements
                                                      // (up to 16 bytes)

    // Core kernel configurations
    using ElementAccumulator = float;  // Element type for internal accumulation
    using ArchTag =
        cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
    // NOTE(Zihao): This tile size is for fp8, need to change for other types
    using TileShape = Shape<_256, _128, _128>;             // Threadblock-level tile size
    using ClusterShape = Shape<_2, _2, _1>;                // Shape of the threadblocks in a cluster
    using StageCountType =
        cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;  // Epilogue to launch

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC,
        EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using DeviceGemmReference =
        cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                                         ElementAccumulator, ElementAccumulator>;

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    // Host-side allocations
    std::vector<int64_t> offset_A;
    std::vector<int64_t> offset_B;
    std::vector<int64_t> offset_C;
    std::vector<int64_t> offset_D;

    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;

    std::vector<ElementAccumulator> alpha_host;
    std::vector<ElementAccumulator> beta_host;

    // Device-side allocations
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

    cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
    cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
    cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

    cutlass::DeviceAllocation<const typename Gemm::ElementA*> ptr_A;
    cutlass::DeviceAllocation<const typename Gemm::ElementB*> ptr_B;
    cutlass::DeviceAllocation<const typename Gemm::ElementC*> ptr_C;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput*> ptr_D;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput*> ptr_ref_D;

    cutlass::DeviceAllocation<StrideA> stride_A;
    cutlass::DeviceAllocation<StrideB> stride_B;
    cutlass::DeviceAllocation<StrideC> stride_C;
    cutlass::DeviceAllocation<StrideD> stride_D;

    // Note, this is an array of pointers to alpha and beta scaling values per group
    cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
    cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
    cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
    cutlass::DeviceAllocation<ElementAccumulator> block_beta;

    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::EpilogueOutputOp::Params params;
    params = typename Gemm::EpilogueOutputOp::Params(/*alpha=*/ElementAccumulator(1.f),
                                                     /*beta=*/ElementAccumulator(0.f));

    typename Gemm::Arguments arguments;
    // TODO(Zihao): fix this
    if (host_problem_shapes_available) {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
          {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
          {params, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
          hw_info};
    } else {
      arguments = typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {options.groups, problem_sizes.get(), nullptr},
          {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
          {params, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
          hw_info};
    }

    // init and allocate memory
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;

    for (int32_t i = 0; i < options.groups; ++i) {

      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = M * K;
      int64_t elements_B = K * N;
      int64_t elements_C = M * N;
      int64_t elements_D = M * N;

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;

      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

    }

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_ref_D.reset(total_elements_D);
    block_alpha.reset(options.groups);
    block_beta.reset(options.groups);
    // TODO(Zihao)
    uint64_t seed = 2020;

    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    //
    // Assign pointers
    //

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementC *> ptr_D_host(options.groups);
    std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

    for (int32_t i = 0; i < options.groups; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
      alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }

    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());

    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    alpha_device.reset(options.groups);
    alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups);
    beta_device.copy_from_host(ptr_beta_host.data());

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());

    // Initialize the gemm kernel

    Gemm gemm;

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    // NOTE(Zihao): use pre-allocated workspace
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());
  }

  return cudaSuccess;
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_WRAPPER_CUH_
