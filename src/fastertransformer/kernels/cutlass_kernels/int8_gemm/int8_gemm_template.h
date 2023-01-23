/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/ft_gemm_configs.h"

#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "src/fastertransformer/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "src/fastertransformer/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages>
void generic_int8_gemm_kernelLauncher(const int8_t*     A,
                                      const int8_t*     B,
                                      QuantMode         quant_mode,
                                      const float*      alpha_col,
                                      const float*      alpha_row,
                                      T*                C,
                                      int               m,
                                      int               n,
                                      int               k,
                                      CutlassGemmConfig gemm_config,
                                      char*             workspace,
                                      size_t            workspace_bytes,
                                      cudaStream_t      stream,
                                      int*              occupancy = nullptr)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef BUILD_CUTLASS_MIXED_GEMM

    using ElementInput = int8_t;

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t,
                                                ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using ElementAccumulator = int32_t;
    using ElementCompute     = float;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // TODO(mseznec): put below types in traits class
    using OperatorClass   = cutlass::arch::OpClassTensorOp;
    using DefaultGemmConf = typename cutlass::gemm::device::
        DefaultGemmConfiguration<OperatorClass, arch, ElementInput, ElementInput, ElementOutput, ElementCompute>;
    using InstructionShape = typename DefaultGemmConf::InstructionShape;
    using GemmOp           = typename DefaultGemmConf::Operator;
    using EpilogueOp       = typename DefaultGemmConf::EpilogueOutputOp;

    // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput,
                                                                    cutlass::layout::RowMajor,
                                                                    DefaultGemmConf::kAlignmentA,
                                                                    ElementInput,
                                                                    cutlass::layout::ColumnMajor,
                                                                    DefaultGemmConf::kAlignmentB,
                                                                    ElementOutput,
                                                                    cutlass::layout::RowMajor,
                                                                    ElementAccumulator,
                                                                    OperatorClass,
                                                                    arch,
                                                                    ThreadblockShape,
                                                                    WarpShape,
                                                                    InstructionShape,
                                                                    EpilogueOp,
                                                                    ThreadblockSwizzle,
                                                                    Stages,
                                                                    true,
                                                                    GemmOp>::GemmKernel;

    using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
            GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
            GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
            cutlass::sizeof_bits<ElementCompute>::value>,
        ElementCompute>;

    // Epilogue visitor
    using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
        ThreadblockShape,
        GemmKernel_::kThreadCount,
        AlphaColTileIterator,
        typename GemmKernel_::Epilogue::OutputTileIterator,
        ElementAccumulator,
        ElementCompute,
        EpilogueOp>;

    /// Epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
        EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

    // GEMM
    using GemmKernel =
        cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

    if (occupancy != nullptr) {
        *occupancy = compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBase<GemmKernel>;

    typename EpilogueOp::Params linear_scaling_params;  // TODO(mseznec): right now it's unused (scaling is done in
                                                        // visitor, no activation needed)
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                  {m, n, k},
                                  1,
                                  {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(A)), k},
                                  {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(B)), k},
                                  quant_mode,
                                  {reinterpret_cast<ElementCompute*>(const_cast<float*>(alpha_col)), 0},
                                  {reinterpret_cast<ElementCompute*>(const_cast<float*>(alpha_row)), 0},
                                  {nullptr, 0},
                                  {reinterpret_cast<ElementOutput*>(C), n},
                                  0,
                                  0,
                                  typename EpilogueVisitor::Arguments(linear_scaling_params, 0, 0, 0)};

    Gemm gemm;
    // TODO(mseznec): handle that
    if (gemm.get_workspace_size(args) > workspace_bytes) {
        FT_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg = "int8gemm cutlass kernel will fail for params. Error: "
                              + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to initialize cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][int8gemm Runner] " + err_msg);
    }
#else
    throw std::runtime_error(
        "[FT Error][int8gemm] FasterTransformer was built was mixed gemm support off. Please rebuild with cmake option -DBUILD_CUTLASS_MIXED_GEMM=ON");
#endif
}

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages, typename Enable = void>
struct dispatch_stages {
    static void dispatch(const int8_t*     A,
                         const int8_t*     B,
                         const float*      alpha_col,
                         const float*      alpha_row,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string err_msg = "Cutlass int8 gemm. Not instantiates for arch "
                              + std::to_string(arch::kMinComputeCapability) + " with stages set to "
                              + std::to_string(Stages);
        throw std::runtime_error("[FT Error][dispatch_stages::dispatch] " + err_msg);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
struct dispatch_stages<T, arch, ThreadblockShape, WarpShape, 2> {
    static void dispatch(const int8_t*     A,
                         const int8_t*     B,
                         QuantMode         quant_mode,
                         const float*      alpha_col,
                         const float*      alpha_row,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_kernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(A,
                                                                                  B,
                                                                                  quant_mode,
                                                                                  alpha_col,
                                                                                  alpha_row,
                                                                                  C,
                                                                                  m,
                                                                                  n,
                                                                                  k,
                                                                                  gemm_config,
                                                                                  workspace,
                                                                                  workspace_bytes,
                                                                                  stream,
                                                                                  occupancy);
    }
};

template<typename T, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatch_stages<T,
                       cutlass::arch::Sm80,
                       ThreadblockShape,
                       WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
    static void dispatch(const int8_t*     A,
                         const int8_t*     B,
                         QuantMode         quant_mode,
                         const float*      alpha_col,
                         const float*      alpha_row,
                         T*                C,
                         int               m,
                         int               n,
                         int               k,
                         CutlassGemmConfig gemm_config,
                         char*             workspace,
                         size_t            workspace_bytes,
                         cudaStream_t      stream,
                         int*              occupancy = nullptr)
    {

        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        generic_int8_gemm_kernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(A,
                                                                                                      B,
                                                                                                      quant_mode,
                                                                                                      alpha_col,
                                                                                                      alpha_row,
                                                                                                      C,
                                                                                                      m,
                                                                                                      n,
                                                                                                      k,
                                                                                                      gemm_config,
                                                                                                      workspace,
                                                                                                      workspace_bytes,
                                                                                                      stream,
                                                                                                      occupancy);
    }
};

template<typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void dispatch_gemm_config(const int8_t*     A,
                          const int8_t*     B,
                          QuantMode         quant_mode,
                          const float*      alpha_col,
                          const float*      alpha_row,
                          T*                C,
                          int               m,
                          int               n,
                          int               k,
                          CutlassGemmConfig gemm_config,
                          char*             workspace,
                          size_t            workspace_bytes,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.stages) {
        case 2:
            using DispatcherStages2 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 2>;
            DispatcherStages2::dispatch(A,
                                        B,
                                        quant_mode,
                                        alpha_col,
                                        alpha_row,
                                        C,
                                        m,
                                        n,
                                        k,
                                        gemm_config,
                                        workspace,
                                        workspace_bytes,
                                        stream,
                                        occupancy);
            break;
        case 3:
            using DispatcherStages3 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 3>;
            DispatcherStages3::dispatch(A,
                                        B,
                                        quant_mode,
                                        alpha_col,
                                        alpha_row,
                                        C,
                                        m,
                                        n,
                                        k,
                                        gemm_config,
                                        workspace,
                                        workspace_bytes,
                                        stream,
                                        occupancy);
            break;
        case 4:
            using DispatcherStages4 = dispatch_stages<T, arch, ThreadblockShape, WarpShape, 4>;
            DispatcherStages4::dispatch(A,
                                        B,
                                        quant_mode,
                                        alpha_col,
                                        alpha_row,
                                        C,
                                        m,
                                        n,
                                        k,
                                        gemm_config,
                                        workspace,
                                        workspace_bytes,
                                        stream,
                                        occupancy);
            break;
        default:
            std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
            throw std::runtime_error("[FT Error][dispatch_gemm_config] " + err_msg);
            break;
    }
}

template<typename T, typename arch>
void dispatch_gemm_to_cutlass(const int8_t*     A,
                              const int8_t*     B,
                              QuantMode         quant_mode,
                              const float*      alpha_col,
                              const float*      alpha_row,
                              T*                C,
                              int               m,
                              int               n,
                              int               k,
                              char*             workspace,
                              size_t            workspace_bytes,
                              CutlassGemmConfig gemm_config,
                              cudaStream_t      stream,
                              int*              occupancy = nullptr)
{

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Note that SIMT configs are omitted here since they are not supported for int8.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
    // for mixed type gemms.
    dispatch_gemm_config<T, arch, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(
        A, B, quant_mode, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream, occupancy);
    /* switch (gemm_config.tile_config) { */
    /*     case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64: */
    /*         dispatch_gemm_config<T, */
    /*                              arch, */
    /*                              cutlass::gemm::GemmShape<32, 128, 64>, */
    /*                              cutlass::gemm::GemmShape<32, 32, 64>>( */
    /*             A, B, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream, occupancy);
     */
    /*         break; */
    /*     case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64: */
    /*         dispatch_gemm_config<T, */
    /*                              arch, */
    /*                              cutlass::gemm::GemmShape<64, 128, 64>, */
    /*                              cutlass::gemm::GemmShape<64, 32, 64>>( */
    /*             A, B, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream, occupancy);
     */
    /*         break; */
    /*     case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64: */
    /*         dispatch_gemm_config<T, */
    /*                              arch, */
    /*                              cutlass::gemm::GemmShape<128, 128, 64>, */
    /*                              cutlass::gemm::GemmShape<128, 32, 64>>( */
    /*             A, B, alpha_col, alpha_row, C, m, n, k, gemm_config, workspace, workspace_bytes, stream, occupancy);
     */
    /*         break; */
    /*     case CutlassTileConfig::Undefined: */
    /*         throw std::runtime_error("[FT Error][int8][dispatch_gemm_to_cutlass] gemm config undefined."); */
    /*         break; */
    /*     case CutlassTileConfig::ChooseWithHeuristic: */
    /*         throw std::runtime_error( */
    /*             "[FT Error][int8][dispatch_gemm_to_cutlass] gemm config should have already been set by heuristic.");
     */
    /*         break; */
    /*     default: */
    /*         throw std::runtime_error( */
    /*             "[FT Error][int8][dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM."); */
    /*         break; */
    /* } */
}

template<typename T>
CutlassInt8GemmRunner<T>::CutlassInt8GemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    sm_ = getSMVersion();
    check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template<typename T>
CutlassInt8GemmRunner<T>::~CutlassInt8GemmRunner()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template<typename T>
void CutlassInt8GemmRunner<T>::dispatch_to_arch(const int8_t*     A,
                                                const int8_t*     B,
                                                QuantMode         quant_mode,
                                                const float*      alpha_col,
                                                const float*      alpha_row,
                                                T*                C,
                                                int               m,
                                                int               n,
                                                int               k,
                                                CutlassGemmConfig gemm_config,
                                                char*             workspace_ptr,
                                                const size_t      workspace_bytes,
                                                cudaStream_t      stream,
                                                int*              occupancy)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(mseznec): add compat for sm_ >= 70
    /* if (sm_ >= 70 && sm_ < 75) { */
    /*     dispatch_gemm_to_cutlass<T, cutlass::arch::Sm70>( */
    /*         A, B, alpha_col, alpha_row, C, m, n, k, workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
     */
    /* } */
    /* if (sm_ >= 75 && sm_ < 80) { */
    /*     dispatch_gemm_to_cutlass<T, cutlass::arch::Sm75>( */
    /*         A, B, alpha_col, alpha_row, C, m, n, k, workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
     */
    /* } */
    if (sm_ >= 80 && sm_ < 90) {
        dispatch_gemm_to_cutlass<T, cutlass::arch::Sm80>(A,
                                                         B,
                                                         quant_mode,
                                                         alpha_col,
                                                         alpha_row,
                                                         C,
                                                         m,
                                                         n,
                                                         k,
                                                         workspace_ptr,
                                                         workspace_bytes,
                                                         gemm_config,
                                                         stream,
                                                         occupancy);
    }
    else {
        throw std::runtime_error(
            "[FT Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}

template<typename T>
void CutlassInt8GemmRunner<T>::run_gemm(const int8_t* A,
                                        const int8_t* B,
                                        QuantMode     quant_mode,
                                        const float*  alpha_col,
                                        const float*  alpha_row,
                                        T*            C,
                                        int           m,
                                        int           n,
                                        int           k,
                                        char*         workspace_ptr,
                                        const size_t  workspace_bytes,
                                        cudaStream_t  stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    static constexpr bool          is_weight_only    = false;
    std::vector<CutlassGemmConfig> candidate_configs = get_candidate_configs(sm_, is_weight_only, false);
    std::vector<int>               occupancies(candidate_configs.size());

    for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
        dispatch_to_arch(A,
                         B,
                         quant_mode,
                         alpha_col,
                         alpha_row,
                         C,
                         m,
                         n,
                         k,
                         candidate_configs[ii],
                         workspace_ptr,
                         workspace_bytes,
                         stream,
                         &occupancies[ii]);
    }
    // Standard GEMM, so 1 "expert". We use the same function for MoE and regular FFN.
    static constexpr int num_experts   = 1;
    CutlassGemmConfig    chosen_config = estimate_best_config_from_occupancies(candidate_configs,
                                                                            occupancies,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            num_experts,
                                                                            split_k_limit,
                                                                            workspace_bytes,
                                                                            multi_processor_count_,
                                                                            is_weight_only);

    dispatch_to_arch(
        A, B, quant_mode, alpha_col, alpha_row, C, m, n, k, chosen_config, workspace_ptr, workspace_bytes, stream);
}

template<typename T>
void CutlassInt8GemmRunner<T>::gemm(const int8_t* A,
                                    const int8_t* B,
                                    QuantMode     quant_mode,
                                    const float*  alpha_row,
                                    const float*  alpha_col,
                                    T*            C,
                                    int           m,
                                    int           n,
                                    int           k,
                                    char*         workspace_ptr,
                                    const size_t  workspace_bytes,
                                    cudaStream_t  stream)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    run_gemm(A, B, quant_mode, alpha_row, alpha_col, C, m, n, k, workspace_ptr, workspace_bytes, stream);
}

template<typename T>
int CutlassInt8GemmRunner<T>::getWorkspaceSize(const int m, const int n, const int k)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}

}  // namespace fastertransformer