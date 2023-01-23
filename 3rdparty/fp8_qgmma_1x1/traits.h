#ifndef TRAITS_H
#define TRAITS_H

#include "scheduler.cuh"
#include "dma.cuh"
#include "compute.cuh"

template<
    int TILE_M_, 
    int TILE_N_, 
    int TILE_K_, 
    int STAGES_, 
    bool PERSISTENT_WEIGHTS_, 
    int CTAS_PER_CGA_, 
    bool PROFILE_ENABLED_, 
    bool USE_COMPUTE_MUTEX_, 
    bool MULTICAST_, 
    bool INTERLEAVED_,
    bool RELU_, 
    bool GELU_
>
struct Kernel_traits_ {

    // The version.
    enum { TILE_M = TILE_M_ };
    enum { TILE_N = TILE_N_ };
    enum { TILE_K = TILE_K_ };
    enum { STAGES = STAGES_ };
    enum { CTAS_PER_CGA = CTAS_PER_CGA_ };
    enum : bool { PERSISTENT_WEIGHTS = PERSISTENT_WEIGHTS_ };
    enum : bool { PROFILE_ENABLED = PROFILE_ENABLED_ };
    enum : bool { USE_COMPUTE_MUTEX = USE_COMPUTE_MUTEX_ };
    enum : bool { MULTICAST = MULTICAST_ };
    enum : bool { INTERLEAVED = INTERLEAVED_ };
    enum : bool { RELU = RELU_ };
    enum : bool { GELU = GELU_ };

    using Scheduler = DynamicScheduler<TILE_M, TILE_N, PERSISTENT_WEIGHTS, CTAS_PER_CGA>;
    using Dma = DMA<TILE_M, TILE_N, TILE_K, INTERLEAVED>;
    using Compute = Compute;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
    bool RELU = false,
    bool GELU = false
>
using Kernel_traits_v1 = Kernel_traits_<128,
                                        128,
                                        128, 
                                        6,
                                        false, 
                                        1, 
                                        false, 
                                        true, 
                                        false, 
                                        false,
                                        RELU,
                                        GELU>;

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // TRAITS_H