#ifndef CONV_1x1_CUH
#define CONV_1x1_CUH

#include "conv1x1_interface.hpp"
#include "fp8_gemm_1x1.h"
#include "tile_profile.cuh"
#include "traits.h"

void swizzleBias(int K, uint16_t* h_bias, uint16_t* d_bias) {
    std::vector<uint16_t> h_bias_swizzled(K);

    /*
    idx:        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, .., 64, 65, ..]
    swiz:       [0,  1,  2,  3,  16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51, 4,  5,  .., 64, 65, ..]
    */
    for (int ni = 0; ni < K; ni++) {
        int swiz = (ni/64)*64 + 16*((ni/4)&3) + 4*((ni%64)/16) + (ni%4); 
        h_bias_swizzled[ni] = h_bias[swiz];
    }

    gpuErrChk(cudaMemcpy(d_bias, &h_bias_swizzled[0], K * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

void swizzleB(int C, int K, uint8_t* h_B, uint8_t* d_B) {
    std::vector<uint8_t> h_B_swizzled(C * K);

    int swiz64[64];
    /*
    idx:        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, ..]
    new_offset: [0,  1,  8,  9,  16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 2,  3,  ..]
    swiz64:     [0,  1,  16, 17, 32, 33, 48, 49, 2,  3,  24, 25, 40, 41, 56, 57, 4,  5,  ..]
    */
    for (int i=0; i<64; i++) {
        int tid = i/16;
        int thread_offset = i%16;
        int new_offset = (thread_offset>>1)*8 + (thread_offset&1) + tid*2;
        swiz64[new_offset] = i;
    }
    for (int ni=0; ni<K; ni++) {
        int ni_swiz = 64*(ni/64) + swiz64[ni%64];
        for (int ki=0; ki<C; ki++) {
            h_B_swizzled[ni*C+ki] = h_B[ni_swiz*C+ki];
        }
    }

    gpuErrChk(cudaMemcpy(d_B, &h_B_swizzled[0], C * K, cudaMemcpyHostToDevice));
}

void swizzleQgmmaWeights(int C, int K, uint8_t* h_B, uint8_t* d_B, uint16_t* h_bias, uint16_t* d_bias) 
{
    // We use swizzle pattern 128, so channel needs to be a multiple of 128
    assert((C % 128) == 0);
    
    // The column swizzle pattern assumes K is a multiple of 64 channels
    assert((K % 64) == 0);
    
    swizzleB(C, K, h_B, d_B);
    
    swizzleBias(K, h_bias, d_bias);
}

template <bool RELU, bool GELU>
class Conv1x1 : public Conv1x1Interface {
    private: 
        using Traits = Kernel_traits_v1<RELU, GELU>;
        
        typename Traits::Scheduler::Host* _p_sched;
        typename Traits::Dma::Host* _p_dma;
        typename Traits::Compute::Host* _p_compute;
        ProfileParams _profileParams;
        cudaStream_t stream_;
        Fp8Gemm1x1Kernel const* mKernels{};
        int32_t mSM = 90;
        
    public:
        Conv1x1(uint8_t* workspace, cudaStream_t stream): stream_(stream)
        {
            _p_dma = new typename Traits::Dma::Host(&workspace, stream);
            _p_sched = new typename Traits::Scheduler::Host(&workspace, stream);
            _p_compute = new typename Traits::Compute::Host();

            mKernels = getFp8Gemm1x1Kernels(mSM);
        }

        static uint32_t getWorkSpaceSize(int K)
        {
            return Traits::Dma::Host::getWorkSpaceSize() + Traits::Scheduler::Host::getWorkSpaceSize(K);
        }

        void run(uint8_t* D, uint8_t* A, uint8_t* B, uint16_t* bias, float ab_scale, float d_scale, int N, int H, int W, int C, int K) override {
            // ... and if we're already making that assumption, let's just assume multiple of TILE_K so that we don't have to have a check in the epilogue
            assert((K % Traits::TILE_K) == 0);

            _p_dma->configure(N, H, W, C, K, A, B, bias);
            _p_sched->configure(N, H, W, K);
            _p_compute->configure(D, N, H, W, C, K, ab_scale, d_scale);

            int tiles_m = (N*H*W+Traits::TILE_M-1)/Traits::TILE_M;
            int tiles_n = (K+Traits::TILE_N-1)/Traits::TILE_N;
            int num_tiles = tiles_m*tiles_n;

            if (Traits::PROFILE_ENABLED)
            {
                gpuErrChk(cudaMalloc(&_profileParams.profile,num_tiles*sizeof(tile_profile)));
                gpuErrChk(cudaMemset(_profileParams.profile,0,sizeof(tile_profile)));
            }

            _profileParams.num_tiles = num_tiles;
            _profileParams.tiles_n = tiles_n;

            KernelParams params;
            params.scheduler_params = _p_sched->params();
            params.dma_params = _p_dma->params();
            params.compute_params = _p_compute->params();
            params.profile_params = _profileParams;
            params.relu = RELU;
            params.gelu = GELU;

            mKernels->run(params, stream_);

            // run_fp8_gemm_1x1(params, _p_sched->grid(), _threads, _smem_size, stream);
        }
};

#endif // CONV_1x1_CUH