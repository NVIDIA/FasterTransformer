#include "fp8_gemm_1x1.h"
#include "utils.h"

template <int TILE_M, int TILE_N, int TILE_K, bool INTERLEAVED>
struct DMA {

    struct DMAHostDescriptors {
        cudaTmaDesc a_desc;
        cudaTmaDesc b_desc;
        cudaTmaDesc bias_desc;
    };

    struct Host {
        static const uint32_t cudaTmaDescAlignment = 64;
        DMAParams _params;
        uint8_t* workspace_;
        cudaStream_t stream_;

        __host__ Host( uint8_t** workspace, cudaStream_t stream): workspace_(*workspace), stream_(stream) {
            if (((uintptr_t)*workspace) % cudaTmaDescAlignment != 0)
            {
                throw std::runtime_error("Address of workspace is not aligned to cudaTmaDescAlignment");
            }
    
            *workspace += getWorkSpaceSize();
        }

        __host__ void configure(int N, int H, int W, int C, int K, uint8_t* A, uint8_t* B, uint16_t* bias)
        {
            _params.C = C;
            _params.HW = H*W;
            _params.H = H;
            _params.W = W;

            if (INTERLEAVED) createTmaDescriptors(N,H,W,C,K,A,B,bias,workspace_, stream_);
            else createTmaDescriptors(N*H*W,1,1,C,K,A,B,bias,workspace_, stream_);

            if (INTERLEAVED) {
                // Fast divmod
                find_divisor(_params.mul_hw, _params.shr_hw, H*W);
                find_divisor(_params.mul_w, _params.shr_w, W);
            }
        }

        static __host__ uint32_t getWorkSpaceSize()
        {
            // descriptor a_desc + b_desc + bias_desc
            return sizeof(cudaTmaDesc) * 3;
        }

        __host__ void createTmaDescriptors(int N, int H, int W, int C, int K, uint8_t* A, uint8_t* B, uint16_t* bias, uint8_t* workspace, cudaStream_t stream) {
            DMAHostDescriptors descs;

            uint32_t tensor_size_a[5];
            tensor_size_a[0] = INTERLEAVED ? W : C;
            tensor_size_a[1] = INTERLEAVED ? H : W;
            tensor_size_a[2] = INTERLEAVED ? 1 : H;
            tensor_size_a[3] = INTERLEAVED ? C/32 : 1;
            tensor_size_a[4] = N;

            uint64_t tensor_stride_a[4];
            tensor_stride_a[0] = INTERLEAVED ? W*32 : C;        
            tensor_stride_a[1] = INTERLEAVED ? H*W*32 : C*W;
            tensor_stride_a[2] = INTERLEAVED ? H*W*32 : C*H*W;
            tensor_stride_a[3] = INTERLEAVED ? (C/32)*H*W*32 : C*H*W;

            uint32_t traversal_stride_a[5] = {1,1,1,1,1};

            traversal_stride_a[3] = 1; // D
            traversal_stride_a[2] = 1; // H
            traversal_stride_a[1] = 1; // W

            uint32_t box_size_ndhw_a = TILE_M;
            uint32_t box_size_rangec_a = INTERLEAVED ? 1 : TILE_K;

            int32_t base_corner_a[3], far_corner_a[3];
            base_corner_a[2] = 0; // -pad_d
            base_corner_a[1] = 0; // -pad_h
            base_corner_a[0] = 0; // -pad_w
            far_corner_a[2] = 0; // pad_d - t*dilation
            far_corner_a[1] = 0; // pad_h - r*dilation
            far_corner_a[0] = 0; // pad_w - s*dilation

            gpuErrChk(cudaSetTmaIm2ColDescriptor(&descs.a_desc,
                                                 A,
                                                 5,
                                                 U8,
                                                 INTERLEAVED ? INTERLEAVE_32B : INTERLEAVE_DISABLED,
                                                 INTERLEAVED ? SWIZZLE_32B : SWIZZLE_128B,
                                                 PROMOTION_DISABLED,
                                                 tensor_size_a,
                                                 tensor_stride_a,
                                                 traversal_stride_a,
                                                 box_size_rangec_a,
                                                 box_size_ndhw_a,
                                                 base_corner_a,
                                                 far_corner_a,
                                                 0,
                                                 0));

            void* initial_workspace = reinterpret_cast<void*>(workspace);
            auto assignAligned = [&](cudaTmaDesc** dst, uint8_t** workspace) {
                const auto alignSize = cudaTmaDescAlignment;
                uint64_t newPtr = reinterpret_cast<std::uintptr_t>(*workspace);
                const auto remainder = newPtr % alignSize;
                if (remainder != 0) {
                    newPtr += alignSize - remainder;
                }

                *workspace = reinterpret_cast<uint8_t*>(newPtr);
                *dst = reinterpret_cast<cudaTmaDesc*>(*workspace);
                *workspace += sizeof(cudaTmaDesc);
            };

            assignAligned(&_params.a_desc, &workspace);

            uint32_t tensor_size_b[5];
            tensor_size_b[0] = C;
            tensor_size_b[1] = 1; // s
            tensor_size_b[2] = 1; // r
            tensor_size_b[3] = 1; // t
            tensor_size_b[4] = K;

            uint64_t tensor_stride_b[4];
            tensor_stride_b[0] = tensor_size_b[0];
            tensor_stride_b[1] = tensor_size_b[1] * tensor_stride_b[0];
            tensor_stride_b[2] = tensor_size_b[2] * tensor_stride_b[1];
            tensor_stride_b[3] = tensor_size_b[3] * tensor_stride_b[2];

            uint32_t traversal_stride_b[5] = {1,1,1,1,1};

            uint32_t box_size_b[5] = {1,1,1,1,1};
            box_size_b[0] = TILE_K;
            box_size_b[4] = TILE_N;

            for (int i=0; i<8; i++) descs.b_desc.data[i] = 0;
            gpuErrChk(cudaSetTmaTileDescriptor(&descs.b_desc,
                                                 B,
                                                 5,
                                                 U8,
                                                 INTERLEAVE_DISABLED,
                                                 SWIZZLE_128B,
                                                 PROMOTION_DISABLED,
                                                 tensor_size_b,
                                                 tensor_stride_b,
                                                 traversal_stride_b,
                                                 box_size_b,
                                                 0,
                                                 0));

            assignAligned(&_params.b_desc, &workspace);

            uint32_t tensor_size_bias[2];
            tensor_size_bias[0] = K;
            tensor_size_bias[1] = 1;

            uint64_t tensor_stride_bias[1];
            tensor_stride_bias[0] = tensor_size_bias[0]*sizeof(uint16_t);

            uint32_t traversal_stride_bias[2] = {1, 1};

            uint32_t box_size_bias[2] = {TILE_N, 1};

            for (int i=0; i<8; i++) descs.bias_desc.data[i] = 0;
            gpuErrChk(cudaSetTmaTileDescriptor(&descs.bias_desc,
                                                 bias,
                                                 2,
                                                 BF16_RN,
                                                 INTERLEAVE_DISABLED,
                                                 SWIZZLE_DISABLED,
                                                 PROMOTION_DISABLED,
                                                 tensor_size_bias,
                                                 tensor_stride_bias,
                                                 traversal_stride_bias,
                                                 box_size_bias,
                                                 0,
                                                 0));

            assignAligned(&_params.bias_desc, &workspace);

            if (workspace - reinterpret_cast<uint8_t*>(initial_workspace) != sizeof(DMAHostDescriptors))
            {
                throw std::runtime_error("Addresses were not aligned and the single memcpy of TMA descriptors is not possible");
            }
            gpuErrChk(cudaMemcpyAsync(initial_workspace, &descs, sizeof(DMAHostDescriptors), cudaMemcpyHostToDevice, stream));
        }

        __host__ DMAParams params() { return _params; }
    };
};
