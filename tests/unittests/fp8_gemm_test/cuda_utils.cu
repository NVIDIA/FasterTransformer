#include "cuda_utils.h"

template<typename T_OUT, typename T_IN>
void __global__ cudaCastKernel(T_OUT* out, T_IN const * const in, const uint32_t size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i+= gridDim.x * blockDim.x) {
        out[i] = (T_OUT)((T_IN)(in[i]));
    }
}


template<typename T_OUT, typename T_IN>
void cudaCast(T_OUT* out, T_IN const * const in, const uint32_t size)
{
    dim3 block(256);
    dim3 grid((size + 1) / 256 * 256);
    cudaCastKernel<<<grid, block>>>(out, in, size);
}

template void cudaCast(float* out, float const * const in, const uint32_t size);
template void cudaCast(__nv_fp8_e4m3* out, float const * const in, const uint32_t size);



template<typename T, int QUANTIZE_MODE>
__global__ void
quatizeVectorE4M3(__nv_fp8_e4m3* output, float const* input_qua_amax_ptr, T const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (QUANTIZE_MODE == 0) {
            output[i] = __nv_fp8_e4m3((float)(input[i]) / __ldg(input_qua_amax_ptr + (i % n)));
        }
        else {
            output[i] = __nv_fp8_e4m3((float)(input[i]) / __ldg(input_qua_amax_ptr));
        }
    }
}

template<typename T, int QUANTIZE_MODE>
void invokeQuatizeVectorE4M3(__nv_fp8_e4m3* output,
                             float const* input_qua_amax_ptr,
                             T const* input,
                             uint32_t size,
                             uint32_t n,
                             cudaStream_t stream)
{
    dim3 grid(1);
    dim3 block(256);
    quatizeVectorE4M3<T, QUANTIZE_MODE><<<grid, block, 0, stream>>>(output, input_qua_amax_ptr, input, size, n);
}

template void invokeQuatizeVectorE4M3<float, 0>(__nv_fp8_e4m3* output,
                                                float const* input_qua_amax_ptr,
                                                float const* input,
                                                uint32_t size,
                                                uint32_t n,
                                                cudaStream_t stream);
template void invokeQuatizeVectorE4M3<half, 0>(__nv_fp8_e4m3* output,
                                               float const* input_qua_amax_ptr,
                                               half const* input,
                                               uint32_t size,
                                               uint32_t n,
                                               cudaStream_t stream);
template void invokeQuatizeVectorE4M3<float, 1>(__nv_fp8_e4m3* output,
                                                float const* input_qua_amax_ptr,
                                                float const* input,
                                                uint32_t size,
                                                uint32_t n,
                                                cudaStream_t stream);
template void invokeQuatizeVectorE4M3<half, 1>(__nv_fp8_e4m3* output,
                                               float const* input_qua_amax_ptr,
                                               half const* input,
                                               uint32_t size,
                                               uint32_t n,
                                               cudaStream_t stream);


template<typename T_OUT, typename T_IN, int QUANTIZE_MODE>
__global__ void
dequatizeVectorE4M3(T_OUT* output, float const* qua_amax_ptr, T_IN const* input, uint32_t size, uint32_t n)
{
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        if (QUANTIZE_MODE == 0) {
            output[i] = T_OUT(float(input[i]) * __ldg(qua_amax_ptr + (i % n)));
        }
        else {
            output[i] = T_OUT(float(input[i]) * __ldg(qua_amax_ptr));
        }
    }
}

template<typename T_OUT, typename T_IN, int QUANTIZE_MODE>
void invokeDequatizeVectorE4M3(
    T_OUT* output, float const* qua_amax_ptr, T_IN const* input, uint32_t size, uint32_t n, cudaStream_t stream)
{
    dim3 grid(1);
    dim3 block(256);
    dequatizeVectorE4M3<T_OUT, T_IN, QUANTIZE_MODE><<<grid, block, 0, stream>>>(output, qua_amax_ptr, input, size, n);
}

template void invokeDequatizeVectorE4M3<float, __nv_fp8_e4m3, 0>(float* output,
                                                  float const* qua_amax_ptr,
                                                  __nv_fp8_e4m3 const* input,
                                                  uint32_t size,
                                                  uint32_t n,
                                                  cudaStream_t stream);
template void invokeDequatizeVectorE4M3<half, __nv_fp8_e4m3, 0>(half* output,
                                                 float const* qua_amax_ptr,
                                                 __nv_fp8_e4m3 const* input,
                                                 uint32_t size,
                                                 uint32_t n,
                                                 cudaStream_t stream);
template void invokeDequatizeVectorE4M3<float, __nv_fp8_e4m3, 1>(float* output,
                                                  float const* qua_amax_ptr,
                                                  __nv_fp8_e4m3 const* input,
                                                  uint32_t size,
                                                  uint32_t n,
                                                  cudaStream_t stream);
template void invokeDequatizeVectorE4M3<half, __nv_fp8_e4m3, 1>(half* output,
                                                 float const* qua_amax_ptr,
                                                 __nv_fp8_e4m3 const* input,
                                                 uint32_t size,
                                                 uint32_t n,
                                                 cudaStream_t stream);
template void invokeDequatizeVectorE4M3<__nv_bfloat16, __nv_bfloat16, 0>(__nv_bfloat16* output,
                                                 float const* qua_amax_ptr,
                                                 __nv_bfloat16 const* input,
                                                 uint32_t size,
                                                 uint32_t n,
                                                 cudaStream_t stream);