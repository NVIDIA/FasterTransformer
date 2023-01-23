#include "fp8_gemm_1x1.h"

template <int TILE_M, int TILE_N, bool PERSISTENT_WEIGHTS, int CTAS_PER_CGA>
struct DynamicScheduler {

    struct Host {
        dim3   _grid;
        dim3   _cga;
        SchedulerParams _params;
        Host(uint8_t** workspace, cudaStream_t stream) {
            cudaDeviceProp prop;
            gpuErrChk(cudaGetDeviceProperties(&prop,0));

            _grid = dim3(prop.multiProcessorCount,1,1);
            _cga = dim3(2,1,1);

            if (PERSISTENT_WEIGHTS)
            {
                throw std::runtime_error("Persistent weights are currently not supported by the interface");
            }

            int num_tile_counters = 1;

            _params.tile_counter = reinterpret_cast<int*>(*workspace);
            *workspace += num_tile_counters * sizeof(int);
            _params.cta_completion_counter = reinterpret_cast<int*>(*workspace);
            *workspace += sizeof(int);
        }

        __host__ void configure(int N, int P, int Q, int K)
        {            
            _params.tiles_n = (K+TILE_N-1)/TILE_N;
            _params.tiles_n /= CTAS_PER_CGA;

            _params.tiles_m = (N*P*Q+TILE_M-1)/TILE_M;
            _params.num_tiles = _params.tiles_m*_params.tiles_n;
        }

        static __host__ uint32_t getWorkSpaceSize(int K)
        {
            if (PERSISTENT_WEIGHTS)
            {
                throw std::runtime_error("Persistent weights are currently not supported by the interface");
            }

            int tiles_n = (K+TILE_N-1)/TILE_N;
            tiles_n /= CTAS_PER_CGA;
            int num_tile_counters = PERSISTENT_WEIGHTS ? tiles_n : 1;
            return num_tile_counters * sizeof(int) + sizeof(int);
        }

        __host__ dim3 grid() { return _grid; }
        __host__ dim3 cga() { return _cga; }
        __host__ SchedulerParams params() { return _params; }
    };
};

