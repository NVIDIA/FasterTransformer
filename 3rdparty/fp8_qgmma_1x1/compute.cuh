#include "fp8_gemm_1x1.h"

struct Compute {
    struct Host {
        ComputeParams _params;
        __host__ Host() {}

        __host__ ComputeParams params() { return _params; }

        __host__ void configure(uint8_t* D, int N, int P, int Q, int C, int K, float ab_scale, float d_scale)
        {
            _params.D = D;
            _params.N = N;
            _params.NPQ = N*P*Q;
            _params.PQ = P*Q;
            _params.P = P;
            _params.Q = Q;
            _params.C = C;
            _params.K = K;
            _params.ab_scale = ab_scale;
            _params.d_scale = d_scale;
        }
    };
};
