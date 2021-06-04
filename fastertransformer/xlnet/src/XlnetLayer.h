/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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



#pragma once

#include "utils.h"
#include "LoadData.h"
#include "layerKernels.h"

template <typename T>
class XlnetDebug;

template <typename T>
class XlnetLayer{
    private:
        //metadata
        int batch_size;
        int seq_len;
        int head_num;
        int size_per_head;

        //Calculated params
        int hidden_dim;
        int buf_size;
        int qk_buf_size; 
        int hidden_dim_ff;//the hidden size in feed-forward layers.

        float epsilon;

        //device environment
        int gpu_id;
        cublasHandle_t cublas_handle;
        cudaStream_t stream;

        T alpha;
        T beta;

        //parameters for cublas
        int cublas_algo[NUM_CUBLAS_FUNC];
        int cublas_func[NUM_CUBLAS_FUNC];
        int start_algo;
        int end_algo;
        cudaDataType_t a_type;
        cudaDataType_t b_type;
        cudaDataType_t c_type;
        cudaDataType_t compute_type;
        std::string dir;
        std::string gemm_file;
        int ifCheck;

        //Params
        //LoadDataLayer<T>* ptr_data_layers;
        LayerWeightDevice<T>  layer_weight_device;
    

        void allocDeviceMem();
        void setCublas();

        float profileCublasGemmEx(cublasOperation_t transa, cublasOperation_t transb,
                int v_m, int v_n, int v_k,int lda,int ldb, int ldc,int ites, int index);
        float profileCublasGemmStride(cublasOperation_t transa, cublasOperation_t transb,
                int v_m, int v_n, int v_k,int lda, int strideA,
                int ldb,int strideB, int ldc,int strideC, int batch,
                int ites, int index);

        void profileCublasAlgo();
        void recordCublasGemm();
        void setCublasAlgo();
        void copyCublasAlgo(const int* cublas_algo, const int* cublas_func);

        void oneToManyCublasGemm(T * d_A, T* d_B, T* d_C,cublasOperation_t transa, cublasOperation_t transb,
                int v_m, int v_n, int v_k,int lda, int strideA,
                int ldb,int strideB, int ldc,int strideC, int batch,int algo, cublasFunction method);

        void blockRelShiftBd(dim3 &grid, dim3& block);
        void invokeRelShiftBd();
        void invokeTranspose102();
        void invokeTranspose201();
        void invokePrepareMatrixes();
        void invokeCalAttnScore(T* attn_mask);
        void invokeTranspose102v2();
        void invokeLayerNorm();
        void invokeGelu();
        void invokeLayerNormv2();

    protected:
        //Attention data 
        T* to_tensor;
        T* qkv_buf;
        T* query_buf;
        T* key_buf;
        T* value_buf;
        T* q_buf;
        T* k_buf;
        T* qk_buf;

        T* k_head_r;

        T* q_buf_bd;
        T* k_buf_bd;
        T* qk_buf_bd;
        T* qk_buf_bd_shift;

        T* q_buf_ef;
        T* k_buf_ef;
        T* qk_buf_ef;
        T* qk_buf_ef_trans;
        T* qk_buf_ef_seg;
        T* qk_buf_ef_seg_trans;

        T* attn_score;
        T* value_buf_trans;

        T* attn_vec;
        T* attn_vec_trans;

        T* attn_out;

        T* attn_layernorm;

        T* output_fc1;
        T* output_fc2;
        T* output_layernorm;

    public:
        XlnetLayer(int batch_size, int seq_len, 
                int head_num, int size_per_head,int hidden_dim, int hidden_dim_ff,float epsilon,
                cudaStream_t stream, cublasHandle_t cublas_handle,
                std::string gemm_file,std::string dir="./",int ifCheck=0);


        XlnetLayer(XlnetLayer<T> const& xlnet_layer);

        void setLayerWeight(LayerWeightHost<T> & layer_weight_host);

        T* forward(T* to_tensor,T* attn_mask,T* seg_mat,T* attr_k_head_r);

        ~XlnetLayer();

        friend class XlnetDebug<T>;

};
