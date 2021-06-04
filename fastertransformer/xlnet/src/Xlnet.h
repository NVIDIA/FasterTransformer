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

#include "XlnetLayer.h"
#include <vector>

template <typename T>
class Xlnet{
    private:
        //Load data
        InputDataDevice input_data_device; 
        PreWeightDevice<T> pre_weight_device;

        std::string gemm_file_name;

        //Configure block
        void blockAttnMask(dim3 &grid, dim3& block);

    protected:
        //Environment
        cudaStream_t stream;
        cublasHandle_t cublas_handle;

        //Metadata 
        int num_layers;
        int batch_size;
        int seq_len;

        int head_num;
        int size_per_head;
        int hidden_dim;

        int hidden_dim_ff;//the hidden size in feed-forward layers.
        float epsilon;

        //Preprocess data
        T* word_emb_k;
        T* attn_mask;
        T* seg_mat;
        T* attr_k_head_r;


        //Layers 
        std::vector< XlnetLayer<T> > arr_xlnet_layer;



    public:
        Xlnet(cudaStream_t stream, cublasHandle_t cublas_handle,
        int num_layers, int batch_size, int seq_len, int head_num, int size_per_head, 
        int hidden_dim,int hidden_dim_ff,int num_token, float epsilon,
        PreWeightHost<T> & pre_weight_host,
        std::vector<LayerWeightHost<T> >& arr_layer_weight_host,
        std::string gemm_file_name);

        void setInput(InputDataHost& input_data_host);
        void preProcess();
        void runAttentionLayers();

        void run(InputDataHost& input_data_host);

        ~Xlnet();
};



