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

#include "Xlnet.h"

template<typename T>
class XlnetDebug : public Xlnet<T>{
    private:
        int buf_size;
        int qk_buf_size;
    public:
        XlnetDebug(cudaStream_t stream, cublasHandle_t cublas_handle,
                int num_layers, int batch_size, int seq_len, int head_num, int size_per_head, 
                int hidden_dim,int hidden_dim_ff,int num_token, float epsilon,
                PreWeightHost<T> & pre_weight_host,
                std::vector<LayerWeightHost<T> >& arr_layer_weight_host,
                std::string gemm_file_name);

        bool verifyPreProcess(cnpy::npz_t& data_npz);
        bool verifyInter(cnpy::npz_t& data_npz,int i_layer);
        bool verifyLayerRes(InputDataHost& input_data_host, std::string output_file);

        float profileOneLayer(int warm_up_time, int profile_run_time);

        ~XlnetDebug();
        friend class XlnetLayer<T>;
};


