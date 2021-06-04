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


#include "Xlnet.h"

#define DATA_TYPE float
void testDataClasses(cudaStream_t stream,
        int num_layers, int batch_size, int seq_len, 
        int head_num, int size_per_head, int hidden_dim_ff, int num_token,
        std::string input_file,std::string para_file){

    int hidden_dim=head_num*size_per_head;

    InputDataHost input_data_host(batch_size,seq_len);
    input_data_host.fillInputData(input_file);

    InputDataDevice input_data_device(stream,batch_size,seq_len);
    input_data_device.copyFromHost(input_data_host);

    PreWeightHost<DATA_TYPE> pre_weight_host(hidden_dim,num_token);
    pre_weight_host.fillPreWeight(para_file);

    PreWeightDevice<DATA_TYPE> pre_weight_device(stream,hidden_dim,num_token);
    pre_weight_device.copyFromHost(pre_weight_host);

    std::vector<LayerWeightHost<DATA_TYPE> > arr_layer_weight_host(num_layers, 
            LayerWeightHost<DATA_TYPE>(hidden_dim,hidden_dim_ff));

    std::vector<LayerWeightDevice<DATA_TYPE> > arr_layer_weight_device(num_layers, 
            LayerWeightDevice<DATA_TYPE>(stream,hidden_dim,hidden_dim_ff));

}
int main(int argc, char* argv[]) {
    // Set metadata
    int batch_size =8;
    int num_layers =12;
    int seq_len =128;
    int head_num =12;
    int size_per_head =64;
    int hidden_dim=head_num*size_per_head;
    int hidden_dim_ff = 3072;
    int num_token=32000;

    int gpu_id=0;
    float epsilon =0.0f;  //__half epsilon =0.0;  

    //Prepare device environment
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("Using Device %s\n", prop.name);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);


    //Prepare Data Files
    std::string input_file="../Data/data.npz";
    std::string para_file="../Data/model.npz";
    std::string gemm_file="./gemm/gemm.fp32.1080ti";


    //Load Data
    InputDataHost input_data_host(batch_size,seq_len);
    input_data_host.fillInputData(input_file);

    PreWeightHost<DATA_TYPE> pre_weight_host(hidden_dim,num_token);
    pre_weight_host.fillPreWeight(para_file);

    std::vector<LayerWeightHost<DATA_TYPE> > arr_layer_weight_host(num_layers, 
            LayerWeightHost<DATA_TYPE>(hidden_dim,hidden_dim_ff));

    for(int i=0;i<num_layers;i++){
        std::cout<<"Run layer "<<i<<std::endl;
        arr_layer_weight_host[i].fillLayerWeight(i,para_file);
    }

    Xlnet<DATA_TYPE> xlnet(stream,cublas_handle,num_layers,batch_size,seq_len,
        head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token,epsilon,
        pre_weight_host,arr_layer_weight_host,gemm_file);


    xlnet.run(input_data_host);

    std::cout<<"END"<<std::endl;
    return 0;
}
