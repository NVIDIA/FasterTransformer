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


#include "XlnetDebug.h"

#define DATA_TYPE float

//#define DATA_TYPE __half
namespace std {
  template <typename _CharT, typename _Traits>
  inline basic_ostream<_CharT, _Traits> &
  tab(basic_ostream<_CharT, _Traits> &__os) {
    return __os.put(__os.widen('\t'));
  }
}

void print_usage(){
    //b:l:m:e:g:j:i:p:r:w:o:h
    std::cout<<"runTest -b batch_size -n num_layers -l seq_len  -e 0 -i input_file -p parameter_file -d dir" <<std::endl;
    std::cout<<"# Two important input parameters:"<<std::endl;
    std::cout<<std::tab<<"-b(--batch) : batch_size(INT).Batchsize for inferencei. Default:8"<<std::endl;
    std::cout<<std::tab<<"-s(--seq) : seq_len(INT). Length of input sequence. Default:128"<<std::endl;
    std::cout<<std::endl<<"# Model Parameters:"<<std::endl;
    std::cout<<std::tab<<"-q(--epsilon) : epsilon(FLOAT). Small float added to variance to avoid dividing by zero.Default:0"<<std::endl;
    std::cout<<std::tab<<"-g(--gpu) :  gpu id (INT). Id of GPU card  Default: 0"<<std::endl;
    std::cout<<std::tab<<"-e(--gemm) : gemm_file(STRING). Filename of the gemm function configuraton file."<<std::endl;
    std::cout<<std::tab<<"-j(--json) : json_file(STRING). Filename of the json configuraton file."<<std::endl;

    std::cout<<std::endl<<"# Run this code in different mode "<<std::endl;
    std::cout<<std::tab<<"-m(--mode) : mode(INT{0,1,2,3}). "<<std::endl;
    std::cout<<std::tab<<std::tab<<"0 (FP16_TIME_TEST): Run perf test in fp16 mode "<<std::endl;
    std::cout<<std::tab<<std::tab<<"1 (FP16_CORRECTNESS_TEST): Run correctness test in fp16 mode "<<std::endl;
    std::cout<<std::tab<<std::tab<<"2 (FP32_TIME_TEST): Run perf test in fp32 mode "<<std::endl;
    std::cout<<std::tab<<std::tab<<"3 (FP32_CORRECTNESS_TEST): Run correctness test in fp32 mode "<<std::endl;

    std::cout<<std::endl<<"## VERIFICATION mode:"<<std::endl;
    std::cout<<std::tab<<"-i(--input) : input_file(STRING). Filename of input."<<std::endl;
    std::cout<<std::tab<<"-p(--para) : parameter_file(STRING). Filename of input."<<std::endl;
    std::cout<<std::tab<<"-r(--result) : result_file(STRING). Filename of the result file generated from tensorflow."<<std::endl;

    std::cout<<std::endl<<"## TIMING mode:"<<std::endl;
    std::cout<<std::tab<<"-w(--warm) : warm_up_ite(INT). Warm up iterations. Default: 100"<<std::endl;
    std::cout<<std::tab<<"-t(--ite) : profile_ite(INT). Profile iterations. Default: 200"<<std::endl;
 
    std::cout<<"Example for correctness check: runTest -m 0 -b 8 -s 128 -g 0 -e gemm.in -j json.xml -i data.npz -p para.npz -r output.npz"<<std::endl;
    std::cout<<"Example for timing: runTest -m 1 -b 8 -g 0 -e gemm.in -j json.xml"<<std::endl;
}

struct option opts[] = {
        {"batch", 1, NULL, 'b'},
        {"seq", 1, NULL, 's'},
        {"epsilon", 1, NULL, 'q'},
        {"gpu", 1, NULL, 'g'},
        {"gemm", 1, NULL, 'e'},
        {"json", 1, NULL, 'j'},
        {"mode", 1, NULL, 'm'},
        {"input", 1, NULL, 'i'},
        {"para", 1, NULL, 'p'},
        {"result", 1, NULL, 'r'},
        {"warm", 1, NULL, 'w'},
        {"ite", 1, NULL, 't'},
        {"help", 0, NULL, 'h'}
};


template<typename T>
void layerRes(cudaStream_t stream, cublasHandle_t cublas_handle,
        int num_layers, int batch_size, int seq_len, 
        int head_num, int size_per_head, int hidden_dim,
        int hidden_dim_ff,int num_token,float epsilon, 
        std::string input_file,std::string para_file,
        std::string gemm_file, std::string output_file, bool use_float16){

    //Load Host Input Data
    InputDataHost input_data_host(batch_size,seq_len);
    input_data_host.fillInputData(input_file);

    //Load Host Weight Data
    PreWeightHost<T> pre_weight_host(hidden_dim,num_token);
    pre_weight_host.fillPreWeight(para_file);

    std::vector<LayerWeightHost<T> > arr_layer_weight_host(num_layers, 
            LayerWeightHost<T>(hidden_dim,hidden_dim_ff));

    for(int i=0;i<num_layers;i++){
        arr_layer_weight_host[i].fillLayerWeight(i,para_file);
    }

    //Construct Debug Class
    XlnetDebug<T> xlnet_debug(stream,cublas_handle,num_layers,batch_size,seq_len,
            head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token,epsilon,
            pre_weight_host,arr_layer_weight_host,gemm_file);

    //Run Debug Class
    bool ifCorrect=xlnet_debug.verifyLayerRes(input_data_host,output_file);
    if(ifCorrect==1){
        std::cout<<"Result Correct"<<std::endl;
    }else{
        std::cout<<"Result Wrong"<<std::endl;
    }
}


template<typename T>
void profile(cudaStream_t stream, cublasHandle_t cublas_handle,
        int num_layers, int batch_size, int seq_len, int head_num, int size_per_head, 
        int hidden_dim,int hidden_dim_ff,int num_token, float epsilon,
        std::string input_file,std::string para_file,
        std::string gemm_file,int warm_up_ite, int profile_ite){
    //Construct Debug Class
    InputDataHost input_data_host(batch_size,seq_len);
    PreWeightHost<T> pre_weight_host(hidden_dim, num_token);
    std::vector<LayerWeightHost<T> > arr_layer_weight_host(num_layers, 
            LayerWeightHost<T>(hidden_dim,hidden_dim_ff));

    XlnetDebug<T> xlnet_debug(stream,cublas_handle,num_layers,batch_size,seq_len,
            head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token, epsilon,
            pre_weight_host,arr_layer_weight_host,gemm_file);

    //Run Debug Class
    float run_time=xlnet_debug.profileOneLayer(warm_up_ite, profile_ite);
    std::cout<<"RUN_TIME: batch_size= "<<batch_size<<" seq_len= "<<seq_len<<" run_time= "<<run_time<<" MS"<<std::endl<<std::endl;
}

void  readJson(std::string json_file, int &d_head, int &d_inner, int &d_model, int& n_head, int &n_layer, int& n_token){
    std::string item_list[6]={"d_head", "d_inner","d_model", "n_head", "n_layer", "n_token"};
    std::ifstream json_stream;
    json_stream.open(json_file.c_str()); 
    
    
    std::string line;
    while(std::getline(json_stream, line))
    {
        int n=line.length();
        //std::cout<<line<<std::endl;
        char sentence[n+1];
        strcpy(sentence,line.c_str());

        char str[20]; 
        int value=0;
        sscanf(sentence, "%s %d,", str, &value);
        if(strstr(str, item_list[0].c_str())){
            d_head=value;
        }else if(strstr(str, item_list[1].c_str())){
            d_inner=value;
        }else if(strstr(str, item_list[2].c_str())){
            d_model=value;
        }else if(strstr(str, item_list[3].c_str())){
            n_head=value;
        }else if(strstr(str, item_list[4].c_str())){
            n_layer=value;
        }else if(strstr(str, item_list[5].c_str())){
            n_token=value;
        }
    }
}
int main(int argc, char* argv[]) {
    //Process input data
    int batch_size =0;
    int num_layers =12;
    int seq_len =128;
    int head_num =12;
    int num_token=32000;
    int size_per_head =64;
    int hidden_dim=head_num*size_per_head;
    int hidden_dim_ff = 3072;
    float epsilon =0.0f;  

    RUN_MODE mode=FP32_TIME_TEST; 
    int gpu_id=0;
    int warm_up_ite=100;
    int profile_ite=100;
    std::string input_file="";
    std::string para_file="";
    std::string gemm_file="./gemm.in";
    std::string result_file="";
    std::string json_file="";

    const char *opt_string = "b:s:q:m:e:g:j:i:p:r:w:t:h";

    int option=1;
    while((option = getopt_long(argc, argv,opt_string,opts,NULL)) != -1){
        switch (option) {
            case 'b' : 
                batch_size = atoi(optarg); 
                break;
            case 's' : 
                seq_len = atoi(optarg); 
                break;
            case 'q' : 
                epsilon = atof(optarg); 
                break;
            case 'g' : 
                gpu_id = atoi(optarg); 
                break;
            case 'm' : 
                mode =(RUN_MODE) atoi(optarg); 
                break;
            case 'e' : 
                gemm_file =optarg; 
                break;
            case 'j' : 
                json_file =optarg; 
                break;
            case 'i' : 
                input_file =optarg; 
                break;
            case 'p' : 
                para_file =optarg; 
                break;
            case 'r' : 
                result_file =optarg; 
                break;
            case 'w' : 
                warm_up_ite = atoi(optarg); 
                break;
            case 't' : 
                profile_ite = atoi(optarg); 
                break;
            case 'h' : 
                print_usage(); 
                exit(EXIT_FAILURE);
            default: 
                print_usage(); 
                exit(EXIT_FAILURE);
        }
    }

    if(batch_size==0){
        print_usage();
        exit(EXIT_FAILURE);
    }

    readJson(json_file, size_per_head,hidden_dim_ff, hidden_dim, head_num, num_layers, num_token);
    
    std::cout<<"Read Json file, got the meta parameters:"<<std::endl;
    std::cout<<"batch_size="<<batch_size<<", seq_len="<<seq_len<<std::endl;   
    std::cout<<"size_per_head(d_head)= "<<size_per_head<<std::endl;
    std::cout<<"hidden_dim_ff(d_inner)="<<hidden_dim_ff<<std::endl;
    std::cout<<"hidden_dim(d_model)="<<hidden_dim<<std::endl;
    std::cout<<"head_num(n_head)="<<head_num<<std::endl;
    std::cout<<"num_layers(n_layer)="<<num_layers<<std::endl;
    std::cout<<"num_token(n_token)="<<num_token<<std::endl<<std::endl;

    //Prepare device environment
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("Using Device %s\n\n", prop.name);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);

    switch(mode){
        case FP16_TIME_TEST:
            std::cout<<"Run in mode FP16_TIME_TEST: gemm_file="<<gemm_file<<std::endl;
            profile<__half>(stream,cublas_handle,
                num_layers,batch_size,seq_len,head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token,epsilon,
                input_file,para_file,gemm_file,warm_up_ite, profile_ite);
            break;
        case FP16_CORRECTNESS_TEST:
            std::cout<<"Run in mode FP16_CORRECTNESS_TEST: gemm_file="<<gemm_file<<std::endl;
            layerRes<__half>(stream,cublas_handle,
                num_layers,batch_size,seq_len,head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token, epsilon,
                input_file,para_file,gemm_file,result_file, true);
            break;
        case FP32_TIME_TEST:
            std::cout<<"Run in mode FP32_TIME_TEST: gemm_file="<<gemm_file<<std::endl;
            profile<float>(stream,cublas_handle,
                num_layers,batch_size,seq_len,head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token,epsilon,
                input_file,para_file,gemm_file,warm_up_ite, profile_ite);
            break;
        case FP32_CORRECTNESS_TEST:
            std::cout<<"Run in mode FP32_CORRECTNESS_TEST gemm_file="<<gemm_file<<std::endl;
            layerRes<float>(stream,cublas_handle,
                num_layers,batch_size,seq_len,head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token, epsilon,
                input_file,para_file,gemm_file,result_file,false);
            break;
    }
    return 0;

} 
