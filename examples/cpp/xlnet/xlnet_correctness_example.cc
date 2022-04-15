/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "cnpy.h"
#include "src/fastertransformer/models/xlnet/Xlnet.h"
using namespace fastertransformer;
using namespace std;
#include <iomanip>

template<typename T>
int xlnetCorrectnessExample(size_t batch_size,
                            size_t num_layers,
                            size_t seq_len,
                            size_t head_num,
                            size_t size_per_head,
                            size_t num_token,
                            string input_name,
                            string model_name,
                            string check_name,
                            bool allow_gemm_test = false);

/*************** NPZ related operations *****************/
template<typename T>
void printDevPtr(const T* d_cache, int len, char* name, bool print)
{
    T* res = (T*)malloc(sizeof(T) * len);
    cudaMemcpy(res, d_cache, sizeof(T) * len, cudaMemcpyDeviceToHost);

    printf("%s ", name);
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (res[i]) {
            printf("%f ", (float)res[i]);
            if (j % 20 == 19) {
                printf("\n");
            }
        }
        j = j + 1;
    }
    free(res);
    printf("\n");
}

template<typename T>
float castToFloat(T input)
{
    float output = (T)(input);
    return output;
}

template<>
float castToFloat(__half input)
{
    float output = __half2float(input);
    return output;
}

template<typename T>
void setByNpz(cnpy::npz_t& my_npz, std::string name, T* d_ptr, int size, int offset = 0)
{
    // printKey(my_npz);

    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];
    // load it into a new array
    T* loaded_data = arr.data<T>();
    check_cuda_error(cudaMemcpy(d_ptr, loaded_data + offset, sizeof(T) * size, cudaMemcpyHostToDevice));
}
template<>
void setByNpz<__half>(cnpy::npz_t& my_npz, std::string name, __half* d_ptr, int size, int offset)
{
    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    // load it into a new array
    float* loaded_data = arr.data<float>();
    __half* half_data = (__half*)malloc(sizeof(__half) * size);

    loaded_data = loaded_data + offset;
    for (int i = 0; i < size; i++) {
        half_data[i] = __float2half_rn(loaded_data[i]);
    }

    check_cuda_error(cudaMemcpy(d_ptr, half_data, sizeof(__half) * size, cudaMemcpyHostToDevice));
    free(half_data);
}

std::string paraName(int i_layer, std::string sub_para)
{
    std::ostringstream s;
    s << "model/transformer/layer_" << i_layer << sub_para;
    std::string str = s.str();
    return str;
}

std::string paraName(std::string s)
{
    std::string str = s;
    return str;
}

template<typename T>
void checkByNpz(cnpy::npz_t& data_npz, cudaStream_t stream, std::string name, T* d_ptr, int size)
{
    std::cout << name << " " << size << std::endl;
    bool ifCorrect = 1;
    cnpy::NpyArray arr = data_npz[name];
    T* loaded_data = arr.data<T>();

    T* h_ptr = (T*)malloc(size * sizeof(T));
    check_cuda_error(cudaMemcpyAsync(h_ptr, d_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));

    float err = 0;
    float max = castToFloat(h_ptr[0]);
    int i = 0;

    for (i = 0; i < size; i++) {
        float sub = abs(castToFloat(h_ptr[i]) - castToFloat(loaded_data[i]));
        if (sub > err) {
            err = sub;
        }
        if (max < castToFloat(h_ptr[i])) {
            max = castToFloat(h_ptr[i]);
        }
    }

    std::cout << name << " Max err :" << err << " Max value :" << max << " Ralative error rate: " << err / max
              << std::endl;

    free(h_ptr);
}

template<typename T>
void printNpz(cnpy::npz_t& my_npz, std::string name, int size, int offset = 0)
{
    // check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];
    // load it into a new array
    T* loaded_data = arr.data<T>();
    for (int i = 0; i < size; i++) {
        cout << loaded_data[i] << " ";
        if (i % 9 == 0) {
            cout << endl;
        }
    }
}
/*************** Main Program *****************/
int main(int argc, char** argv)
{
    string input_name = "./data/data.npz";
    string model_name = "./data/model.npz";
    string check_name = "./data/output.npz";

    if (argc != 11) {
        printf("[ERROR] ./bin/xlnet_correctness_example batch_size num_layers seq_len "
               "head_num size_per_head num_token input_name model_name check_name "
               "is_fp16\n");
        printf("e.g., ./bin/xlnet_correctness_example 8 12 128 12 64 32000 "
               "./data/data.npz ./data/model.npz ./data/output.npz 0\n");
        return 0;
    }
    bool allow_gemm_test = false;

    int batch_size = atoi(argv[1]);
    int num_layers = atoi(argv[2]);
    int seq_len = atoi(argv[3]);
    int head_num = atoi(argv[4]);
    int size_per_head = atoi(argv[5]);
    int num_token = atoi(argv[6]);
    input_name = argv[7];
    model_name = argv[8];
    check_name = argv[9];
    bool is_fp16 = atoi(argv[10]);

    cout << " " << batch_size << " " << num_layers << " " << seq_len << " " << head_num << " " << size_per_head << " "
         << num_token << " " << input_name << " " << model_name << " " << check_name << " " << is_fp16 << endl;

    if (is_fp16 == 0) {
        return xlnetCorrectnessExample<float>(batch_size,
                                              num_layers,
                                              seq_len,
                                              head_num,
                                              size_per_head,
                                              num_token,
                                              input_name,
                                              model_name,
                                              check_name,
                                              allow_gemm_test);
    }
    else if (is_fp16 == 1) {
        return xlnetCorrectnessExample<half>(batch_size,
                                             num_layers,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             num_token,
                                             input_name,
                                             model_name,
                                             check_name,
                                             allow_gemm_test);
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] is_fp16 should be 0 (use float)"
                                             "or 1 (use half). \n "));
    }
}

/*************** Correctness Check*****************/
template<typename T>
int xlnetCorrectnessExample(size_t batch_size,
                            size_t num_layers,
                            size_t seq_len,
                            size_t head_num,
                            size_t size_per_head,
                            size_t num_token,
                            string input_name,
                            string model_name,
                            string check_name,
                            bool allow_gemm_test)
{
    printf("[INFO] Device: %s \n", getDeviceName().c_str());

    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);

    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in", "");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();

    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    // Set layer weight
    std::vector<XlnetLayerWeight<T>> xlnet_layer_weights(num_layers, XlnetLayerWeight<T>(hidden_units, inter_size));
    const int weight_nums = 17;
    string weight_name[17] = {"/rel_attn/q/kernel:0",
                              "/rel_attn/k/kernel:0",
                              "/rel_attn/v/kernel:0",
                              "/rel_attn/r/kernel:0",
                              "model/transformer/r_w_bias:0",
                              "model/transformer/r_r_bias:0",
                              "model/transformer/r_s_bias:0",
                              "model/transformer/seg_embed:0",
                              "/rel_attn/o/kernel:0",
                              "/rel_attn/LayerNorm/gamma:0",
                              "/rel_attn/LayerNorm/beta:0",
                              "/ff/layer_1/kernel:0",
                              "/ff/layer_1/bias:0",
                              "/ff/layer_2/kernel:0",
                              "/ff/layer_2/bias:0",
                              "/ff/LayerNorm/gamma:0",
                              "/ff/LayerNorm/beta:0"};

    cnpy::npz_t model_npz = cnpy::npz_load(model_name);

    for (int i = 0; i < num_layers; i++) {
        T** weight_ptrs = xlnet_layer_weights[i].getWeightPtrs();
        int* weight_sizes = xlnet_layer_weights[i].getWeightSizes();
        for (int j = 0; j < weight_nums; j++) {
            string str;
            if (j < 3) {
                str = paraName(i, weight_name[j]);
                cout << str << endl;
                setByNpz(model_npz, str, weight_ptrs[0] + j * hidden_units * hidden_units, hidden_units * hidden_units);
            }
            else {
                if (j == 3 || j >= 8) {
                    str = paraName(i, weight_name[j]);
                    cout << str << endl;
                    setByNpz(model_npz, str, weight_ptrs[j - 2], weight_sizes[j - 2]);
                }
                else {
                    str = paraName(weight_name[j]);
                    cout << str << endl;
                    setByNpz(model_npz, str, weight_ptrs[j - 2], weight_sizes[j - 2], i * weight_sizes[j - 2]);
                }

            }  // end for else
        }      // end for j
    }          // end for i

    // Allocate Input & Output
    float* input_mask;
    deviceMalloc(&input_mask, batch_size * seq_len, false);
    int* seg_id;
    deviceMalloc(&seg_id, batch_size * seq_len, false);

    int* inp_k;
    deviceMalloc(&inp_k, batch_size * seq_len, false);
    T* params_word_emb_k;
    deviceMalloc(&params_word_emb_k, num_token * hidden_units, false);
    T* word_emb_k;
    deviceMalloc(&word_emb_k, batch_size * seq_len * hidden_units, false);

    T* out_tensor;
    deviceMalloc(&out_tensor, batch_size * seq_len * hidden_units, false);

    cnpy::npz_t input_npz = cnpy::npz_load(input_name);
    cnpy::npz_t check_npz = cnpy::npz_load(check_name);
    setByNpz(input_npz, "input_mask:0", input_mask, batch_size * seq_len);
    setByNpz(input_npz, "segment_ids:0", seg_id, batch_size * seq_len);
    setByNpz(input_npz, "input_ids:0", inp_k, batch_size * seq_len);
    setByNpz(model_npz, "model/transformer/word_embedding/lookup_table:0", params_word_emb_k, num_token * hidden_units);

    genWordEmdK(batch_size, seq_len, hidden_units, word_emb_k, params_word_emb_k, inp_k, stream);
    checkByNpz(check_npz, stream, "output_h", word_emb_k, batch_size * seq_len * hidden_units);

    // Prepare for the inputs and outputs as vector
    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, word_emb_k},
        Tensor{MEMORY_GPU, getTensorType<float>(), std::vector<size_t>{batch_size, seq_len}, input_mask},
        Tensor{MEMORY_GPU, getTensorType<int>(), std::vector<size_t>{batch_size, seq_len}, seg_id}};
    std::vector<Tensor> output_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU, getTensorType<T>(), std::vector<size_t>{batch_size, seq_len, hidden_units}, out_tensor}};
    Xlnet<T> xlnet = Xlnet<T>(batch_size,
                              seq_len,
                              head_num,
                              size_per_head,
                              inter_size,
                              num_layers,
                              1.0f,
                              stream,
                              &cublas_wrapper,
                              &allocator,
                              false);

    xlnet.forward(&output_tensors, &input_tensors, &xlnet_layer_weights);

    // Check result
    std::ostringstream s;
    s << "layer_" << (num_layers - 1);
    std::string label = s.str();
    checkByNpz(check_npz, stream, label, out_tensor, batch_size * seq_len * hidden_units);

    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    cudaFree(inp_k);
    cudaFree(params_word_emb_k);
    cudaFree(word_emb_k);

    cudaFree(input_mask);
    cudaFree(seg_id);

    cudaFree(out_tensor);
    return 0;
}
