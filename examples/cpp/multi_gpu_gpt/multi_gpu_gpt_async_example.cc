/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

static bool USE_ASYNC = true;
const int START_TOKEN_ID = 50256;
const int END_TOKEN_ID = 50256;

#ifdef USE_NVTX
bool NVTX_ON = true;
#endif

using namespace fastertransformer;

namespace strutils {

template<typename T>
std::string join(const T* arr, const int length, const std::string sep = " ")
{
    std::string str = "";
    for (int i = 0; i < length; i++) {
        str += std::to_string(arr[i]);
        if (i < length - 1) {
            str += sep;
        }
    }
    return str;
}

template<typename T>
std::string toString(const T* arr, const int length, const bool is_device_array = true)
{
    size_t size = sizeof(T) * length;
    T* h_arr;
    std::string token_ids_str;
    if (is_device_array) {
        h_arr = (T*)malloc(size);
        cudaMemcpy(h_arr, arr, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        token_ids_str = join(h_arr, length, " ");
        free(h_arr);
    }
    else {
        token_ids_str = join(arr, length, " ");
    }
    return token_ids_str;
}

}  // namespace strutils

template<typename T>
class GptStreamer {

protected:
    // stremer settings
    const int max_batch_size;
    const int max_output_length;  // including both input and generated tokens.

    // results
    int* output_ids;
    int* sequence_lengths;
    bool* finished;

    // decoder settings
    ParallelGpt<T>* gpt_;
    std::unordered_map<std::string, Tensor>* output_tensors_;
    const std::unordered_map<std::string, Tensor>* input_tensors_;
    const ParallelGptWeight<T>* gpt_weights_;

    // streamer status and internal buffers
    int prev_step = 0;
    int curr_step = 0;
    bool is_generation_done = false;
    cudaStream_t stream;

    /**
     * \brief Stop criteria whether to stop generation
     *
     * This function determines whether to stop generation. If a user
     * wants to stop by a customized criteria, it is enough to override
     * this function that updates `finished` and `sequence_lengths`.
     *
     * \param step_from The generation step where we check from.
     * \param step_to The current generation step.
     *      The function will check step_from <= step < step_to.
     * \param output_ids An integer array pointer of a host in order to copy
     *      generated outputs from device.
     *
     * \return true if all generation have done. Otherwise, false.
     */
    virtual bool stopCriteria(const int step_from, const int step_to, const int* output_ids)
    {
        assert(step_from <= step_to);
        int batch_size = output_tensors_->at("output_ids").shape[0];
        for (int step = step_from; step < step_to; step++) {
            bool stop_generation = true;
            for (int i = 0; i < batch_size; i++) {
                if (!finished[i]) {
                    // Let 50256 be the end token id.
                    if (output_ids[step * batch_size + i] == END_TOKEN_ID) {
                        finished[i] = true;
                    }
                    else {
                        sequence_lengths[i] += 1;
                    }
                }
                if (!finished[i]) {
                    stop_generation = false;
                }
            }
            if (stop_generation) {
                return true;
            }
        }
        return false;
    }

    /**
     * \brief A hook function that performs during streaming
     *
     * This function performs a custom logic just after copying generated
     * tokens (`output_ids`) to the host.
     *
     * \param step_from The generation step where the function starts from.
     * \param step_to The generation step where the function ends at.
     *      The function deals the output_ids step_from <= step < step_to.
     * \param output_ids The output ids
     */
    virtual void streamHook(const int prev_step, const int curr_step, const int* output_ids) {}

    /**
     * \brief Stream preparation function
     *
     * This function calls just before streaming inside `streamDecoding`.
     * Please override this function to do anything, if a user further needs
     * inside an overriden function (`streamHook` or `stopCriteria`), e.g.
     * allocating buffers.
     */
    virtual void onStreamBegin(const int batch_size, const int input_len) {}

    /**
     * \brief Stream cooling down function
     *
     * This function calls just after streaming inside `streamDecoding`.
     * Please override this function to do anything needed, e.g.
     * free allocated buffers.
     */
    virtual void onStreamEnd() {}

    virtual void sendStopSignal()
    {
        // Update the finished buffer in decoding to let the GPT model know,
        // terminating the async thread.
        cudaMemcpyAsync(gpt_->getFinishBuffer(),
                        finished,
                        sizeof(bool) * output_tensors_->at("output_ids").shape[0]
                            * output_tensors_->at("output_ids").shape[1],
                        cudaMemcpyHostToDevice,
                        stream);
        cudaStreamSynchronize(stream);
    }

    void streamDecoding()
    {
        int input_len = input_tensors_->at("input_ids").shape[1];
        int max_output_len = output_tensors_->at("output_ids").shape[0];
        int batch_size = output_tensors_->at("output_ids").shape[0];

        // initialization
        is_generation_done = false;
        prev_step = 0;
        curr_step = input_len;
        int* seqlen_buf_ = new int[batch_size];
        bool* decoding_finished_buf_ = new bool[batch_size];
        bool* finished = new bool[batch_size];
        std::fill(seqlen_buf_, seqlen_buf_ + batch_size, input_len);
        std::fill(decoding_finished_buf_, decoding_finished_buf_ + batch_size, false);
        std::fill(finished, finished + batch_size, false);

        // start streaming
        onStreamBegin(batch_size, input_len);
        while (!(is_generation_done || curr_step == max_output_len)) {
            cudaMemcpyAsync(seqlen_buf_,
                            (int*)output_tensors_->at("sequence_length").data,
                            sizeof(int) * batch_size,
                            cudaMemcpyDeviceToHost,
                            stream);
            cudaStreamSynchronize(stream);
            curr_step = *std::max_element(seqlen_buf_, seqlen_buf_ + batch_size);
            if (prev_step < curr_step) {
                int idx_from = prev_step * batch_size;
                cudaMemcpyAsync(output_ids + idx_from,
                                (int*)(output_tensors_->at("output_ids").data) + idx_from,
                                sizeof(int) * (curr_step - prev_step) * batch_size,
                                cudaMemcpyDeviceToHost,
                                stream);
                cudaStreamSynchronize(stream);
                is_generation_done = stopCriteria(prev_step, curr_step, output_ids);
                streamHook(prev_step, curr_step, output_ids);
            }
            else {
                // The last token isn't accounted in the length of a sequence
                // since the end-token is not included in generated  length.
                // So when all the sample generation is done, prev_step  and
                // curr_step remain the same.
                cudaMemcpyAsync(decoding_finished_buf_,
                                gpt_->getFinishBuffer(),
                                sizeof(bool) * batch_size,
                                cudaMemcpyDeviceToHost,
                                stream);
                cudaStreamSynchronize(stream);
                is_generation_done =
                    std::all_of(decoding_finished_buf_, decoding_finished_buf_ + batch_size, [](bool b) { return b; });
            }

#ifndef NDEBUG
            if (prev_step < curr_step) {
                int batch_size = output_tensors_->at("output_ids").shape[0];
                std::cout << "\r[DEBUG] Step " << curr_step
                          << " | seqlen: " << strutils::toString(sequence_lengths, batch_size, false)
                          << " | output_ids: "
                          << strutils::toString(output_ids + (curr_step - 1) * batch_size, batch_size, false)
                          << " | h_finished_buf: " << strutils::toString(finished, batch_size, false)
                          << " | d_finished_buf: " << strutils::toString(gpt_->getFinishBuffer(), batch_size, true)
                          << std::flush;
            }
#endif
            prev_step = curr_step;
        }

        if (is_generation_done) {
            // The EOD token isn't accounted in the sequence length.
            // So, we need to copy the last step's output_ids
            // (all of them are the end tokens).
            int idx_from = curr_step * batch_size;
            cudaMemcpyAsync(output_ids + idx_from,
                            (int*)(output_tensors_->at(0).data) + idx_from,
                            sizeof(int) * batch_size,
                            cudaMemcpyDeviceToHost,
                            stream);
            cudaStreamSynchronize(stream);
            curr_step++;

            streamHook(prev_step, curr_step, output_ids);
            sendStopSignal();
        }
        onStreamEnd();

        delete[] seqlen_buf_;
        delete[] decoding_finished_buf_;
        delete[] finished;
    }

public:
    GptStreamer(int max_batch_size, int max_output_length):
        max_batch_size(max_batch_size), max_output_length(max_output_length)
    {
        output_ids = new int[max_output_length * max_batch_size];
        sequence_lengths = new int[max_batch_size];
        finished = new bool[max_batch_size];
        cudaStreamCreate(&stream);
    }

    ~GptStreamer()
    {
        delete[] output_ids;
        delete[] sequence_lengths;
        delete[] finished;
        cudaStreamDestroy(stream);
    }

    void initialize(ParallelGpt<T>* gpt,
                    std::unordered_map<std::string, Tensor>* output_tensors,
                    const std::unordered_map<std::string, Tensor>* input_tensors,
                    const ParallelGptWeight<T>* gpt_weights)
    {
        gpt_ = gpt;
        output_tensors_ = output_tensors;
        input_tensors_ = input_tensors;
        gpt_weights_ = gpt_weights;

        int total_output_tokens = max_output_length * max_batch_size;
        std::fill(output_ids, output_ids + total_output_tokens, 0);
        std::fill(sequence_lengths, sequence_lengths + max_batch_size, output_tensors_->at("sequence_length").shape[0]);
        std::fill(finished, finished + max_batch_size, false);

        prev_step = 0;
        curr_step = 0;
        is_generation_done = false;
    }

    /**
     * \brief Forward a model and asynchronously check whether to stop.
     *
     * The device having the last rank of a pipeline parallel group checks and
     * broadcasts to the other devices. So only the last rank runs asychronously
     * and monitor whether to terminate by given stop criteria.
     *
     * For now, we provide a streaming function as a separated example with
     * minimum update of the GPT module.
     *
     * \param gpt An ParallelGpt pointer to generate tokens.
     * \param output_tensors A vector of tensors, containing the output tensors of gpt, including
     *                          output_ids, parent_ids, sequence_lengths and cum_log_probs
     * \param input_tensors A vector of tensors, containing the input tensors of gpt, including
     *                          input_ids, input_lengths and request_output_len
     * \param gpt_weights A ParallelGptWeight pointer, which continas the weights of gpt model
     */
    void run(ParallelGpt<T>* gpt,
             std::unordered_map<std::string, Tensor>* output_tensors,
             const std::unordered_map<std::string, Tensor>* input_tensors,
             const ParallelGptWeight<T>* gpt_weights)
    {
        initialize(gpt, output_tensors, input_tensors, gpt_weights);
        // Only the last rank of pipeline parallel will run asynchronously
        // and monitor whether to terminate by given stop criteria.
        if (gpt_->getPipelineParallelRank() < gpt_->getPipelineParallelSize() - 1) {
            gpt_->forward(output_tensors_, input_tensors_, gpt_weights_);
            return;
        }

        int device;
        check_cuda_error(cudaGetDevice(&device));
        std::async(std::launch::async, [&]() {
            check_cuda_error(cudaSetDevice(device));
            gpt_->forward(output_tensors_, input_tensors_, gpt_weights_);
        });
        streamDecoding();
    }
};

template<typename T>
class GptFileStreamer: public GptStreamer<T> {

protected:
    const std::string output_file;
    std::ofstream ofs;

    void streamHook(const int prev_step, const int curr_step, const int* output_ids) override
    {
        if (ofs.is_open()) {
            int batch_size = this->output_tensors_->at("output_ids").shape[0];
            for (int s = prev_step; s < curr_step; s++) {
                ofs << strutils::toString(output_ids + s * batch_size, batch_size, false) << std::endl;
            }
        }
    }

    void onStreamBegin(const int batch_size, const int input_len) override
    {
        if (!output_file.empty() && this->gpt_->getTensorParallelRank() == 0) {
            ofs.open(output_file, std::ios::out);
        }
    }

    void onStreamEnd() override
    {
        if (!output_file.empty()) {
            ofs.close();
        }
    }

public:
    GptFileStreamer(const int max_batch_size, const int max_output_length, const std::string output_file):
        GptStreamer<T>(max_batch_size, max_output_length), output_file(output_file)
    {
    }
    ~GptFileStreamer() {}
};

template<typename T>
void multi_gpu_gpt_example(const INIReader reader);

int main(int argc, char* argv[])
{
    MPICHECK(MPI_Init(&argc, &argv));
    srand(0);

    std::string ini_name;
    if (argc >= 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    }
    std::cout << "[INFO] Configuration file path: " << ini_name << std::endl;

    if (argc >= 3) {
        USE_ASYNC = std::atoi(argv[2]) == 1;
    }
    if (USE_ASYNC) {
        std::cout << "[INFO] Enable async forward" << std::endl;
    }
    else {
        std::cout << "[INFO] Disable async forward" << std::endl;
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        multi_gpu_gpt_example<__nv_bfloat16>(reader);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    MPI_Finalize();
    return 0;
}

int read_start_ids(int batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   int& max_input_len,
                   const int end_id,
                   const int beam_width)
{
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int> tmp_start_lengths;

    std::string file_name = "../examples/cpp/multi_gpu_gpt/start_ids.csv";
    std::ifstream start_id_file(file_name, std::ios::in);
    if (start_id_file.is_open()) {
        std::string line;
        int i0 = 0;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string vals;
            int i1 = 0;
            std::vector<int> tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            i0++;
        }
    }
    else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); i++) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; i++) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int b = 0; b < beam_width; b++) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); j++) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
        }
    }
    return 0;
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader)
{
    const std::string model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    const size_t max_batch_size = reader.GetInteger("ft_instance_hyperparameter", "max_batch_size");
    const size_t max_seq_len = reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");
    const size_t beam_width = reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const int top_k = reader.GetInteger("ft_instance_hyperparameter", "top_k");
    const float top_p = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    const float temperature = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    const float repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty");
    const std::string model_dir = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    const bool sparse = static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    const float len_penalty = 1.0f;
    const float beam_search_diversity_rate = 0.0f;

    const int tensor_para_size = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    const int pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    const int int8_mode = reader.GetInteger("ft_instance_hyperparameter", "int8_mode");

    const size_t head_num = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head = reader.GetInteger(model_name, "size_per_head");
    const size_t vocab_size = reader.GetInteger(model_name, "vocab_size");
    const size_t decoder_layers = reader.GetInteger(model_name, "decoder_layers");
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    // The length of tokens we hope this model to generate
    const int request_output_len = reader.GetInteger("request", "request_output_len");

    const int start_id = 50256;
    const int end_id = 50256;

    if (USE_ASYNC) {
        FT_CHECK(beam_width == 1);  // async forward does not support beam search
    }

    FT_CHECK(head_num % tensor_para_size == 0);
    FT_CHECK(decoder_layers % pipeline_para_size == 0);

    // Prepare the parallelism parameters
    int rank, world_size, device, device_count;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is runing with %d GPU.\n", rank, device);

    if (tensor_para_size * pipeline_para_size != world_size) {
        printf("[ERROR] tensor_para_size * pipeline_para_size should equal to world_size \n");
        exit(-1);
    }

    const int tensor_para_rank = rank % tensor_para_size;
    const int pipeline_para_rank = rank / tensor_para_size;
    const int layers_per_group = decoder_layers / pipeline_para_size;
    if (layers_per_group * pipeline_para_size != (int)decoder_layers) {
        printf("[ERROR] layers_per_group (%d) * pipeline_para_size (%d) should equal to decoder_layers (%ld) \n",
               layers_per_group,
               pipeline_para_size,
               decoder_layers);
        exit(-1);
    }
    ncclUniqueId tensor_para_nccl_uid;
    ncclUniqueId pipeline_para_nccl_uid;

    // assume gpu_num = n * k,
    // tensor parallelism group size is n
    // pipeline parallelism group size is k

    if (tensor_para_rank == 0) {
        // get the uid of each tensor parallelism group
        // here, 0, 1, ..., n-1 are in group 0,
        //       n, ..., 2n - 1 are in group 1.
        NCCLCHECK(ncclGetUniqueId(&tensor_para_nccl_uid));
        for (int i = 1; i < tensor_para_size; i++) {
            printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n", rank, rank + i);
            MPICHECK(
                MPI_Send(&tensor_para_nccl_uid, sizeof(tensor_para_nccl_uid), MPI_BYTE, rank + i, 0, MPI_COMM_WORLD));
        }
    }
    else {
        MPI_Status status;
        printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n", rank, rank - tensor_para_rank);
        MPICHECK(MPI_Recv(&tensor_para_nccl_uid,
                          sizeof(tensor_para_nccl_uid),
                          MPI_BYTE,
                          rank - tensor_para_rank,
                          0,
                          MPI_COMM_WORLD,
                          &status));
    }

    if (pipeline_para_rank == 0) {
        // get the uid of each pipeline parallelism group
        // 0, k, 2k, are in group 0
        // 1, k+1, 2k+1 are in group 1
        NCCLCHECK(ncclGetUniqueId(&pipeline_para_nccl_uid));
        for (int i = 1; i < pipeline_para_size; i++) {
            printf("[INFO] rank %d sends pipeline_para_nccl_uid to rank %d \n", rank, rank + i * tensor_para_size);
            MPICHECK(MPI_Send(&pipeline_para_nccl_uid,
                              sizeof(pipeline_para_nccl_uid),
                              MPI_BYTE,
                              rank + i * tensor_para_size,
                              0,
                              MPI_COMM_WORLD));
        }
    }
    else {
        MPI_Status status;
        printf("[INFO] rank %d receives pipeline_para_nccl_uid from rank %d \n", rank, rank % tensor_para_size);
        MPICHECK(MPI_Recv(&pipeline_para_nccl_uid,
                          sizeof(pipeline_para_nccl_uid),
                          MPI_BYTE,
                          rank % tensor_para_size,
                          0,
                          MPI_COMM_WORLD,
                          &status));
    }

    ncclComm_t tensor_para_nccl_comm, pipeline_para_nccl_comm;
    NCCLCHECK(ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
    NCCLCHECK(
        ncclCommInitRank(&pipeline_para_nccl_comm, pipeline_para_size, pipeline_para_nccl_uid, pipeline_para_rank));

    // Read ids of request from file.
    int max_input_len = -1;
    std::vector<int> v_start_lengths;
    std::vector<int> v_start_ids;
    read_start_ids(request_batch_size, &v_start_lengths, &v_start_ids, max_input_len, end_id, beam_width);

    int* d_input_ids;
    int* d_input_lengths;
    if (max_input_len == 0) {
        // unconditional case, no input ids, so do nothing.
        d_input_ids = nullptr;
        d_input_lengths = nullptr;
        max_input_len = 0;
    }
    else {
        // conditional case.
        deviceMalloc(&d_input_ids, request_batch_size * beam_width * max_input_len, false);
        deviceMalloc(&d_input_lengths, request_batch_size * beam_width, false);
        cudaH2Dcpy(d_input_ids, v_start_ids.data(), request_batch_size * beam_width * max_input_len);
        cudaH2Dcpy(d_input_lengths, v_start_lengths.data(), request_batch_size * beam_width);
    }

    const int total_output_len = max_input_len + request_output_len;
    if (total_output_len > (int)max_seq_len) {
        printf("[ERROR] total_output_len (%d) should be <= max_seq_len (%ld). \n", total_output_len, max_seq_len);
        exit(-1);
    }

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    cublasSetStream(cublas_handle, stream);
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

    Allocator<AllocatorType::CUDA> allocator(getDevice());

    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper cublas_wrapper =
        cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, &allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    fastertransformer::ParallelGptWeight<T> gpt_weights(hidden_units,
                                                        inter_size,
                                                        vocab_size,
                                                        decoder_layers,
                                                        max_seq_len,
                                                        tensor_para_size,
                                                        tensor_para_rank,
                                                        pipeline_para_size,
                                                        pipeline_para_rank,
                                                        int8_mode);
    gpt_weights.loadModel(model_dir);
    NcclParam tensor_para(tensor_para_rank, tensor_para_size, tensor_para_nccl_comm);
    NcclParam pipeline_para(pipeline_para_rank, pipeline_para_size, pipeline_para_nccl_comm);
    unsigned long long int random_seed = 0;

    ParallelGpt<T> gpt = ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                        0,  // max_seq_len, FT will adjust the buffer automatically.
                                        0,  // max_input_len, FT will adjust the buffer automatically.
                                        beam_width,
                                        head_num,
                                        size_per_head,
                                        inter_size,
                                        decoder_layers,
                                        vocab_size,
                                        start_id,
                                        end_id,
                                        0.0f,
                                        top_k,
                                        top_p,
                                        random_seed,
                                        temperature,
                                        1.0f,  // len_penalty,
                                        repetition_penalty,
                                        tensor_para,
                                        pipeline_para,
                                        stream,
                                        &cublas_wrapper,
                                        &allocator,
                                        false,
                                        &prop,
                                        false,
                                        int8_mode);

    int* d_output_ids;
    int* d_parent_ids;
    int* d_sequence_lengths;
    deviceMalloc(&d_output_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_parent_ids, request_batch_size * beam_width * total_output_len, false);
    deviceMalloc(&d_sequence_lengths, request_batch_size * beam_width, false);

    std::unordered_map<std::string, Tensor> input_tensors = std::unordered_map<std::string, Tensor>{
        {"input_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size * beam_width, (size_t)max_input_len},
                d_input_ids}},
        {"input_lengths",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size * beam_width}, d_input_lengths}},
        {"max_output_seq_len", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &total_output_len}}};
    if (top_k == 0 && top_p == 0.0f) {
        FT_CHECK(beam_width > 1);
        input_tensors.insert({"beam_search_diversity_rate",
                              Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    }
    else {
        if (top_p != 0.0f) {
            input_tensors.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &top_p}});
        }
        if (top_k != 0) {
            input_tensors.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &top_k}});
        }
    }
    input_tensors.insert({"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &temperature}});
    input_tensors.insert({"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
    input_tensors.insert(
        {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    input_tensors.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});

    std::unordered_map<std::string, Tensor> output_tensors = std::unordered_map<std::string, Tensor>{
        {"output_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                d_output_ids}},
        {"parent_ids",
         Tensor{MEMORY_GPU,
                TYPE_INT32,
                std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width},
                d_parent_ids}},
        {"sequence_length",
         Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths}},
        {"output_log_probs",
         Tensor{MEMORY_GPU,
                TYPE_FP32,
                std::vector<size_t>{(size_t)request_output_len, request_batch_size, beam_width},
                nullptr}}};

    print_mem_usage();

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    int total_output_ids = total_output_len * request_batch_size;
    int* h_output_ids = new int[total_output_ids];
    int* h_sequence_lengths = new int[request_batch_size * beam_width];

    if (rank == 0) {
        printf("[INFO] Warming up\n");
    }

    // initialize output buffers
    std::fill(h_output_ids, h_output_ids + total_output_ids, 0);
    std::fill(h_sequence_lengths, h_sequence_lengths + request_batch_size, 0);

    std::string stream_file = pipeline_para_rank == (pipeline_para_size - 1) ? "out.stream" : "";
    GptFileStreamer<T> gpt_streamer(request_batch_size * beam_width, total_output_len, stream_file);

    cudaProfilerStart();
    // warm up
    nvtx::setScope("warmup_time");
    PUSH_RANGE("warmup time")
    if (USE_ASYNC) {
        gpt_streamer.run(&gpt, &output_tensors, &input_tensors, &gpt_weights);
    }
    else {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    POP_RANGE;
    nvtx::resetScope();

    if (rank == 0) {
        printf("[INFO] Profiling\n");
    }
    // reset output buffers
    std::fill(h_output_ids, h_output_ids + total_output_ids, 0);
    std::fill(h_sequence_lengths, h_sequence_lengths + request_batch_size, 0);
    deviceFill(d_sequence_lengths, request_batch_size, 0);

    struct timeval start, end;
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&start, NULL);

    nvtx::setScope("total_time");
    PUSH_RANGE("total time")
    int ite = 1;
    for (int i = 0; i < ite; ++i) {
        if (USE_ASYNC) {
            gpt_streamer.run(&gpt, &output_tensors, &input_tensors, &gpt_weights);
        }
        else {
            gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
        }
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    POP_RANGE;
    nvtx::resetScope();
    gettimeofday(&end, NULL);

    cudaProfilerStop();

    if (rank == 0) {
        printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %d"
               " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
               request_batch_size,
               beam_width,
               head_num,
               size_per_head,
               total_output_len,
               decoder_layers,
               vocab_size,
               ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

        std::string fName = USE_ASYNC ? "out.async" : "out.sync";
        auto outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
        }
        else {
            size_t outCount = total_output_len * request_batch_size;
            int* hBuf = new int[outCount];
            cudaDeviceSynchronize();
            cudaMemcpyAsync(hBuf, d_output_ids, outCount * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaDeviceSynchronize();

            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0)) {
                        zeroCount++;
                    }
                    outFile << hBuf[i];
                    if ((i + 1) % (total_output_len) == 0) {
                        outFile << std::endl;
                    }
                    else {
                        outFile << " ";
                    }

                    if (i < 10) {
                        printf("%5d ", hBuf[i]);
                    }
                    if ((i + 1) % (total_output_len) == 0 && i < 10) {
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
            }
            delete[] hBuf;
        }
    }

    ncclCommDestroy(tensor_para_nccl_comm);
    ncclCommDestroy(pipeline_para_nccl_comm);

    delete[] h_output_ids;
    delete[] h_sequence_lengths;
    return;
}
