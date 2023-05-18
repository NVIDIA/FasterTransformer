#pragma once

#include <cstdlib>
#include <map>
#include <string>

namespace fastertransformer {

class TokenPipe {
private:
    int pipe_fd_;
    int64_t request_id_;

public:
    TokenPipe(int pipe_fd, int64_t request_id) : pipe_fd_(pipe_fd), request_id_(request_id) {}

    ssize_t stream_tokens(std::unordered_map<std::string, Tensor>* tensors) {
        auto output_ids_iter = tensors->find("output_ids");
        auto sequence_length_iter = tensors->find("sequence_length");
        if (output_ids_iter == tensors->end()) return -1;
        if (sequence_length_iter == tensors->end()) return -1;
        auto &output_ids = output_ids_iter->second;
        auto &sequence_length = sequence_length_iter->second;
        // printf("output_ids size %ld, %ld\n", output_ids.size(), output_ids.sizeBytes());
        // printf("output_ids shape %ld, %ld, %ld\n", output_ids.shape[0], output_ids.shape[1], output_ids.shape[2]);

        int32_t output_ids_cpu[output_ids.sizeBytes() / sizeof(int32_t) + 1];
        int32_t sequence_length_cpu[sequence_length.sizeBytes() / sizeof(int32_t) + 1];
        cudaDeviceSynchronize();
        cudaMemcpy(output_ids_cpu, output_ids.getPtr<int32_t>(), output_ids.sizeBytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(sequence_length_cpu, sequence_length.getPtr<int32_t>(), sequence_length.sizeBytes(), cudaMemcpyDeviceToHost);

        int32_t token = output_ids_cpu[sequence_length_cpu[0]-1];
        char buf[PIPE_BUF];
        int len = snprintf(buf, sizeof(buf), "{ \"id\": %ld, \"token\": [ %d ] }\n", this->request_id_, token);
        buf[sizeof(buf)-1] = '\n';

        // printf("%d = write %d, \"%s\"\n", len, this->pipe_fd_, buf);
        return write(this->pipe_fd_, buf, len);
    }

    static void stream_tokens_callback(std::unordered_map<std::string, Tensor>* tensors, void *opaque) {
        reinterpret_cast<TokenPipe*>(opaque)->stream_tokens(tensors);
    }
};

}  // namespace fastertransformer