#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace fastertransformer {
using memcpy_fn = std::function<void(void*, void*, size_t)>;

template<typename T>
void serialize(uint8_t*& buffer, const std::vector<std::pair<size_t, T*>>& v, const memcpy_fn& fn)
{
    for (int i = 0; i < v.size(); i++) {
        size_t bytes = v[i].first * sizeof(T);
        fn((void*)buffer, v[i].second, bytes);
        buffer += bytes;
    }
}

template<typename T>
void serialize(uint8_t*& buffer, const std::vector<T*>& v, const memcpy_fn& fn)
{
    for (int i = 0; i < v.size(); i++) {
        size_t bytes = sizeof(T);
        fn((void*)buffer, v[i], bytes);
        buffer += bytes;
    }
}

template<typename T>
void serialize_d2h(uint8_t*& buffer, const std::vector<std::pair<size_t, T*>>& v)
{
    serialize(buffer, v, [](void* src, void* dst, size_t sz) { cudaMemcpy(src, dst, sz, cudaMemcpyDeviceToHost); });
}

template<typename T>
void serialize_d2h(uint8_t*& buffer, const std::vector<T*>& v)
{
    serialize(buffer, v, [](void* src, void* dst, size_t sz) { cudaMemcpy(src, dst, sz, cudaMemcpyDeviceToHost); });
}

template<typename T>
void serialize_h2h(uint8_t*& buffer, const std::vector<std::pair<size_t, T*>>& v)
{
    serialize(buffer, v, std::memcpy);
}

template<typename T>
void serialize_h2h(uint8_t*& buffer, const std::vector<T*>& v)
{
    serialize(buffer, v, std::memcpy);
}

template<typename T>
void deserialize(const uint8_t*& buffer, std::vector<std::pair<size_t, T*>>& v, const memcpy_fn& fn)
{
    for (int i = 0; i < v.size(); i++) {
        size_t bytes = v[i].first * sizeof(T);
        fn(v[i].second, (void*)buffer, bytes);
        buffer += bytes;
    }
}

template<typename T>
void deserialize(const uint8_t*& buffer, std::vector<T*>& v, const memcpy_fn& fn)
{
    for (int i = 0; i < v.size(); i++) {
        size_t bytes = sizeof(T);
        fn(v[i], (void*)buffer, bytes);
        buffer += bytes;
    }
}

template<typename T>
void deserialize_h2d(const uint8_t*& buffer, std::vector<std::pair<size_t, T*>>& v)
{
    deserialize(buffer, v, [](void* src, void* dst, size_t sz) { cudaMemcpy(src, dst, sz, cudaMemcpyHostToDevice); });
}

template<typename T>
void deserialize_h2d(const uint8_t*& buffer, std::vector<T*>& v)
{
    deserialize(buffer, v, [](void* src, void* dst, size_t sz) { cudaMemcpy(src, dst, sz, cudaMemcpyHostToDevice); });
}

template<typename T>
void deserialize_h2h(const uint8_t*& buffer, std::vector<std::pair<size_t, T*>>& v)
{
    deserialize(buffer, v, std::memcpy);
}

template<typename T>
void deserialize_h2h(const uint8_t*& buffer, std::vector<T*>& v)
{
    deserialize(buffer, v, std::memcpy);
}

}  // namespace fastertransformer