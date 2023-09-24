#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

namespace fastertransformer {

template<typename T>
static void _print_tensor1(T* out, int dim1, int indent)
{
    std::string ind(indent, ' ');
    int         start0 = 0;
    int         end0   = (dim1 < 3) ? dim1 : 3;
    int         start1 = (dim1 < 3) ? 0 : dim1 - 3;
    int         end1   = (dim1 < 3) ? 0 : dim1;

    std::cout << "[";
    for (int i = start0; i < end0; ++i) {
        std::cout << std::fixed << std::setw(7) << std::setprecision(4) << std::setfill(' ') << out[i];
        if (i != dim1 - 1)
            std::cout << " ";
    }
    if (end0 != start1) {
        std::cout << "... ";
    }
    for (int i = start1; i < end1; ++i) {
        std::cout << std::fixed << std::setw(7) << std::setprecision(4) << std::setfill(' ') << out[i];
        if (i != end1 - 1)
            std::cout << " ";
    }
    std::cout << "]";
}

template<typename T>
static void _print_tensor2(T* out, int dim1, int dim2, int stride, int indent)
{
    std::string ind(indent, ' ');
    int         start0 = 0;
    int         end0   = (dim1 < 3) ? dim1 : 3;
    int         start1 = (dim1 < 3) ? 0 : dim1 - 3;
    int         end1   = (dim1 < 3) ? 0 : dim1;
    std::cout << "[";
    for (int i = start0; i < end0; ++i) {
        if (i != start0)
            std::cout << ind;
        _print_tensor1(&out[i * stride], dim2, indent + 1);
        if (i != dim1 - 1)
            std::cout << "\n";
    }
    if (end0 != start1) {
        std::cout << ind;
        std::cout << "...\n";
    }
    for (int i = start1; i < end1; ++i) {
        std::cout << ind;
        _print_tensor1(&out[i * stride], dim2, indent + 1);
        if (i != end1 - 1)
            std::cout << "\n";
    }
    std::cout << "]";
}

template<typename T>
static void _print_tensor3(T* out, int dim1, int dim2, int dim3, int stride1, int stride2, int indent)
{
    std::string ind(indent, ' ');

    int start0 = 0;
    int end0   = (dim1 < 3) ? dim1 : 3;
    int start1 = (dim1 < 3) ? 0 : dim1 - 3;
    int end1   = (dim1 < 3) ? 0 : dim1;
    std::cout << "[";
    for (int i = start0; i < end0; ++i) {
        if (i != start0)
            std::cout << ind;
        _print_tensor2(&out[i * stride1], dim2, dim3, stride2, indent + 1);
        if (i != dim1 - 1)
            std::cout << "\n\n";
    }
    if (start1 != end1) {
        std::cout << ind;
        std::cout << "...\n\n";
    }
    for (int i = start1; i < end1; ++i) {
        std::cout << ind;
        _print_tensor2(&out[i * stride1], dim2, dim3, stride2, indent + 1);
        if (i != end1 - 1)
            std::cout << "\n";
    }
    std::cout << "]\n";
}

template<typename T>
static void
_print_tensor4(T* out, int dim1, int dim2, int dim3, int dim4, int stride1, int stride2, int stride3, int indent)
{
    std::string ind(indent, ' ');

    int start0 = 0;
    int end0   = (dim1 < 3) ? dim1 : 3;
    int start1 = (dim1 < 3) ? 0 : dim1 - 3;
    int end1   = (dim1 < 3) ? 0 : dim1;
    std::cout << "[";
    for (int i = start0; i < end0; ++i) {
        if (i != start0)
            std::cout << ind;
        _print_tensor3(&out[i * stride1], dim2, dim3, dim4, stride2, stride3, indent + 1);
        if (i != dim1 - 1)
            std::cout << "\n\n";
    }
    if (start1 != end1) {
        std::cout << ind;
        std::cout << "...\n\n";
    }
    for (int i = start1; i < end1; ++i) {
        std::cout << ind;
        _print_tensor3(&out[i * stride1], dim2, dim3, dim4, stride2, stride3, indent + 1);
        if (i != end1 - 1)
            std::cout << "\n";
    }
    std::cout << "]\n";
}

template<typename T>
static void print_tensor3(T* in, int dim1, int dim2, int dim3, int stride1, int stride2, int size, int start)
{
    T* out = (T*)malloc(sizeof(T) * size);
    cudaMemcpy(out, in, sizeof(T) * size, cudaMemcpyDeviceToHost);
    _print_tensor3(&out[start], dim1, dim2, dim3, stride1, stride2, 1);

    /*
    if (stride2 != dim3) {
        for (int i = dim1 * dim2 * 3 * dim3 - 1 * dim3 - 8; i < dim1 * dim2 * 3 * dim3 - 1 * dim3; ++i) {
            std::cout << out[i] << " ";
        }
        std::cout << "\n";
    }
    */
    free(out);
}

template<typename T>
static void print_tensor3(T* in, int dim1, int dim2, int dim3)
{
    print_tensor3(in, dim1, dim2, dim3, dim2 * dim3, dim3, dim1 * dim2 * dim3, 0);
}

template<typename T>
static void
print_tensor4(T* in, int dim1, int dim2, int dim3, int dim4, int stride1, int stride2, int stride3, int size, int start)
{
    T* out = (T*)malloc(sizeof(T) * size);
    cudaMemcpy(out, in, sizeof(T) * size, cudaMemcpyDeviceToHost);
    _print_tensor4(&out[start], dim1, dim2, dim3, dim4, stride1, stride2, stride3, 1);
    for (int i = dim1 * dim2 * dim3 * dim4 - 8; i < dim1 * dim2 * dim3 * dim4; ++i) {
        std::cout << out[i] << " ";
    }
    std::cout << "\n";
    free(out);
}

template<typename T>
static void print_tensor4(T* in, int dim1, int dim2, int dim3, int dim4)
{
    print_tensor4(in, dim1, dim2, dim3, dim4, dim2 * dim3 * dim4, dim3 * dim4, dim4, dim1 * dim2 * dim3 * dim4, 0);
}

}  // namespace fastertransformer
