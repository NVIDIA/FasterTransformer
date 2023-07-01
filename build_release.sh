SM=80
BUILD_MIXED_GEMM=OFF
FORCE_BACKEND_REBUILD=0
SPARSITY_SUPPORT=OFF
BUILD_MULTI_GPU=OFF
ENABLE_FP8=OFF

export TORCH_CUDA_ARCH_LIST="8.0"

export CUDACXX=/usr/local/cuda/bin/nvcc  
export PATH=$PATH:/usr/local/mpi/bin/
# mkdir build -p && cd build && \
# # wget https://developer.download.nvidia.com/compute/libcusparse-lt/0.1.0/local_installers/libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
# # tar -xzvf libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
# cmake -DSM=${SM} -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYT=OFF -DSPARSITY_SUPPORT=${SPARSITY_SUPPORT} -DMEASURE_BUILD_TIME=ON \
#  -DBUILD_CUTLASS_MIXED_GEMM=${BUILD_MIXED_GEMM} -DCUSPARSELT_PATH=/workspace/FasterTransformer/build/libcusparse_lt/ \
#  -DBUILD_MULTI_GPU=${BUILD_MULTI_GPU} -DBUILD_TRT=OFF -DENABLE_FP8=${ENABLE_FP8} .. && \
# make -j"$(grep -c ^processor /proc/cpuinfo)"

 mkdir build_release -p && cd build_release && \
 # wget https://developer.download.nvidia.com/compute/libcusparse-lt/0.1.0/local_installers/libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
 # tar -xzvf libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
 cmake -DDEBUG_MEMORY_IN_FORWARD=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DSM=${SM} -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DSPARSITY_SUPPORT=${SPARSITY_SUPPORT} -DMEASURE_BUILD_TIME=ON \
   -DBUILD_CUTLASS_MIXED_GEMM=${BUILD_MIXED_GEMM} -DCUSPARSELT_PATH=/workspace/FasterTransformer/build/libcusparse_lt/ \
   -DBUILD_MULTI_GPU=${BUILD_MULTI_GPU} -DBUILD_TRT=ON -DENABLE_FP8=${ENABLE_FP8} .. && \
 make -j"$(grep -c ^processor /proc/cpuinfo)"


