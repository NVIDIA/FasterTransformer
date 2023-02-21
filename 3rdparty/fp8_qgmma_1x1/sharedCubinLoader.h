/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef SHARED_CUBIN_LOADER_H
#define SHARED_CUBIN_LOADER_H

#include "3rdparty/common/cudaDriverWrapper.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

template <typename TKernelMeta, typename TKernelParam>
class TSharedCubinKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    virtual uint64_t hashID(KernelMeta const& kernelMeta) const = 0;
    virtual uint64_t hashID(TKernelParam const& param) const = 0;

    TSharedCubinKernel(TKernelMeta const* pMetaStart, int32_t nMetaCount, int32_t sm)
        : mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
        gpuErrChk(cudaGetDeviceProperties(&mProp, 0));
    }

    void loadCubinKernels(int32_t smVersion)
    {
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            auto const kernelKey = hashID(kernelMeta);
            if (kernelMeta.mSM == smVersion
                && mFunctions.find(kernelKey) == mFunctions.end())
            {
                int32_t const DEFAULT_SMEM_SIZE{48 * 1024};
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    int32_t deviceID{0};
                    cudaGetDevice(&deviceID);
                    int32_t sharedMemPerMultiprocessor{0};
                    if (cudaDeviceGetAttribute(
                            &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID)
                            != cudaSuccess
                        || sharedMemPerMultiprocessor < kernelMeta.mSharedMemBytes)
                    {
                        // skip load function because not enough shared memory to launch the kernel
                        continue;
                    }
                }
                
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                Fp8Gemm1x1KernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes)
                        != CUDA_SUCCESS)
                    {
                        // some chip may not have enough shared memory to launch the kernel
                        continue;
                    }
                }
                mFunctions.insert({kernelKey, funcInfo});
            }
        }
    }

    void loadCubinKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        loadCubinKernels(mSM);
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        auto const findIter = mFunctions.find(hashID(params));
        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        CUfunction const func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver.cuLaunchKernel(func, mProp.multiProcessorCount, 1, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                        kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    virtual ~TSharedCubinKernel() = default;

protected:
    cudaDeviceProp mProp;

    CUDADriverWrapper mDriver;

    TKernelMeta const* mKernelMeta;
    int32_t mKernelMetaCount;
    int32_t mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;
    struct Fp8Gemm1x1KernelInfo
    {
        int32_t mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, Fp8Gemm1x1KernelInfo> mFunctions;
};

template <typename TKernelList>
class TSharedCubinKernelFactory
{
public:
    TKernelList const* getCubinKernels(
        typename TKernelList::KernelMeta const* pKernelList, int32_t nbKernels, int32_t sm)
    {
        static std::mutex sMutex;
        std::lock_guard<std::mutex> lg(sMutex);

        auto const id = hashID(sm);
        auto const findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            auto* newKernel = new TKernelList{pKernelList, nbKernels, sm};
            newKernel->loadCubinKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TSharedCubinKernelFactory<TKernelList>& Get()
    {
        static TSharedCubinKernelFactory<TKernelList> gFactory;
        return gFactory;
    }

private:
    TSharedCubinKernelFactory() = default;

    inline uint64_t hashID(int32_t sm) const
    {
        // Concatenate sm with deviceID to support Multi-GPU cubin loading
        // Bottom 32 bits are for SM, top 32 bits for deviceID
        int32_t deviceID{0};
        cudaGetDevice(&deviceID);
        return (uint64_t) deviceID << 32 | (uint64_t)sm;
    }

    std::unordered_map<uint64_t, std::unique_ptr<TKernelList> const> mKernels;
};

#endif // SHARED_CUBIN_LOADER_H
