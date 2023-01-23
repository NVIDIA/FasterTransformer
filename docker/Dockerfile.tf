# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -------------------------------------------------- #
# This is a Docker image dedicated to develop
# FasterTransformer.
# -------------------------------------------------- #

ARG DOCKER_VERSION=22.09
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:${DOCKER_VERSION}-tf1-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends bc git-lfs&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# backend build
WORKDIR /workspace/FasterTransformer
ADD . /workspace/FasterTransformer

RUN git submodule update --init --recursive

ENV NCCL_LAUNCH_MODE=GROUP
ARG SM=80
ARG FORCE_BACKEND_REBUILD=0
ARG ENABLE_FP8=OFF
RUN mkdir /var/run/sshd -p && \
    mkdir build -p && cd build && \
    cmake -DSM=${SM} -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow_core/ -DBUILD_MULTI_GPU=ON -DENABLE_FP8=${ENABLE_FP8} .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)"
