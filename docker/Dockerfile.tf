# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

ARG DOCKER_VERSION=22.07
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:${DOCKER_VERSION}-tf1-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends bc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# backend build
WORKDIR /workspace/FasterTransformer
ADD . /workspace/FasterTransformer

RUN git submodule update --init --recursive

ARG SM=80
ARG FORCE_BACKEND_REBUILD=0
RUN mkdir /var/run/sshd -p && \
    mkdir build -p && cd build && \
    cmake -DSM=${SM} -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow_core/ -DBUILD_MULTI_GPU=ON .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)"
