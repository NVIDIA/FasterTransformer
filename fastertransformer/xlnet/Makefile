# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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



GCC=g++
NVCC=nvcc
BUILDDIR=obj
SRCDIR=src

CFLAGS= --std=c++11 -O3  -I.
FLAGS= --std=c++11 -O3 -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61  -gencode=arch=compute_62,code=sm_62  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_75,code=sm_75  -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 \
	   -I. -I/usr/local/cuda/include -lcudart -lcublas -lz

OBJ =$(BUILDDIR)/LoadData.o $(BUILDDIR)/layerKernels.o  $(BUILDDIR)/XlnetDebug.o $(BUILDDIR)/Xlnet.o $(BUILDDIR)/XlnetLayer.o $(BUILDDIR)/utils.o $(BUILDDIR)/cnpy.o
DEPS =$(SRCDIR)/LoadData.h $(SRCDIR)/layerKernels.h $(SRCDIR)/XlnetLayer.h $(SRCDIR)/Xlnet.h $(SRCDIR)/XlnetDebug.h $(SRCDIR)/utils.h $(SRCDIR)/cnpy.h

all: runXlnet runTest
.PHONY: all

#Compile
$(BUILDDIR)/%.o:$(SRCDIR)/%.cpp
	$(GCC) -c -o $@ $< $(CFLAGS)
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -c -o $@ $< $(FLAGS)
$(BUILDDIR)/runTest.o: $(SRCDIR)/runTest.cu ${DEPS}
	$(NVCC) -c -o $@ $< $(FLAGS)
$(BUILDDIR)/runXlnet.o: $(SRCDIR)/runXlnet.cu ${DEPS}
	$(NVCC) -c -o $@ $< $(FLAGS)

#Link
runTest: $(BUILDDIR)/runTest.o $(OBJ)
	$(NVCC) -o $@ $^ $(FLAGS)
runXlnet: $(BUILDDIR)/runXlnet.o $(OBJ)
	$(NVCC) -o $@ $^ $(FLAGS)

#Clean
clean:
	rm -f $(BUILDDIR)/*.o runXlnet runTest


