CXX := g++
CXXFLAGS := -Wall -Werror -g
CUDAFLAG := -Xcompiler -Wall -ldl -DWMMA -O0 -G -Xcompiler -Wall

CUBLAS_PATH := /home/scratch.bhsueh_sw_1/cublas_build_x86_64_centos7_cuda11.7_roma_nostromo_gpgpu_31055760_release/
SDK_PATH := /home/scratch.bhsueh_sw_1/cuda_toolkit-gpgpu-31073558/



CUBLAS_INC := -I$(CUBLAS_PATH)/include/
CUBLAS_LIBS_PATH := -L$(CUBLAS_PATH)/lib64/
CUBLAS_LIBS := -lcublasLt

SDK_INC := -I$(SDK_PATH)/include
SDK_LIBS_PATH := -L$(SDK_PATH)/lib64/
SDK_LIBS := -lcudart_static

LOCAL_INC := -I$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LINKER_BONUS := -Wl,-rpath=$(CUBLAS_PATH)/lib64

# static cudart
SYS_LIBS := -L/lib/x86_64-linux-gnu/ -ldl -lpthread -lrt

OBJECTS := main.o worker.o cuda_utils.o

tester_app: $(OBJECTS)
	$(CXX) $(OBJECTS) -o tester_app $(CUBLAS_LIBS_PATH) $(CUBLAS_LIBS) $(SDK_LIBS_PATH) $(SDK_LIBS) $(SYS_LIBS) $(LINKER_BONUS)

cuda_utils.o : cuda_utils.cu
	nvcc ${CUDAFLAG} -c cuda_utils.cu -o cuda_utils.o ${SDK_INC}

worker.o : worker.cpp
	$(CXX) $(CXXFLAGS) -c worker.cpp -o worker.o $(LOCAL_INC) $(CUBLAS_INC) $(SDK_INC)

main.o : main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o $(SDK_INC)

clean:
	rm -f *.o tester_app

#all :
#	echo $(CASK_OBJS)
