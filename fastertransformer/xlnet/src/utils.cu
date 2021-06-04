/*
* Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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


#include "utils.h"


using namespace std;
namespace std {
  template <typename _CharT, typename _Traits>
  inline basic_ostream<_CharT, _Traits> &
  tab(basic_ostream<_CharT, _Traits> &__os) {
    return __os.put(__os.widen('\t'));
  }
}

std::string stringPadding(std::string original, size_t charCount)
{
    original.resize(charCount, ' ');
    return original;
}

/*************Error Handling**************/
bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << endl;
        exit(0);
        return false;
    }
    return true;
}
const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

bool check(cublasStatus_t e, int iLine, const char *szFile) {
    if (e !=CUBLAS_STATUS_SUCCESS) {
        cout << "CUDA runtime API error " << cublasGetErrorString(e) << " at line " << iLine << " in file " << szFile << endl;
        exit(0);
        return false;
    }
    return true;
}

/*************Time Handling**************/
CudaTimer::CudaTimer(cudaStream_t stream){
    this->stream=stream;
}

void CudaTimer::start(){
    ck(cudaEventCreate(&event_start));
    ck(cudaEventCreate(&event_stop));
    ck(cudaEventRecord(event_start, stream)); 
}
float CudaTimer::stop(){
    ck(cudaEventRecord(event_stop,stream));
    ck(cudaEventSynchronize(event_stop));
    ck(cudaEventElapsedTime(&time, event_start, event_stop));
    ck(cudaEventDestroy(event_start));
    ck(cudaEventDestroy(event_stop));
    return time;
}
CudaTimer:: ~CudaTimer(){
}


/*************Useful functions***********************/
int blockNum(int size, int blockSize){
    int nblock= (size-1)/blockSize+1;
    return nblock;
}
int next_pow2(int a){
    int rval=32;
    if(a>32){
        while(rval<a) rval<<=1;
    }
    return rval;
}


template<typename T>
int numPerThread(){
    return sizeof(float)/sizeof(T); 
}

    template <typename T>
void deviceMalloc(T** ptr, int size)
{
    ck(cudaMalloc((void**)ptr, sizeof(T) * size));
}

    template <typename T>
void deviceMemset(T* ptr, int value, int size)
{
    ck(cudaMemset((void*)ptr,0, sizeof(T) * size));
}

    template <typename T>
void deviceFree(T* & ptr){
    if(ptr!=NULL){
        ck(cudaFree(ptr));
        ptr=NULL;
    }

}

    template <typename T>
void deviceMemcpyHtoD(cudaStream_t stream, T* d_ptr,T* h_ptr, int size)
{
   ck(cudaMemcpyAsync(d_ptr, h_ptr,size *sizeof(T),cudaMemcpyHostToDevice,stream));
}


    template <typename T>
float castToFloat(T input){
    float output=(T)(input);
    return output;
}

template<>
float castToFloat(__half input){
    float output=__half2float(input);
    return output;
}


/*********************Npz &Npy File Process functions***********************/
std::string paraName(int i_layer, std::string sub_para){
    std::ostringstream s;
    s<<"model/transformer/layer_"<<i_layer<<sub_para;
    std::string str= s.str();
    return str;
}


std::string paraName(std::string s){
    std::string str= s;
    return str;
}

template <typename T>
void setByNpz(cnpy::npz_t & my_npz, std::string name, T* h_ptr, int size, int offset){
    //printKey(my_npz);

    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    //load it into a new array
    T* loaded_data = arr.data<T>();
    memcpy (h_ptr, loaded_data+offset, sizeof(T)*size);
}
template<>
void setByNpz<__half>(cnpy::npz_t & my_npz, std::string name, __half* h_ptr, int size, int offset){
   //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    //load it into a new array
    float* loaded_data = arr.data<float>();
    __half* half_data=(__half*)malloc(sizeof(__half)*size);

    loaded_data=loaded_data+offset;
    for(int i=0;i<size;i++){
        half_data[i]=__float2half_rn(loaded_data[i]);
    }


    memcpy (h_ptr, half_data, sizeof(__half)*size);
    free(half_data);
}


void printKey(cnpy::npz_t & npz){
    std::map<std::string,cnpy::NpyArray>::iterator iter;
    for(iter = npz.begin(); iter != npz.end(); iter++){
        std::cout<<iter->first<<std::endl;
    }
}
void setByNpz(cudaStream_t stream,cnpy::npz_t & my_npz, std::string name, int* d_ptr, int size, int offset){
    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];
    //load it into a new array
    int* loaded_data = arr.data<int>();
    ck(cudaMemcpyAsync(d_ptr, loaded_data+offset, sizeof(int)*size, cudaMemcpyHostToDevice,stream));
    cudaDeviceSynchronize();
    ck(cudaGetLastError());

}
void setByNpz(cudaStream_t stream,cnpy::npz_t & my_npz, std::string name, float* d_ptr, int size, int offset){

    //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    //load it into a new array
    float* loaded_data = arr.data<float>();

    //std::cout<<name<<" "<<size<<" "<<d_ptr<<" "<<loaded_data<<std::endl;
    ck(cudaMemcpyAsync(d_ptr, loaded_data+offset, sizeof(float)*size, cudaMemcpyHostToDevice,stream));
    cudaDeviceSynchronize();
    ck(cudaGetLastError());

}

void setByNpz(cudaStream_t stream,cnpy::npz_t & my_npz, std::string name, __half* d_ptr, int size, int offset){
   //check that the loaded myVar1 matches myVar1
    cnpy::NpyArray arr = my_npz[name];

    //load it into a new array
    float* loaded_data = arr.data<float>();
    __half* half_data=(__half*)malloc(sizeof(__half)*size);

    loaded_data=loaded_data+offset;
    for(int i=0;i<size;i++){
        half_data[i]=__float2half_rn(loaded_data[i]);
    }

    ck(cudaMemcpyAsync(d_ptr, half_data, sizeof(__half)*size, cudaMemcpyHostToDevice,stream));
    free(half_data);
    cudaDeviceSynchronize();
    ck(cudaGetLastError());
}
void setByNpy(cudaStream_t stream,float* d_ptr, int size,std::string dir, std::string fname){
    std::ostringstream s;
    s<<dir<<fname;
    std::string fullFname= s.str();
    //load it into a new array
    cnpy::NpyArray arr = cnpy::npy_load(fullFname);
    float* loaded_data = arr.data<float>();

    ck(cudaMemcpy(d_ptr, loaded_data, sizeof(__half)*size, cudaMemcpyHostToDevice));
}
void setByNpy(cudaStream_t stream, __half* d_ptr, int size,std::string dir, std::string fname){
    std::ostringstream s;
    s<<dir<<fname;
    std::string fullFname= s.str();
    //load it into a new array
    cnpy::NpyArray arr = cnpy::npy_load(fullFname);
    float* loaded_data = arr.data<float>();

    __half* half_data=(__half*)malloc(sizeof(__half)*size);

    for(int i=0;i<size;i++){
        half_data[i]=__float2half_rn(loaded_data[i]);
    }
    ck(cudaMemcpy(d_ptr, half_data, sizeof(__half)*size, cudaMemcpyHostToDevice));

    free(half_data);
}

void checkByNpy(cudaStream_t stream,float* d_ptr, int size,std::string dir, std::string fname){
    //load it into a new array
    std::ostringstream s;
    s<<dir<<fname;
    std::string fullFname= s.str();

    float* h_ptr=(float*)malloc(sizeof(float)*size);
    ck(cudaMemcpyAsync(h_ptr, d_ptr,sizeof(float)*size, cudaMemcpyDeviceToHost,stream));

    FILE * test=fopen(fullFname.c_str(), "r");
    if(test){
        fclose(test); 
        cnpy::NpyArray arr = cnpy::npy_load(fullFname);
        float* loaded_data = arr.data<float>();
        double err=0;
        double max=-1e30f;
        int loc_err=0;
        int i=0;
        for(;i<size;i++){
            double sub=abs(h_ptr[i]-loaded_data[i]);
            if(sub>err){
                err=sub;
                loc_err=i;
            }
            if(h_ptr[i]>max){
                max=h_ptr[i];
            }
        }
        if(i==size){
            std::cout<<stringPadding(fname,30)<<" ,Max Abs-Err: " << std::fixed << std::setw(11)<<err<<" ,Err Loc: "
                << std::fixed << std::setw(11)<<loc_err<<" ,Max Value: "
                << std::fixed << std::setw(11)<<max<<" ,Rel-Err: "
                << std::fixed << std::setw(11)<<err/max*100<<"%"<<std::endl;
        }
    }else{
        std::cout<<"Can not find file: "<<fullFname<<std::endl;
    }
    free(h_ptr);
}

void checkByNpy(cudaStream_t stream, __half* d_ptr, int size,std::string dir, std::string fname){
    std::ostringstream s;
    s<<dir<<fname;
    std::string fullFname= s.str();

    __half* h_ptr=(__half*)malloc(sizeof(__half)*size);
    ck(cudaMemcpyAsync(h_ptr, d_ptr,sizeof(float)*size, cudaMemcpyDeviceToHost,stream));

    FILE * test=fopen(fullFname.c_str(), "r");
    if(test){
        fclose(test); 

        //load it into a new array
        cnpy::NpyArray arr = cnpy::npy_load(fullFname);
        float* loaded_data = arr.data<float>();

        double max=-1e30f;
        double err=0;

        int loc_err=0;
        int i=0;
        for(;i<size;i++){
            double tmp=__half2float(h_ptr[i]);
            double sub=abs(tmp-loaded_data[i]);
            if(sub>err){
                err=sub;
                loc_err=i;
            }
            if(tmp>max){
                max=tmp;
            }
        }
        if(i==size){
            std::cout<<stringPadding(fname,30)<<" ,Max Abs-Err: " << std::fixed << std::setw(11)<<err<<" ,Err Loc: "
                << std::fixed << std::setw(11)<<loc_err<<" ,Max Value: "
                << std::fixed << std::setw(11)<<max<<" ,Rel-Err: "
                << std::fixed << std::setw(11)<<err/max*100<<"%"<<std::endl;
            //Filename, error, max error location, max value, relative error
        }
    }
    free(h_ptr);
}

template <typename T>
bool checkByNpz(cnpy::npz_t& data_npz,cudaStream_t stream,std::string name, T* d_ptr, int size){
    std::cout<<name<<" "<<size<<std::endl;
    bool ifCorrect=1;
    cnpy::NpyArray arr = data_npz[name];
    T* loaded_data = arr.data<T>();
 
    T * h_ptr=(T*)malloc(size*sizeof(T));
    ck(cudaMemcpyAsync(h_ptr, d_ptr,sizeof(T)*size, cudaMemcpyDeviceToHost,stream));

    double err=0;
    double max=castToFloat(h_ptr[0]);
    int i=0;

    for(i=0;i<size;i++){
        double sub=abs(castToFloat(h_ptr[i])-castToFloat(loaded_data[i]));
        if(sub>err){
            err=sub;
        }
        if(max<castToFloat(h_ptr[i])){
            max=castToFloat(h_ptr[i]);
        }
    }

    if(err/max>0.05){
        ifCorrect=0;
        std::cout<<"Wrong: "<< std::setw(20)<<name<<" Max err :"<<err <<" Max value :"<<max<<" Ralative error rate: "<< err/max <<std::endl;
    }else{
        ifCorrect=1;
        //std::cout<<"Correct: "<< std::setw(20)<<name<<" Max err :"<<err <<" Max value :"<<max<<" Ralative error rate: "<< err/max <<std::endl;
    }

    free(h_ptr);
    return ifCorrect;
}

void checkByNpz(cudaStream_t stream,string data_fname, string name, float* d_ptr, int size ){
    cnpy::npz_t data_npz=cnpy::npz_load(data_fname);
    cnpy::NpyArray arr = data_npz[name];
    float* loaded_data = arr.data<float>();
 
    float * h_ptr=(float*)malloc(size*sizeof(float));
    //ck(cudaMemcpy(h_ptr,d_ptr, sizeof(float)*size, cudaMemcpyDeviceToHost));
    ck(cudaMemcpyAsync(h_ptr, d_ptr,sizeof(float)*size, cudaMemcpyDeviceToHost,stream));

    double err=0;
    int i=0;

    for(i=0;i<size;i++){
        double sub=abs(h_ptr[i]-loaded_data[i]);
        if(sub>err){
            err=sub;
        }
        if(sub>0.01){
            std::cout<<data_fname<<" "<<name<<"  Got error at: "<<i<<" Calculated="<<h_ptr[i]<<" Ori="<<loaded_data[i]<<" Err: "<<sub<<std::endl;
            break;
        }
    }

    if(i==size){
        std::cout<<"Correct: "<< data_fname<<" Max err :"<<err<<std::endl;
    }

    free(h_ptr);
}

void checkByNpz(cudaStream_t stream,string data_fname, string name, __half* d_ptr, int size ){
    cnpy::npz_t data_npz=cnpy::npz_load(data_fname);
    cnpy::NpyArray arr = data_npz[name];
    float* loaded_data = arr.data<float>();
 
    __half * h_ptr=(__half*)malloc(size*sizeof(float));
    //ck(cudaMemcpy(h_ptr,d_ptr, sizeof(__half)*size, cudaMemcpyDeviceToHost));
    ck(cudaMemcpyAsync(h_ptr, d_ptr,sizeof(float)*size, cudaMemcpyDeviceToHost,stream));

    double err=0;
    int i=0;
    for(;i<size;i++){
        float tmp=__half2float(h_ptr[i]);
        double sub=abs(tmp-loaded_data[i]);
        if(sub>err){
            err=sub;
        }
        if(sub>10){
            std::cout<<data_fname<<"  Got error at: "<<i<<" value: calculated="<<tmp<<" tensorRT="<<loaded_data[i]<<" Err: "<<sub<<std::endl;
            break;
        }
    }

    if(i==size){
        std::cout<<"Correct: "<< data_fname<<" Max err :"<<err<<std::endl;
    }
    free(h_ptr);
}
/*********************The explicit instantiation part***********************/
template int numPerThread<float>();
template int numPerThread<__half>();


template float castToFloat<float>(float input);
template float castToFloat<__half>(__half input);

template  void deviceMalloc<float>(float** ptr, int size);
template  void deviceMemset<float>(float* ptr, int value, int size);
template  void deviceFree<float>(float* & ptr);
template  void deviceMemcpyHtoD<float>(cudaStream_t stream, float* d_ptr,float* h_ptr, int size);


template  void deviceMalloc<int>(int** ptr, int size);
template  void deviceMemset<int>(int* ptr, int value, int size);
template  void deviceFree<int>(int* & ptr);
template  void deviceMemcpyHtoD<int>(cudaStream_t stream, int* d_ptr,int* h_ptr, int size);


template  void deviceMalloc<__half>(__half** ptr, int size);
template  void deviceMemset<__half>(__half* ptr, int value, int  size);
template  void deviceFree<__half>(__half* & ptr);
template  void deviceMemcpyHtoD<__half>(cudaStream_t stream, __half* d_ptr,__half* h_ptr, int size);


template  void setByNpz<int>(cnpy::npz_t & my_npz, std::string name, int* h_ptr, int size, int offset);
template  void setByNpz<float>(cnpy::npz_t & my_npz, std::string name, float* h_ptr, int size, int offset);
template  void setByNpz<__half>(cnpy::npz_t & my_npz, std::string name, __half* h_ptr, int size, int offset);

template bool checkByNpz<float>(cnpy::npz_t& data_npz,cudaStream_t stream,std::string name, float* d_ptr, int size);
template bool checkByNpz<__half>(cnpy::npz_t& data_npz,cudaStream_t stream,std::string name, __half* d_ptr, int size);

