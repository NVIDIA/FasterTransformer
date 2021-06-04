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



#include "XlnetDebug.h"
/***********************Current varification version************************/
template <typename T>
bool XlnetDebug<T>::verifyPreProcess(cnpy::npz_t& data_npz){
    bool ifCorrect=1;
    ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,"attn_mask",this->attn_mask,this->batch_size*this->seq_len*this->seq_len);
    ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,"output_h",this->word_emb_k,this->batch_size*this->seq_len*this->hidden_dim);
    ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,"seg_mat",this->seg_mat,this->batch_size*this->seq_len*this->seq_len*2);
    ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,"pos_emb",this->attr_k_head_r,2*this->seq_len*this->hidden_dim);
    return ifCorrect;
}

template <typename T>
bool XlnetDebug<T>::verifyInter(cnpy::npz_t& data_npz,int i_layer){
    bool ifCorrect=1;
    std::string nameList[VERIFY_NUM]={"_q_head_h","_k_head_h","_v_head_h","_k_head_r", "_attn_vec", "_attn_output","_layer_1"};
    int sizeList[VERIFY_NUM]={this->buf_size,this->buf_size,this->buf_size,this->seq_len*2*this->hidden_dim,this->buf_size,
            this->buf_size,this->batch_size*this->seq_len*this->hidden_dim_ff};

    T* ptrList[VERIFY_NUM]={this->arr_xlnet_layer[i_layer].query_buf,this->arr_xlnet_layer[i_layer].key_buf,
        this->arr_xlnet_layer[i_layer].value_buf,this->arr_xlnet_layer[i_layer].k_head_r,
        this->arr_xlnet_layer[i_layer].attn_vec_trans, this->arr_xlnet_layer[i_layer].attn_layernorm,
        this->arr_xlnet_layer[i_layer].output_fc1};

    for(int i=0;i<VERIFY_NUM;i++){
        std::ostringstream s;
        s<<"layer_"<<i_layer<<nameList[i];
        std::string lname= s.str();
        //std::cout<<"Check Var:"<<lname<<std::endl;
        ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,lname,ptrList[i],sizeList[i]);
    }
    return ifCorrect;
}

template <typename T>
bool XlnetDebug<T>::verifyLayerRes(InputDataHost& input_data_host, std::string output_file){
    bool ifCorrect = 1;

    cnpy::npz_t data_npz=cnpy::npz_load(output_file);

    this->setInput(input_data_host);
    this->preProcess();

    //ifCorrect=ifCorrect&verifyPreProcess(data_npz);

    T* input=this->word_emb_k;
    T* output=NULL;

    for(int i_layer=0;i_layer<this->num_layers;i_layer++){
        std::cout<<std::endl;
        output=this->arr_xlnet_layer[i_layer].forward(input,this->attn_mask,this->seg_mat,
                this->attr_k_head_r);
        input=output;

        std::ostringstream s;
        s<<"layer_"<<i_layer;
        std::string label= s.str();

        ifCorrect=ifCorrect&verifyInter(data_npz,i_layer);
        ifCorrect=ifCorrect&checkByNpz(data_npz,this->stream,label,output,this->buf_size);
    }//end for

    return ifCorrect;
}


template <typename T>
float XlnetDebug<T>::profileOneLayer(int warm_up_time, int profile_run_time){
    T* input=this->word_emb_k;
    //Warm up
    for(int i=0;i<warm_up_time;i++){
         this->arr_xlnet_layer[0].forward(input,this->attn_mask,this->seg_mat,
                this->attr_k_head_r);
    }

    //Profile
    CudaTimer cuda_timer;
    cuda_timer.start();
    for(int i=0;i<profile_run_time;i++){
         this->arr_xlnet_layer[0].forward(input,this->attn_mask,this->seg_mat,
                this->attr_k_head_r);
    }

    float time=(cuda_timer.stop())/profile_run_time;
    return time;
}
/***********************Constructor & Deconstrctor************************/
template <typename T>
XlnetDebug<T>::XlnetDebug(cudaStream_t stream, cublasHandle_t cublas_handle,
        int num_layers, int batch_size, int seq_len, int head_num, int size_per_head,
        int hidden_dim,int hidden_dim_ff,int num_token,float epsilon,
        PreWeightHost<T> & pre_weight_host,
        std::vector<LayerWeightHost<T> >& arr_layer_weight_host,
        std::string gemm_file_name):
    Xlnet<T>(stream,cublas_handle,num_layers,batch_size,seq_len, 
        head_num,size_per_head,hidden_dim,hidden_dim_ff,num_token,epsilon,
        pre_weight_host,arr_layer_weight_host,gemm_file_name)

{
    //std::cout << "Object XlnetDebug is being created" <<std::endl;

    //set buf_size
    this->buf_size = this->batch_size * this->seq_len*this->head_num * this->size_per_head;
    this->qk_buf_size = this->batch_size * this->seq_len * this->head_num * this->seq_len;

    //report required device memory
    long long int total_mem=0;
    total_mem+=this->seq_len*2*this->hidden_dim;
    total_mem+=buf_size*3;

    total_mem+=buf_size;
    total_mem+=buf_size;
    total_mem+=qk_buf_size;

    total_mem+=buf_size;
    total_mem+=this->batch_size*this->seq_len*2*this->hidden_dim;
    total_mem+=this->batch_size*this->seq_len*head_num*this->seq_len*2;
    total_mem+=this->batch_size*this->seq_len*head_num*this->seq_len;

    total_mem+=buf_size;
    total_mem+=this->batch_size*2*this->hidden_dim;

    total_mem+=this->batch_size*head_num*this->seq_len*2;
    total_mem+=this->batch_size*head_num*this->seq_len*2;
    total_mem+=this->batch_size*head_num*this->seq_len*this->seq_len;
    total_mem+=this->batch_size*head_num*this->seq_len*this->seq_len;

    total_mem+=this->batch_size*head_num*this->seq_len*this->seq_len;
    total_mem+=buf_size;
    total_mem+=this->batch_size*head_num*this->seq_len*size_per_head;
    total_mem+=this->batch_size*head_num*this->seq_len*size_per_head;

    total_mem+=this->batch_size*this->hidden_dim*this->seq_len;
    total_mem+=this->batch_size*this->hidden_dim*this->seq_len;

    total_mem+=this->batch_size*this->seq_len*this->hidden_dim_ff;
    total_mem+=this->batch_size*this->seq_len*this->hidden_dim;
    total_mem+=this->batch_size*this->seq_len*this->hidden_dim;

    total_mem=total_mem*this->num_layers;

    total_mem+=(this->batch_size+2)*this->seq_len*this->hidden_dim;
    total_mem+=this->batch_size*this->seq_len*this->seq_len*2;

    std::cout<<"Device memory required: "<<float(total_mem*sizeof(T))/1024/1024/1024<<" GB"<<std::endl<<std::endl;
}



template <typename T>
XlnetDebug<T>::~XlnetDebug() {
    //std::cout << "Object XlnetDebug is being deleted" <<std::endl;
}


//The explicit instantiation part
template class XlnetDebug<__half>;
template class XlnetDebug<float>;

