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


#include "LoadData.h"

/**********************LoadDataInput***********************************/
// InputData
InputData::InputData(int batch_size, int seq_len){
    this->batch_size=batch_size;
    this->seq_len=seq_len;
}
 InputData::~InputData(){
 }

// InputDataHost
InputDataHost::InputDataHost(int batch_size, int seq_len):InputData(batch_size,seq_len){
    inp_k=new int[batch_size * seq_len];
    input_mask=new float[batch_size * seq_len];
    seg_id=new int[batch_size * seq_len];
}


InputDataHost::~InputDataHost(){
    delete []inp_k;
    delete []input_mask;
    delete []seg_id;
}

void InputDataHost::fillInputData(std::string file_name){
    cnpy::npz_t my_npz = cnpy::npz_load(file_name);
    setByNpz(my_npz, "input_ids:0", inp_k, batch_size * seq_len);
    setByNpz(my_npz, "input_mask:0", input_mask, batch_size * seq_len);
    setByNpz(my_npz, "segment_ids:0", seg_id, batch_size * seq_len);
}

// InputDataDevice
InputDataDevice::InputDataDevice(cudaStream_t stream,int batch_size, int seq_len):InputData(batch_size,seq_len){
    //std::cout<<"Construct InputDataDevice: "<<this->batch_size<<" "<<this->seq_len<<std::endl;
    this->stream=stream;
    deviceMalloc(&inp_k, batch_size * seq_len);
    deviceMalloc(&input_mask, batch_size * seq_len);
    deviceMalloc(&seg_id, batch_size * seq_len);
}


void InputDataDevice::copyFromHost(InputDataHost& inputDataHost){
    ck(cudaMemcpyAsync(inp_k, inputDataHost.inp_k,batch_size * seq_len*sizeof(int),cudaMemcpyHostToDevice,stream)); 
    ck(cudaMemcpyAsync(input_mask, inputDataHost.input_mask,batch_size * seq_len*sizeof(float),cudaMemcpyHostToDevice,stream)); 
    ck(cudaMemcpyAsync(seg_id, inputDataHost.seg_id,batch_size * seq_len*sizeof(int),cudaMemcpyHostToDevice,stream)); 
}


InputDataDevice::~InputDataDevice(){
    //std::cout << "Deconstruct InputDataDevice" <<std::endl;
    deviceFree(inp_k);
    deviceFree(input_mask);
    deviceFree(seg_id);
}

/*************************LoadDataPre********************************/

template <typename T>
PreWeight<T>::PreWeight(int hidden_dim, int num_token){
    this->num_token=num_token;
    this->hidden_dim=hidden_dim;
    this->params_word_emb_k=NULL;
}

template <typename T>
PreWeight<T>::~PreWeight(){

}

template <typename T>
PreWeightDevice<T>::PreWeightDevice<T>(cudaStream_t stream,int hidden_dim, int num_token):PreWeight<T>(hidden_dim,num_token){
    //std::cout<<"Construct PreWeightDevice: "<<this->hidden_dim<<std::endl;
    this->stream=stream;
    deviceMalloc(&(this->params_word_emb_k), num_token* hidden_dim);
}


template <typename T>
void PreWeightDevice<T>::copyFromHost(PreWeightHost<T>& preWeightDevice){
    deviceMemcpyHtoD(this->stream,this->params_word_emb_k,preWeightDevice.params_word_emb_k,
            this->num_token* this->hidden_dim);
}


template <typename T>
PreWeightDevice<T>::~PreWeightDevice<T>(){
    //std::cout << "Deconstruct PreWeightDevice" <<std::endl;
    deviceFree(this->params_word_emb_k);
}


template <typename T>
PreWeightHost<T>::PreWeightHost<T>(int hidden_dim, int num_token)
    :PreWeight<T>(hidden_dim,num_token){
        this->params_word_emb_k=new T[num_token* hidden_dim];
}


template <typename T>
void PreWeightHost<T>::fillPreWeight(std::string file_name){
    cnpy::npz_t params_npz= cnpy::npz_load(file_name);
    setByNpz(params_npz, "model/transformer/word_embedding/lookup_table:0", 
            this->params_word_emb_k, this->num_token* this->hidden_dim);
}

template <typename T>
PreWeightHost<T>::~PreWeightHost<T>(){
    delete []this->params_word_emb_k;
}

/**************************LayerWeight*******************************/

template <typename T>
LayerWeight<T>::LayerWeight(int hidden_dim, int hidden_dim_ff){
    this->hidden_dim=hidden_dim;
    this->hidden_dim_ff=hidden_dim_ff;
}

template <typename T>
LayerWeight<T>::~LayerWeight(){
}


template <typename T>
LayerWeightDevice<T>::LayerWeightDevice(cudaStream_t stream,int hidden_dim, int hidden_dim_ff):
    LayerWeight<T>(hidden_dim, hidden_dim_ff){

        //std::cout<<"Construct LayerWeightDevice: "<<this->hidden_dim<<" "<<this->hidden_dim_ff<<std::endl;
        this->stream=stream;
        deviceMalloc(&this->attr_kernel_QKV,3*hidden_dim * hidden_dim);
        this->attr_kernel_Q=this->attr_kernel_QKV;
        this->attr_kernel_K=this->attr_kernel_QKV+hidden_dim * hidden_dim;
        this->attr_kernel_V=this->attr_kernel_QKV+2*hidden_dim * hidden_dim;
        deviceMalloc(&this->attr_bias_Q_w, hidden_dim);
        deviceMalloc(&this->attr_pos_emb,hidden_dim*hidden_dim);
        deviceMalloc(&this->attr_bias_Q_r, hidden_dim);
        deviceMalloc(&this->attr_seg_embed,2*hidden_dim);
        deviceMalloc(&this->attr_bias_Q_s, hidden_dim);
        deviceMalloc(&this->attr_proj_o, hidden_dim*hidden_dim);
        deviceMalloc(&this->attr_layernorm_gamma, hidden_dim);
        deviceMalloc(&this->attr_layernorm_beta, hidden_dim);
        deviceMalloc(&this->attr_fc1_kernel,hidden_dim*hidden_dim_ff);
        deviceMalloc(&this->attr_fc1_bias, hidden_dim_ff);
        deviceMalloc(&this->attr_fc2_kernel,hidden_dim*hidden_dim_ff);
        deviceMalloc(&this->attr_fc2_bias, hidden_dim);
        deviceMalloc(&this->attr_ff_gamma, hidden_dim);
        deviceMalloc(&this->attr_ff_beta, hidden_dim);
    }

template <typename T>
LayerWeightDevice<T>::LayerWeightDevice(LayerWeightDevice<T> const& layer_weight_device):
    LayerWeight<T>(layer_weight_device.hidden_dim, layer_weight_device.hidden_dim_ff){
        //std::cout<<"COPY Constructor LayerWeightDevice Without Value COPY: "<<this->hidden_dim<<" "<<this->hidden_dim_ff<<std::endl;
        this->stream=layer_weight_device.stream;

        deviceMalloc(&this->attr_kernel_QKV,3*this->hidden_dim * this->hidden_dim);
        this->attr_kernel_Q=this->attr_kernel_QKV;
        this->attr_kernel_K=this->attr_kernel_QKV+this->hidden_dim * this->hidden_dim;
        this->attr_kernel_V=this->attr_kernel_QKV+2*this->hidden_dim * this->hidden_dim;
        deviceMalloc(&this->attr_bias_Q_w, this->hidden_dim);
        deviceMalloc(&this->attr_pos_emb,this->hidden_dim*this->hidden_dim);
        deviceMalloc(&this->attr_bias_Q_r, this->hidden_dim);
        deviceMalloc(&this->attr_seg_embed,2*this->hidden_dim);
        deviceMalloc(&this->attr_bias_Q_s, this->hidden_dim);
        deviceMalloc(&this->attr_proj_o, this->hidden_dim*this->hidden_dim);
        deviceMalloc(&this->attr_layernorm_gamma, this->hidden_dim);
        deviceMalloc(&this->attr_layernorm_beta, this->hidden_dim);
        deviceMalloc(&this->attr_fc1_kernel,this->hidden_dim*this->hidden_dim_ff);
        deviceMalloc(&this->attr_fc1_bias, this->hidden_dim_ff);
        deviceMalloc(&this->attr_fc2_kernel,this->hidden_dim*this->hidden_dim_ff);
        deviceMalloc(&this->attr_fc2_bias, this->hidden_dim);
        deviceMalloc(&this->attr_ff_gamma, this->hidden_dim);
        deviceMalloc(&this->attr_ff_beta, this->hidden_dim);


    }

template <typename T>
void LayerWeightDevice<T>::copyFromHost(LayerWeightHost<T>& layer_weight_host){
    deviceMemcpyHtoD(stream,this->attr_kernel_Q, layer_weight_host.attr_kernel_Q,this->hidden_dim* this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_kernel_K, layer_weight_host.attr_kernel_K,this->hidden_dim* this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_kernel_V, layer_weight_host.attr_kernel_V,this->hidden_dim* this->hidden_dim);

    deviceMemcpyHtoD(stream,this->attr_bias_Q_w,layer_weight_host.attr_bias_Q_w,  this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_pos_emb,layer_weight_host.attr_pos_emb,this->hidden_dim*this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_bias_Q_r,layer_weight_host.attr_bias_Q_r, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_seg_embed,layer_weight_host.attr_seg_embed,2*this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_bias_Q_s,layer_weight_host.attr_bias_Q_s, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_proj_o,layer_weight_host.attr_proj_o, this->hidden_dim*this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_layernorm_gamma, layer_weight_host.attr_layernorm_gamma, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_layernorm_beta, layer_weight_host.attr_layernorm_beta, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_fc1_kernel,layer_weight_host.attr_fc1_kernel,this->hidden_dim*this->hidden_dim_ff);
    deviceMemcpyHtoD(stream,this->attr_fc1_bias, layer_weight_host.attr_fc1_bias, this->hidden_dim_ff);
    deviceMemcpyHtoD(stream,this->attr_fc2_kernel,layer_weight_host.attr_fc2_kernel,this->hidden_dim*this->hidden_dim_ff);
    deviceMemcpyHtoD(stream,this->attr_fc2_bias,layer_weight_host.attr_fc2_bias, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_ff_gamma, layer_weight_host.attr_ff_gamma, this->hidden_dim);
    deviceMemcpyHtoD(stream,this->attr_ff_beta,layer_weight_host.attr_ff_beta, this->hidden_dim);

}
template <typename T>
LayerWeightDevice<T>::~LayerWeightDevice(){
    //std::cout<<"Deconstruct LayerWeightDevice"<<std::endl;

    deviceFree(this->attr_kernel_QKV);
    deviceFree(this->attr_bias_Q_w);
    deviceFree(this->attr_pos_emb);
    deviceFree(this->attr_bias_Q_r);
    deviceFree(this->attr_seg_embed);
    deviceFree(this->attr_bias_Q_s);
    deviceFree(this->attr_proj_o);
    deviceFree(this->attr_layernorm_gamma);
    deviceFree(this->attr_layernorm_beta);
    deviceFree(this->attr_fc1_kernel);
    deviceFree(this->attr_fc1_bias);
    deviceFree(this->attr_fc2_kernel);
    deviceFree(this->attr_fc2_bias);
    deviceFree(this->attr_ff_gamma);
    deviceFree(this->attr_ff_beta);

}

template <typename T>
LayerWeightHost<T>::LayerWeightHost(int hidden_dim, int hidden_dim_ff):
    LayerWeight<T>(hidden_dim, hidden_dim_ff){

        //std::cout<<"Constructor LayerWeightHost: "<<this->hidden_dim<<" "<<this->hidden_dim_ff<<std::endl;
        this->attr_kernel_Q=new T[hidden_dim* this->hidden_dim];
        this->attr_kernel_K=new T[hidden_dim* this->hidden_dim];
        this->attr_kernel_V=new T[hidden_dim* this->hidden_dim];

        this->attr_bias_Q_w=new T[hidden_dim];
        this->attr_pos_emb=new T[hidden_dim*hidden_dim];
        this->attr_bias_Q_r=new T[hidden_dim];
        this->attr_seg_embed=new T[hidden_dim*2];
        this->attr_bias_Q_s=new T[hidden_dim];
        this->attr_proj_o=new T[hidden_dim*hidden_dim];
        this->attr_layernorm_gamma=new T[hidden_dim];
        this->attr_layernorm_beta=new T[hidden_dim];
        this->attr_fc1_kernel=new T[hidden_dim*hidden_dim_ff];
        this->attr_fc1_bias=new T[hidden_dim_ff];
        this->attr_fc2_kernel=new T[hidden_dim*hidden_dim_ff];
        this->attr_fc2_bias=new T[hidden_dim];
        this->attr_ff_gamma=new T[hidden_dim];
        this->attr_ff_beta=new T[hidden_dim];
    }

template <typename T>
LayerWeightHost<T>::LayerWeightHost(LayerWeightHost<T> const& layer_weight_host):
    LayerWeight<T>(layer_weight_host.hidden_dim, layer_weight_host.hidden_dim_ff){
    //std::cout<<"COPY Constructor LayerWeightHost: "<<this->hidden_dim<<" "<<this->hidden_dim_ff<<std::endl;

    this->attr_kernel_Q=new T[this->hidden_dim* this->hidden_dim];
    this->attr_kernel_K=new T[this->hidden_dim* this->hidden_dim];
    this->attr_kernel_V=new T[this->hidden_dim* this->hidden_dim];

    this->attr_bias_Q_w=new T[this->hidden_dim];
    this->attr_pos_emb=new T[this->hidden_dim*this->hidden_dim];
    this->attr_bias_Q_r=new T[this->hidden_dim];
    this->attr_seg_embed=new T[this->hidden_dim*2];
    this->attr_bias_Q_s=new T[this->hidden_dim];
    this->attr_proj_o=new T[this->hidden_dim*this->hidden_dim];
    this->attr_layernorm_gamma=new T[this->hidden_dim];
    this->attr_layernorm_beta=new T[this->hidden_dim];
    this->attr_fc1_kernel=new T[this->hidden_dim*this->hidden_dim_ff];
    this->attr_fc1_bias=new T[this->hidden_dim_ff];
    this->attr_fc2_kernel=new T[this->hidden_dim*this->hidden_dim_ff];
    this->attr_fc2_bias=new T[this->hidden_dim];
    this->attr_ff_gamma=new T[this->hidden_dim];
    this->attr_ff_beta=new T[this->hidden_dim];

    memcpy(this->attr_kernel_Q, layer_weight_host.attr_kernel_Q,this->hidden_dim* this->hidden_dim*sizeof(T));
    memcpy(this->attr_kernel_K, layer_weight_host.attr_kernel_K,this->hidden_dim* this->hidden_dim*sizeof(T));
    memcpy(this->attr_kernel_V, layer_weight_host.attr_kernel_V,this->hidden_dim* this->hidden_dim*sizeof(T));

    memcpy(this->attr_bias_Q_w,layer_weight_host.attr_bias_Q_w,  this->hidden_dim*sizeof(T));
    memcpy(this->attr_pos_emb,layer_weight_host.attr_pos_emb,this->hidden_dim*this->hidden_dim*sizeof(T));
    memcpy(this->attr_bias_Q_r,layer_weight_host.attr_bias_Q_r, this->hidden_dim*sizeof(T));
    memcpy(this->attr_seg_embed,layer_weight_host.attr_seg_embed,2*this->hidden_dim*sizeof(T));
    memcpy(this->attr_bias_Q_s,layer_weight_host.attr_bias_Q_s, this->hidden_dim*sizeof(T));
    memcpy(this->attr_proj_o,layer_weight_host.attr_proj_o, this->hidden_dim*this->hidden_dim*sizeof(T));
    memcpy(this->attr_layernorm_gamma, layer_weight_host.attr_layernorm_gamma, this->hidden_dim*sizeof(T));
    memcpy(this->attr_layernorm_beta, layer_weight_host.attr_layernorm_beta, this->hidden_dim*sizeof(T));
    memcpy(this->attr_fc1_kernel,layer_weight_host.attr_fc1_kernel,this->hidden_dim*this->hidden_dim_ff*sizeof(T));
    memcpy(this->attr_fc1_bias, layer_weight_host.attr_fc1_bias, this->hidden_dim_ff*sizeof(T));
    memcpy(this->attr_fc2_kernel,layer_weight_host.attr_fc2_kernel,this->hidden_dim*this->hidden_dim_ff*sizeof(T));
    memcpy(this->attr_fc2_bias,layer_weight_host.attr_fc2_bias, this->hidden_dim*sizeof(T));
    memcpy(this->attr_ff_gamma, layer_weight_host.attr_ff_gamma, this->hidden_dim*sizeof(T));
    memcpy(this->attr_ff_beta,layer_weight_host.attr_ff_beta, this->hidden_dim*sizeof(T));
}

template <typename T>
void LayerWeightHost<T>::fillLayerWeight(int i_layer,std::string file_name){
    cnpy::npz_t params_npz= cnpy::npz_load(file_name);
    std::string str;

    str=paraName(i_layer, "/rel_attn/q/kernel:0");
    setByNpz(params_npz, str, this->attr_kernel_Q, this->hidden_dim * this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/k/kernel:0");
    setByNpz(params_npz, str, this->attr_kernel_K, this->hidden_dim * this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/v/kernel:0");
    setByNpz(params_npz, str, this->attr_kernel_V, this->hidden_dim * this->hidden_dim);

    str=paraName("model/transformer/r_w_bias:0");
    setByNpz(params_npz, str, this->attr_bias_Q_w, this->hidden_dim, i_layer*this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/r/kernel:0");
    setByNpz(params_npz, str, this->attr_pos_emb,this->hidden_dim*this->hidden_dim);

    str=paraName("model/transformer/r_r_bias:0");
    setByNpz(params_npz, str, this->attr_bias_Q_r, this->hidden_dim, i_layer*this->hidden_dim);

    str=paraName("model/transformer/seg_embed:0");
    setByNpz(params_npz, str, this->attr_seg_embed, 2*this->hidden_dim, i_layer*2*this->hidden_dim);

    str=paraName("model/transformer/r_s_bias:0");
    setByNpz(params_npz, str, this->attr_bias_Q_s,this->hidden_dim, i_layer*this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/o/kernel:0");
    setByNpz(params_npz, str,this->attr_proj_o ,this->hidden_dim*this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/LayerNorm/gamma:0");
    setByNpz(params_npz, str,this->attr_layernorm_gamma, this->hidden_dim);

    str=paraName(i_layer, "/rel_attn/LayerNorm/beta:0");
    setByNpz(params_npz, str,this->attr_layernorm_beta, this->hidden_dim);

    str=paraName(i_layer, "/ff/layer_1/kernel:0");
    setByNpz(params_npz, str,this->attr_fc1_kernel, this->hidden_dim*this->hidden_dim_ff);

    str=paraName(i_layer, "/ff/layer_1/bias:0");
    setByNpz(params_npz, str,this->attr_fc1_bias,this->hidden_dim_ff);

    str=paraName(i_layer, "/ff/layer_2/kernel:0");
    setByNpz(params_npz, str,this->attr_fc2_kernel, this->hidden_dim*this->hidden_dim_ff);

    str=paraName(i_layer, "/ff/layer_2/bias:0");
    setByNpz(params_npz, str,this->attr_fc2_bias, this->hidden_dim);

    str=paraName(i_layer, "/ff/LayerNorm/gamma:0");
    setByNpz(params_npz, str,this->attr_ff_gamma, this->hidden_dim);

    str=paraName(i_layer, "/ff/LayerNorm/beta:0");
    setByNpz(params_npz, str,this->attr_ff_beta, this->hidden_dim);
}

template <typename T>
LayerWeightHost<T>::~LayerWeightHost(){
    delete []this->attr_kernel_Q;
    delete []this->attr_kernel_K;
    delete []this->attr_kernel_V;

    delete []this->attr_bias_Q_w;
    delete []this->attr_pos_emb;
    delete []this->attr_bias_Q_r;
    delete []this->attr_seg_embed;
    delete []this->attr_bias_Q_s;
    delete []this->attr_proj_o;
    delete []this->attr_layernorm_gamma;
    delete []this->attr_layernorm_beta;
    delete []this->attr_fc1_kernel;
    delete []this->attr_fc1_bias;
    delete []this->attr_fc2_kernel;
    delete []this->attr_fc2_bias;
    delete []this->attr_ff_gamma;
    delete []this->attr_ff_beta;

}



//The explicit instantiation part
template class PreWeightDevice<__half>; 
template class PreWeightDevice<float>;

template class PreWeightHost<__half>; 
template class PreWeightHost<float>;

template class LayerWeightDevice<__half>; 
template class LayerWeightDevice<float>;

template class LayerWeightHost<__half>; 
template class LayerWeightHost<float>;


