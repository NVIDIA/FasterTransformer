#include "Xlnet.h"
/*************Device Function**************/
template<typename T> T __device__ cast(float v){
    return (T)v;
}

template<>
__half  __device__ cast(float v){
    return __float2half(v);
}


/********************** Kernels ************************/
template<typename T>
void __global__ getWordEmdK(T* word_emb_k, T* params_word_emb_k, int * inp_k, 
        int seq_len, int hidden_dim){

    int col=threadIdx.x;// the index of column
    int row=blockIdx.y;//the index of row
    int batch=blockIdx.x;// the index of batch 

    int index=inp_k[batch*seq_len+row];
    T data=params_word_emb_k[index*hidden_dim+col];

    word_emb_k[batch*seq_len*hidden_dim+row*hidden_dim+col]=data;
} 

//    dim3 grid(batch_size, seq_len);
//    getWordEmdK<<<grid, hidden_dim/2,0, stream>>>(word_emb_k, params_word_emb_k, inp_k, seq_len, hidden_dim);
template<>
void __global__ getWordEmdK(__half* word_emb_k, __half* params_word_emb_k, int * inp_k, 
        int seq_len, int hidden_dim){
    int col=threadIdx.x;// the index of column
    int row=blockIdx.y;//the index of row
    int batch=blockIdx.x;// the index of batch 

    int index=__ldg(inp_k+batch*seq_len+row);
    half2 data=((half2*)params_word_emb_k)[(index*hidden_dim+col*2)>>1];

    ((half2*)word_emb_k)[(batch*seq_len*hidden_dim+row*hidden_dim+col*2)>>1]=data;
} 


template<typename T>
void __global__ getAttnMask(T* attn_mask, float* input_mask, int seq_len){
    int col=threadIdx.x;
    int row=blockIdx.y;
    int batch=blockIdx.x;

    float data=1;
    if(col==row){
        data=0;
    }
    float mask=input_mask[batch*seq_len+col];
    attn_mask[batch*seq_len*seq_len+row*seq_len+col]=cast<T>(data*mask);
}

template<>
void __global__ getAttnMask(__half* attn_mask, float* input_mask, int seq_len){
    int in_index=blockIdx.y*blockDim.x+threadIdx.x;
    int col=in_index%(seq_len/2)*2;
    int row=in_index/(seq_len/2);
    int batch=blockIdx.x;

    float2 tmp;
    if(row<seq_len&&col<seq_len-1){
        float data=1;
        if(col==row){
            data=0;
        }
        tmp.x=input_mask[batch*seq_len+col]*data;

        col+=1;
        data=1;
        if(col==row){
            data=0;
        }
        tmp.y=input_mask[batch*seq_len+col]*data;

        int out_index=(batch*seq_len*seq_len+row*seq_len+col)>>1;
        ((half2*)attn_mask)[out_index]=__float22half2_rn(tmp);
    }
}


template<typename T>
void __global__ getSegMat(T* seg_mat, int* seg_id, int seq_len){
    int col=threadIdx.x;
    int row=blockIdx.y;
    int batch=blockIdx.x;

    int w[4]={0,1,1,0};
    int d1=seg_id[batch*seq_len+col];
    int d2=seg_id[batch*seq_len+row];
    int d=0; 

    d=int(floor(exp(-1*abs(double(d1-d2)))));

    int index=batch*seq_len*seq_len+row*seq_len+col;
    seg_mat[index*2]=w[d*2+0];
    seg_mat[index*2+1]=w[d*2+1];

}

template<>
void __global__ getSegMat(__half* seg_mat, int* seg_id, int seq_len){
    int col=threadIdx.x;
    int row=blockIdx.y;
    int batch=blockIdx.x;

    int w[4]={0,1,1,0};
    int d1=seg_id[batch*seq_len+col];
    int d2=seg_id[batch*seq_len+row];
    int d=0; 

    d=int(floor(exp(-1*abs(double(d1-d2)))));

    int index=batch*seq_len*seq_len+row*seq_len+col;
    float2 tmp_w;
    tmp_w.x=w[d*2+0];
    tmp_w.y=w[d*2+1];

    ((half2*)seg_mat)[index]=__float22half2_rn(tmp_w);
    //seg_mat[index*2]=__int2half_rn(w[d*2+0]);
    //seg_mat[index*2+1]=__int2half_rn(w[d*2+1]);

}

template<typename T>
void __global__  relativePosition(T* attr_k_head_r, int hidden_dim, int seq_len){
    int row=blockIdx.x;//(0,256)
    int col=threadIdx.x;//(0,384)

    float freq_seq=col*2;
    float inv_freq=1/(pow(10000, freq_seq/(hidden_dim)));

    float fwd_pos_seq=seq_len-row;

    float pos_emd=inv_freq*fwd_pos_seq;
    float s=sinf(pos_emd);
    float c=cosf(pos_emd);

    attr_k_head_r[row*hidden_dim+col]=cast<T>(s);
    attr_k_head_r[row*hidden_dim+hidden_dim/2+col]=cast<T>(c);

}

/***********************Input Process************************/
template<typename T>
void Xlnet<T>::setInput(InputDataHost & input_data_host){
    input_data_device.copyFromHost(input_data_host);
}

/***********************Pre-Process************************/

template<>
void Xlnet<float>::blockAttnMask(dim3 &grid, dim3& block){
    grid.x=batch_size;
    grid.y=seq_len;
    block.x=seq_len;
}

template<>
void Xlnet<__half>::blockAttnMask(dim3 &grid, dim3 &block){
    int numThreads=512;
    int numBlocky=(seq_len*seq_len/2-1)/numThreads+1;
    grid.x=batch_size;
    grid.y=numBlocky;
    block.x=numThreads;
}

template<typename T>
void Xlnet<T>::preProcess(){
    dim3 grid_word_emd_k(batch_size, seq_len);
    dim3 block_word_emd_k(hidden_dim/numPerThread<T>());
    
    getWordEmdK<<<grid_word_emd_k, block_word_emd_k,0, stream>>>(word_emb_k, 
            pre_weight_device.params_word_emb_k,
           input_data_device.inp_k, seq_len, hidden_dim);

    dim3 grid_attn_mask;
    dim3 block_attn_mask;
    blockAttnMask(grid_attn_mask,block_attn_mask);
    getAttnMask<<<grid_attn_mask,block_attn_mask,0,  stream>>>(attn_mask,input_data_device.input_mask, seq_len);


    dim3 grid_seg_mat(batch_size,seq_len);
    dim3 block_seg_mat(seq_len);
    getSegMat<<<grid_seg_mat, block_seg_mat,0, stream>>>(seg_mat, input_data_device.seg_id, seq_len);


    //relative_positional_encoding
    dim3 grid_rel_position(seq_len*2);
    dim3 block_rel_position(hidden_dim/2);
    relativePosition<<<grid_rel_position,block_rel_position,0,stream>>>(attr_k_head_r,hidden_dim,seq_len);


}

/********************************Attention*********************************/
template<typename T>
void Xlnet<T>::runAttentionLayers(){
    T* input=word_emb_k;
    T* output=NULL;
    for(int i_layer=0;i_layer<num_layers;i_layer++){
        output=arr_xlnet_layer[i_layer].forward(input,attn_mask,seg_mat,attr_k_head_r);
        input=output;
    }//end for
}
/***********************Constructor & Deconstrctor************************/

template <typename T>
Xlnet<T>::Xlnet(cudaStream_t stream, cublasHandle_t cublas_handle,
        int num_layers, int batch_size, int seq_len, int head_num, int size_per_head, 
        int hidden_dim, int hidden_dim_ff,int num_token, float epsilon,
        PreWeightHost<T> & pre_weight_host,
        std::vector<LayerWeightHost<T> > & arr_layer_weight_host,
        std::string gemm_file_name):
    stream(stream),
    cublas_handle(cublas_handle),
    num_layers(num_layers),
    batch_size(batch_size),
    seq_len(seq_len),
    head_num(head_num),
    size_per_head(size_per_head),
    hidden_dim(hidden_dim),
    hidden_dim_ff(hidden_dim_ff),
    epsilon(epsilon),
    input_data_device(stream,batch_size,seq_len),
    pre_weight_device(stream,hidden_dim,num_token),
    arr_xlnet_layer(num_layers, XlnetLayer<T>(batch_size,seq_len,head_num,
                size_per_head,hidden_dim,hidden_dim_ff,epsilon,stream,
                cublas_handle,gemm_file_name))
{
    // Set metadata
    //std::cout<<"Construct XLNet "<<std::endl; 
    //Preprocess
    deviceMalloc(&word_emb_k, batch_size*seq_len*hidden_dim);
    deviceMalloc(&attn_mask, batch_size*seq_len*seq_len);
    deviceMalloc(&seg_mat, batch_size*seq_len*seq_len*2);
    deviceMalloc(&attr_k_head_r, seq_len*2*hidden_dim);

    //Set weight
    pre_weight_device.copyFromHost(pre_weight_host);

    //Create layers
    for(int i=0;i<num_layers;i++){
        //std::cout<<"SET WEIGHT "<<i<<std::endl;
        arr_xlnet_layer[i].setLayerWeight(arr_layer_weight_host[i]);
    }
}


template <typename T>
void Xlnet<T>::run(InputDataHost& input_data_host){
    setInput(input_data_host);
    preProcess();
    runAttentionLayers();
}


template <typename T>
Xlnet<T>::~Xlnet() {
    //std::cout << "Deconstruct Xlnet" <<std::endl;
    deviceFree(word_emb_k);
    deviceFree(attn_mask);
    deviceFree(seg_mat);
    deviceFree(attr_k_head_r);

}


//The explicit instantiation part
template class Xlnet<__half>; 
template class Xlnet<float>;

