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



#include "XlnetLayer.h"


/********************** Kernel Configuration ************************/
template<>
void XlnetLayer<float>::blockRelShiftBd(dim3 &grid, dim3& block){
    grid.x=batch_size;
    grid.y=head_num;
    grid.z=seq_len;

    block.x=seq_len*2;
}
template<>
void XlnetLayer<__half>::blockRelShiftBd(dim3 &grid, dim3& block){
    int threads=512;
    int seq_dim1=threads/seq_len;
    int seq_dim2=seq_len/seq_dim1;

    grid.x=batch_size;
    grid.y=head_num;
    grid.z=seq_dim2;

    block.x=seq_dim1;
    block.y=seq_len;
}

/********************** Kernel Invocation ************************/
template<typename T>
void XlnetLayer<T>::oneToManyCublasGemm(T * d_A, T* d_B, T* d_C,cublasOperation_t transa, cublasOperation_t transb,
        int v_m, int v_n, int v_k,int lda, int strideA,
        int ldb,int strideB, int ldc,int strideC, int batch,int algo, cublasFunction method){
    int m;
    int n;
    int k;

    switch(method){
        case GEMM_STRIDE: 
            ck(cublasGemmStridedBatchedEx(cublas_handle, 
                        transa, transb, 
                        v_m, v_n, v_k, 
                        &alpha, 
                        d_A, a_type, lda,strideA, 
                        d_B, b_type, ldb,strideB, 
                        &beta, 
                        d_C, c_type, ldc,strideC, 
                        batch,
                        compute_type, 
                        static_cast<cublasGemmAlgo_t>(algo)));
            break;
        case GEMM_A_0: 
            m=v_m;
            n=v_n*batch;
            k=v_k;

            ck(cublasGemmEx(cublas_handle, 
                        transa, transb, 
                        m,n,k, 
                        &alpha, 
                        d_A, a_type, lda, 
                        d_B, b_type, ldb, 
                        &beta, 
                        d_C, c_type, ldc, 
                        compute_type, 
                        static_cast<cublasGemmAlgo_t>(algo)));
            break;

        case GEMM_B_0: 
            for(int count=0;count<batch;count++){
                ck(cublasGemmEx(cublas_handle, 
                            transa, transb, 
                            v_m,v_n,v_k, 
                            &alpha, 
                            d_A+strideA*count, a_type, lda, 
                            d_B+strideB*count, b_type, ldb, 
                            &beta, 
                            d_C+strideC*count, c_type, ldc, 
                            compute_type, 
                            static_cast<cublasGemmAlgo_t>(algo)));
            }
            break;

    }
}//end func

template<typename T>
void XlnetLayer<T>::invokePrepareMatrixes(){
    int off0=seq_len*hidden_dim;//seq_len*head_num*size_per_head
    int i_off1=hidden_dim;//head_num*size_per_head
    int o_off1=seq_len*size_per_head;
    int off2=size_per_head;

    dim3 grid(seq_len,batch_size);
    dim3 block(next_pow2(hidden_dim)/numPerThread<T>());
    prepareMatrixes<<<grid, block,0, stream>>>(
            q_buf,q_buf_bd,q_buf_ef,k_buf,k_buf_bd,k_buf_ef,
            query_buf, key_buf, k_head_r,layer_weight_device.attr_seg_embed,
            layer_weight_device.attr_bias_Q_w,layer_weight_device.attr_bias_Q_r,
            layer_weight_device.attr_bias_Q_s,
            off0,i_off1,o_off1,off2);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());
}

template<typename T>
void XlnetLayer<T>::invokeTranspose102(){
    dim3 grid_trans_v(batch_size, head_num);

    //dim3 block_trans_v(seq_len,2);//float
    //dim3 block_trans_v(seq_len);//__half
    dim3 block_trans_v(seq_len,2/(numPerThread<T>()));

    int toff0=head_num*seq_len*2;
    int ti_off1=seq_len*2;
    int to_off1=head_num*2;
    int toff2=2;

    transpose102<<<grid_trans_v, block_trans_v,0,stream>>>(qk_buf_ef_trans, qk_buf_ef,toff0,
            ti_off1,to_off1,toff2);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}

template<typename T>
void XlnetLayer<T>::invokeTranspose201(){
    dim3 grid_trans2(batch_size, seq_len);
    dim3 block_trans2(seq_len);
    int t2_off0=seq_len*seq_len*head_num;
    int t2_i_off1=seq_len*head_num;
    int t2_o_off1=seq_len*seq_len;
    int t2_i_off2=head_num;
    int t2_o_off2=seq_len;

    transpose201<<<grid_trans2, block_trans2,seq_len*(head_num+1)*sizeof(float),stream>>>
        (qk_buf_ef_seg_trans, qk_buf_ef_seg, t2_off0, t2_i_off1,t2_i_off2,t2_o_off1,t2_o_off2);

    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());
}
template<typename T>
void XlnetLayer<T>::invokeRelShiftBd(){
    dim3 grid_shift;
    dim3 block_shift;
    blockRelShiftBd(grid_shift,block_shift);

    int off0=head_num*seq_len*seq_len;
    int off1=seq_len*seq_len;

    relShiftBd<<<grid_shift, block_shift,0, stream>>>(qk_buf_bd_shift, qk_buf_bd,off0,off1,seq_len);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}


template<>
void XlnetLayer<float>::invokeCalAttnScore(float* attn_mask){
    int off0=head_num*seq_len*seq_len;
    int off1=seq_len*seq_len;
    float p=(1/(pow(size_per_head,0.5)));

    int voff0=head_num*seq_len*size_per_head;
    int v_o_off1=seq_len*size_per_head;
    int voff2=size_per_head;
    int v_i_off1=head_num*size_per_head;


    dim3 grid_score(batch_size,head_num,seq_len);
    dim3 block_score(next_pow2(seq_len));
    calAttnScore_valueBuf<<<grid_score, block_score,0, stream>>>(attn_score, qk_buf, qk_buf_bd_shift, 
            qk_buf_ef_seg_trans,attn_mask, off0, off1,seq_len,p,
            value_buf_trans, value_buf,voff0, v_i_off1, v_o_off1, voff2);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}

template<>
void XlnetLayer<__half>::invokeCalAttnScore(__half* attn_mask){
    int off0=head_num*seq_len*seq_len;
    int off1=seq_len*seq_len;
    float p=1/(pow(size_per_head,0.5));

    int voff0=head_num*seq_len*size_per_head;
    int v_o_off1=seq_len*size_per_head;
    int voff2=size_per_head;
    int v_i_off1=head_num*size_per_head;
    if(seq_len<=32){
        dim3 grid_score(batch_size,head_num,2);
        dim3 block_score(seq_len/2*next_pow2(seq_len/2));

        calAttnScore_valueBuf_small<<<grid_score, block_score,0, stream>>>(attn_score, 
                qk_buf, qk_buf_bd_shift, qk_buf_ef_seg_trans,attn_mask, 
                off0, off1,seq_len,seq_len/2, p,
                value_buf_trans, value_buf,voff0, v_i_off1, v_o_off1, voff2);

    }else if(seq_len<=64){
        dim3 grid_score(batch_size,head_num,seq_len/2);
        dim3 block_score(2*next_pow2(seq_len/2));

        calAttnScore_valueBuf_small<<<grid_score, block_score,0, stream>>>(attn_score, qk_buf, qk_buf_bd_shift, 
                qk_buf_ef_seg_trans,attn_mask, 
                off0, off1,seq_len,2,p,
                value_buf_trans, value_buf,voff0, v_i_off1, v_o_off1, voff2);
    }else{
        dim3 grid_score(batch_size,head_num,seq_len);
        dim3 block_score(next_pow2(seq_len/2));
        calAttnScore_valueBuf_large<<<grid_score, block_score,0, stream>>>(attn_score, 
                qk_buf, qk_buf_bd_shift, qk_buf_ef_seg_trans,attn_mask, 
                off0, off1,seq_len,p,
                value_buf_trans, value_buf,voff0, v_i_off1, v_o_off1, voff2);
    }
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}

template<typename T>
void XlnetLayer<T>::invokeTranspose102v2(){
    dim3 grid_trans_v(batch_size,seq_len);
    dim3 block_trans_v(head_num*size_per_head/numPerThread<T>());

    int off0=head_num*seq_len*size_per_head;
    int i_off1=seq_len*size_per_head;
    int o_off1=head_num*size_per_head;
    int off2=size_per_head;

    transpose102_v2<<<grid_trans_v, block_trans_v,0,stream>>>(attn_vec_trans, attn_vec,off0, i_off1, o_off1, off2);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}
    template<typename T>
void XlnetLayer<T>::invokeLayerNorm()
{
    dim3 grid(batch_size*seq_len);
    dim3 block(hidden_dim/numPerThread<T>());
    assert(block.x <= 1024);
    addBias_layerNorm<T><<<grid, block, 0, stream>>>(attn_layernorm,attn_out,to_tensor,
            layer_weight_device.attr_layernorm_gamma,layer_weight_device.attr_layernorm_beta,
            batch_size*seq_len, hidden_dim, epsilon);
    ck(cudaDeviceSynchronize());
    ck(cudaGetLastError());

}
template<typename T>
void XlnetLayer<T>::invokeGelu(){
    dim3 block(1024/numPerThread<T>());
    dim3 grid(batch_size, seq_len);
    gelu_bias_loop<<<grid, block, 0, stream>>>(output_fc1, layer_weight_device.attr_fc1_bias, hidden_dim_ff,seq_len); 

}

//New LayerNorm
    template<typename T>
void XlnetLayer<T>::invokeLayerNormv2()
{
    dim3 grid(batch_size*seq_len);
    dim3 block(hidden_dim/numPerThread<T>());
    assert(block.x <= 1024);
    addBias_layerNorm2<T><<<grid, block, 0, stream>>>(output_layernorm, output_fc2,attn_layernorm,
            layer_weight_device.attr_fc2_bias,layer_weight_device.attr_ff_gamma,
            layer_weight_device.attr_ff_beta,batch_size*seq_len, hidden_dim, epsilon);
}

/********************** Attention ************************/
template<typename T>
T* XlnetLayer<T>::forward(T* to_tensor,T* attn_mask,T* seg_mat,T* attr_k_head_r){
    this->to_tensor=to_tensor;

    oneToManyCublasGemm(layer_weight_device.attr_kernel_QKV,to_tensor,qkv_buf,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim, batch_size*seq_len, hidden_dim,
            hidden_dim,hidden_dim*hidden_dim,hidden_dim, 0,
            hidden_dim,buf_size,3,cublas_algo[0],(cublasFunction)cublas_func[0]);


    ck(cublasGemmEx(cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                hidden_dim, seq_len*2,hidden_dim,
                &alpha,
                layer_weight_device.attr_pos_emb, a_type, hidden_dim, 
                attr_k_head_r, b_type, hidden_dim, 
                &beta, 
                k_head_r, c_type, hidden_dim, 
                compute_type, 
                static_cast<cublasGemmAlgo_t>(cublas_algo[2])));

    //rel_attn_core: content, position, segment based attention score
    invokePrepareMatrixes();

    //ac = build_block_multiply_heads(network, bag, q_head_h, k_head_h, i_layer, 'w')
    ck(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, size_per_head,
                &alpha,
                k_buf, a_type, size_per_head, seq_len * size_per_head,
                q_buf, b_type, size_per_head, seq_len * size_per_head,
                &beta,
                qk_buf, c_type, seq_len, seq_len * seq_len,
                batch_size * head_num,
                compute_type,
                static_cast<cublasGemmAlgo_t>(cublas_algo[1])));

    //bd = build_block_multiply_heads(network, bag, q_head_h, k_head_r, i_layer, 'r')
    ck(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len*2, seq_len, size_per_head,
                &alpha,
                k_buf_bd, a_type, size_per_head, seq_len *2* size_per_head,
                q_buf_bd, b_type, size_per_head, seq_len *size_per_head,
                &beta,
                qk_buf_bd, c_type, seq_len*2, seq_len * seq_len*2,
                batch_size * head_num,
                compute_type,
                static_cast<cublasGemmAlgo_t>(cublas_algo[3])));

    //ef = build_block_multiply_heads(network, bag, q_head_h, seg_embed, i_layer, 's')
    //ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
    ck(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                2, seq_len, size_per_head,
                &alpha,
                k_buf_ef, a_type, size_per_head,2*size_per_head,
                q_buf_ef, b_type, size_per_head, seq_len *size_per_head,
                &beta,
                qk_buf_ef, c_type, 2, seq_len*2,
                batch_size * head_num,
                compute_type,
                static_cast<cublasGemmAlgo_t>(cublas_algo[4])));

    invokeTranspose102();

    ck(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                head_num, seq_len, 2,
                &alpha,
                qk_buf_ef_trans, a_type, 2, 2* head_num,
                seg_mat, b_type, 2, seq_len *2,
                &beta,
                qk_buf_ef_seg, c_type, head_num, seq_len*head_num,
                batch_size * seq_len,
                compute_type,
                static_cast<cublasGemmAlgo_t>(cublas_algo[5])));

    invokeTranspose201();

    //shift bd
    invokeRelShiftBd();

    //attention output,merge attention scores and perform masking
    //value_buf_trans=trans102(value_buf)
    invokeCalAttnScore(attn_mask);

    //attn_vec=value_buf_trans*attn_score
    ck(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                size_per_head, seq_len, seq_len,
                &alpha,
                value_buf_trans, a_type, size_per_head, seq_len* size_per_head,
                attn_score, b_type, seq_len, seq_len*seq_len,
                &beta,
                attn_vec, c_type, size_per_head, seq_len*size_per_head,
                batch_size * head_num,
                compute_type,
                static_cast<cublasGemmAlgo_t>(cublas_algo[6])));

    //attn_vec_trans=trans102(attn_vec)
    invokeTranspose102v2();

    //attn_out=attn_vec_trans* (attr_proj_o)T 
    oneToManyCublasGemm(layer_weight_device.attr_proj_o, attn_vec_trans, attn_out,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, seq_len,hidden_dim,
            hidden_dim, 0,
            hidden_dim, seq_len*hidden_dim,
            hidden_dim, seq_len*hidden_dim,
            batch_size,cublas_algo[7], (cublasFunction)cublas_func[7]);  

    invokeLayerNorm();

    oneToManyCublasGemm(layer_weight_device.attr_fc1_kernel,attn_layernorm,output_fc1,CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim_ff, seq_len,hidden_dim,hidden_dim_ff, 0,
            hidden_dim, seq_len*hidden_dim,hidden_dim_ff, seq_len*hidden_dim_ff,
            batch_size,cublas_algo[8],(cublasFunction)cublas_func[8]);


    invokeGelu();
    oneToManyCublasGemm(layer_weight_device.attr_fc2_kernel, output_fc1,output_fc2,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim, seq_len,hidden_dim_ff,hidden_dim, 0,hidden_dim_ff, 
            seq_len*hidden_dim_ff,hidden_dim, seq_len*hidden_dim,
            batch_size,cublas_algo[9],(cublasFunction)cublas_func[9]);

    invokeLayerNormv2();

    return output_layernorm;
}
/********************** Cublas Related Functions ************************/

template<>
void XlnetLayer<float>::setCublas(){
    a_type=CUDA_R_32F; 
    b_type=CUDA_R_32F; 
    c_type=CUDA_R_32F; 
    compute_type=CUDA_R_32F;

    start_algo = (int)CUBLAS_GEMM_DEFAULT;
    end_algo = (int)CUBLAS_GEMM_ALGO23;

    alpha=1.0f;
    beta=0.0f;

    for(int i=0;i<10;i++){
        cublas_algo[i] = -1;
        cublas_func[i] =0;
    }

}

template<>
void XlnetLayer<__half>::setCublas(){
    a_type=CUDA_R_16F; 
    b_type=CUDA_R_16F; 
    c_type=CUDA_R_16F; 
    compute_type=CUDA_R_16F;

    start_algo=(int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    end_algo=(int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    alpha=__float2half_rn(1.0f);
    beta=__float2half_rn(0.0f);

    for(int i=0;i<10;i++){
        cublas_algo[i] = 99;
        cublas_func[i] =0;
    }
}
template <typename T>
float  XlnetLayer<T>:: profileCublasGemmEx(cublasOperation_t transa, cublasOperation_t transb,
        int v_m, int v_n, int v_k,int lda,int ldb, int ldc,int ites, int index){
    int fast_algo = 0;
    T* d_A;
    T* d_B;
    T* d_C;
    ck(cudaMalloc((void**)&d_A, sizeof(T) * v_m * v_k));
    ck(cudaMalloc((void**)&d_B, sizeof(T) * v_k * v_n));
    ck(cudaMalloc((void**)&d_C, sizeof(T) * v_m * v_n));

    float exec_time = 99999.0f;

    int algo = start_algo;

    cublasHandle_t cublas_handle_default;
    cublasCreate(&cublas_handle_default);

    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
    for(; algo <= end_algo; algo++)
    {

        status=CUBLAS_STATUS_SUCCESS;
        CudaTimer timer;
        timer.start();
        for(int ite = 0; ite < ites; ++ite)
        {
            status=cublasGemmEx(cublas_handle_default, 
                    transa, transb, 
                    v_m, v_n, v_k, 
                    &alpha, 
                    d_A, a_type, lda, 
                    d_B, b_type, ldb, 
                    &beta, 
                    d_C, c_type, ldc, 
                    compute_type, 
                    static_cast<cublasGemmAlgo_t>(algo));

        }//end for ite
        float t=timer.stop();
        if(status == CUBLAS_STATUS_SUCCESS){
            //std::cout<<" Test Passed : "<<algo<<" with Time:"<< t<<" ms"<<std::endl;
            if(t<exec_time){
                exec_time=t;
                fast_algo=algo;
            }
        }

    }//end for algo

    std::cout<<"Get Best Cublas Function(cublasGemmEx): "<<fast_algo<<" Time: "<<exec_time<<std::endl;
    cublas_algo[index]=fast_algo;
    ck(cudaFree(d_A));
    ck(cudaFree(d_B));
    ck(cudaFree(d_C));

    ck(cublasDestroy(cublas_handle_default));
    return exec_time;
}

template <typename T>
float  XlnetLayer<T>:: profileCublasGemmStride(cublasOperation_t transa, cublasOperation_t transb,
        int v_m, int v_n, int v_k,int lda, int strideA,
        int ldb,int strideB, int ldc,int strideC, int batch,
        int ites, int index){
    T* d_A;
    T* d_B;
    T* d_C;
    ck(cudaMalloc((void**)&d_A, sizeof(T) * v_m * v_k*batch));
    ck(cudaMalloc((void**)&d_B, sizeof(T) * v_k * v_n*batch));
    ck(cudaMalloc((void**)&d_C, sizeof(T) * v_m * v_n*batch));

    float exec_time = 99999.0f;
    int fast_algo = start_algo;
    int algo = start_algo;
    int second_fast_algo=0;
    float second_exec_time = 99999.0f;

    cublasHandle_t cublas_handle_default;
    cublasCreate(&cublas_handle_default);
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;

    for(; algo <= end_algo; algo++)
    {
        status=CUBLAS_STATUS_SUCCESS;
        CudaTimer timer;
        timer.start();
        for(int ite = 0; ite < ites; ++ite)
        {
            status=cublasGemmStridedBatchedEx(cublas_handle_default, 
                    transa, transb, 
                    v_m, v_n, v_k, 
                    &alpha, 
                    d_A, a_type, lda,strideA, 
                    d_B, b_type, ldb,strideB, 
                    &beta, 
                    d_C, c_type, ldc,strideC, 
                    batch,
                    compute_type, 
                    static_cast<cublasGemmAlgo_t>(algo));
        }//end for ite
        float t=timer.stop()/ites;
        if(status == CUBLAS_STATUS_SUCCESS){
            if(t<exec_time){
                exec_time=t;
                fast_algo=algo;
            }
        }
    }//end for algo

    cublas_func[index]=GEMM_STRIDE;
    cublas_algo[index]=fast_algo;

    cublasFunction f;
    if(strideA==0){
        int m=v_m;
        int n=v_n*batch;
        int k=v_k;
        f=GEMM_A_0;
        for(algo=start_algo; algo <= end_algo; algo++)
        {
            status=CUBLAS_STATUS_SUCCESS;
            CudaTimer timer;
            timer.start();
            for(int ite = 0; ite < ites; ++ite)
            {
                status=cublasGemmEx(cublas_handle_default, 
                        transa, transb, 
                        m, n, k, 
                        &alpha, 
                        d_A, a_type, lda, 
                        d_B, b_type, ldb, 
                        &beta, 
                        d_C, c_type, ldc, 
                        compute_type, 
                        static_cast<cublasGemmAlgo_t>(algo));
            }//end for ite
            float t=timer.stop()/ites;
            if(status == CUBLAS_STATUS_SUCCESS){
                if(t<second_exec_time){
                    second_exec_time=t;
                    second_fast_algo=algo;
                }
            }//end if (status == CUBLAS_STATUS_SUCCESS)
        }//end for algo
    }//end strideA

    if(strideB==0){
        int m=v_m;
        int n=v_n;
        int k=v_k;
        f=GEMM_B_0;

        for(algo=start_algo; algo <= end_algo; algo++)
        {
            status=CUBLAS_STATUS_SUCCESS;
            CudaTimer timer;
            timer.start();
            for(int ite = 0; ite < ites; ++ite)
            {
                for(int count=0;count<batch;count++){
                    status=cublasGemmEx(cublas_handle_default, 
                            transa, transb, 
                            m, n, k, 
                            &alpha, 
                            d_A, a_type, lda, 
                            d_B, b_type, ldb, 
                            &beta, 
                            d_C, c_type, ldc, 
                            compute_type, 
                            static_cast<cublasGemmAlgo_t>(algo));
                }
            }//end for ite
            float t=timer.stop()/ites;
            if(status == CUBLAS_STATUS_SUCCESS){
                if(t<second_exec_time){
                    second_exec_time=t;
                    second_fast_algo=algo;
                }
            }
        }//end for algo
    }//end strideB

    //Set the best cublas function
    if(second_exec_time<exec_time){
        std::cout<<"Get Best Cublas Function(cublasGemmEx): "<<second_fast_algo<<" cublasGemmStridedBatchedEx Time: "
            << exec_time<<" cublasGemmEx Time: "<<second_exec_time<<std::endl;
        exec_time=second_exec_time;
        fast_algo=second_fast_algo;
        cublas_func[index]=(int)f;
        cublas_algo[index]=fast_algo;
    }else{
        std::cout<<"Get Best Cublas Function(cublasGemmStridedBatchedEx): "<<second_fast_algo<<" cublasGemmStridedBatchedEx Time: "
            << exec_time<<" cublasGemmEx Time: "<<second_exec_time<<std::endl;
    }

    ck(cudaFree(d_A));
    ck(cudaFree(d_B));
    ck(cudaFree(d_C));

    ck(cublasDestroy(cublas_handle_default));
    return exec_time;
}




template <typename T>
void XlnetLayer<T>:: profileCublasAlgo(){
    //int ites=50;
    int ites=5;

    float cublas_time[10];
    cublas_time[0]=profileCublasGemmStride(
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim, batch_size * seq_len, hidden_dim,
            hidden_dim,hidden_dim*hidden_dim,hidden_dim, 0,
            hidden_dim,buf_size,3, ites,0);
    cublas_time[1]=profileCublasGemmStride(CUBLAS_OP_T, CUBLAS_OP_N,seq_len, seq_len, size_per_head,
            size_per_head, seq_len * size_per_head,size_per_head, seq_len * size_per_head,seq_len, seq_len * seq_len,
            batch_size * head_num, ites,1);
    cublas_time[2]=profileCublasGemmEx(
            CUBLAS_OP_N, CUBLAS_OP_N, 
            hidden_dim, seq_len*2,hidden_dim,
            hidden_dim, hidden_dim, hidden_dim, ites,2);

    cublas_time[3]=profileCublasGemmStride(CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len*2, seq_len, size_per_head,
            size_per_head, seq_len *2* size_per_head,
            size_per_head, seq_len *size_per_head,
            seq_len*2, seq_len * seq_len*2,
            batch_size * head_num, ites, 3);

    cublas_time[4]=profileCublasGemmStride(CUBLAS_OP_T, CUBLAS_OP_N,
            2, seq_len, size_per_head,
            size_per_head, 2* size_per_head,
            size_per_head, seq_len *size_per_head,
            2, seq_len*2,
            batch_size * head_num,ites,4);

    cublas_time[5]=profileCublasGemmStride(
            CUBLAS_OP_T, CUBLAS_OP_N,
            head_num, seq_len, 2,
            2, 2* head_num,
            2, seq_len *2,
            head_num, seq_len*head_num,
            batch_size * seq_len,
            ites,5 );

    cublas_time[6]=profileCublasGemmStride(CUBLAS_OP_N, CUBLAS_OP_N,
            size_per_head, seq_len, seq_len,
            size_per_head, seq_len* size_per_head,
            seq_len, seq_len*seq_len,
            size_per_head, seq_len*size_per_head,
            batch_size * head_num,
            ites, 6);

    cublas_time[7]=profileCublasGemmStride(CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, seq_len,hidden_dim,
            hidden_dim, 0,
            hidden_dim, seq_len*hidden_dim,
            hidden_dim, seq_len*hidden_dim,
            batch_size,ites,7);

    cublas_time[8]=profileCublasGemmStride(CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim_ff, seq_len,hidden_dim,hidden_dim_ff, 0,
            hidden_dim, seq_len*hidden_dim,hidden_dim_ff, seq_len*hidden_dim_ff,
            batch_size,
            ites,8);

    cublas_time[9]=profileCublasGemmStride(
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim, seq_len,hidden_dim_ff,hidden_dim, 0,
            hidden_dim_ff, seq_len*hidden_dim_ff,hidden_dim, seq_len*hidden_dim,
            batch_size,ites,9);
    std::cout<<"Sequnece length: "<<seq_len<<", Batch size: "<<batch_size<<" Selected cuBLAS method id: ";
    for(int i=0;i<10;i++){
        std::cout<<cublas_algo[i]<<",";
    }
    for(int i=0;i<10;i++){
        std::cout<<cublas_func[i]<<",";
    }
    std::cout<<std::endl<<"Running time of each gemm: ";
    for(int i=0;i<10;i++){
        std::cout<<cublas_time[i]<<",";
    }
    std::cout<<std::endl;
}


template <typename T>
void XlnetLayer<T>:: recordCublasGemm(){
    using namespace std;
    ofstream outfile;
    outfile.open(gemm_file.c_str(),ios::app);

    if (outfile.is_open())
    {
        cout<<"Write profile result in file "<<gemm_file<<endl;

        std::ostringstream ss;
        ss<<gpu_id<<" ,"<<seq_len<<" ,"<<batch_size<<" ,";
        for(int i=0;i<10;i++){ ss<<cublas_algo[i]<<" ,"; }
        for(int i=0;i<10;i++){ ss<<cublas_func[i]<<" ,";
        }

        std::string s= ss.str();
        outfile<<s<<endl;
        outfile.close();
    }
    else
    {
        std::cout<< "Can not write profile result to "<<gemm_file<<endl;
    }

}



template <typename T>
void XlnetLayer<T>:: setCublasAlgo(){
    FILE * fd=fopen(gemm_file.c_str(), "r"); 
    int t_seq_len=0;
    int t_batch=0;
    int t_gpu_id=-1;
    int ifFound=0;
    if(fd != NULL)
    {
        while(!feof(fd)){
            int res=fscanf(fd, "%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,%d ,",
                    &t_gpu_id, &t_seq_len, &t_batch, 
                    &cublas_algo[0], &cublas_algo[1], &cublas_algo[2],&cublas_algo[3], &cublas_algo[4], &cublas_algo[5],
                    &cublas_algo[6], &cublas_algo[7], &cublas_algo[8],&cublas_algo[9], 
                    &cublas_func[0], &cublas_func[1], &cublas_func[2],&cublas_func[3], &cublas_func[4], &cublas_func[5],
                    &cublas_func[6], &cublas_func[7], &cublas_func[8],&cublas_func[9]);
            if(t_seq_len==seq_len&&t_batch==batch_size&&gpu_id==t_gpu_id&&res==FULL_GEMM_LENGTH){
                ifFound=1;
                break;
            }
        }
        fclose(fd);
    }else if(fd == NULL && ifFound == 0){
        printf("Can not find the cublas configuration data. Run profiling code to find the best cublas function.\n");
        profileCublasAlgo();
        recordCublasGemm();
    }
}

template <typename T>
void XlnetLayer<T>::copyCublasAlgo(const int* cublas_algo, const int* cublas_func){
    memcpy(this->cublas_algo, cublas_algo, NUM_CUBLAS_FUNC*sizeof(int));
    memcpy(this->cublas_func, cublas_func, NUM_CUBLAS_FUNC*sizeof(int));
}


/********************** Constructor & Deconstructor ************************/
template <typename T>
XlnetLayer<T>::XlnetLayer(int batch_size, int seq_len, 
        int head_num, int size_per_head,int hidden_dim,int hidden_dim_ff,float epsilon,
        cudaStream_t stream, cublasHandle_t cublas_handle,
        std::string gemm_file,std::string dir,int ifCheck):
    batch_size(batch_size),
    seq_len(seq_len),
    head_num(head_num),
    size_per_head(size_per_head),
    stream(stream),
    cublas_handle(cublas_handle),
    epsilon(epsilon),
    hidden_dim(hidden_dim),
    hidden_dim_ff(hidden_dim_ff),
    dir(dir),
    ifCheck(ifCheck),
    gemm_file(gemm_file),
    layer_weight_device(stream,head_num*size_per_head,hidden_dim_ff){
        //set buf_size
        this->buf_size = batch_size * seq_len*head_num * size_per_head;
        this->qk_buf_size = batch_size * seq_len * head_num * seq_len;

        //set cublas Alogrithm
        cudaGetDevice(&gpu_id);
        setCublas();
        setCublasAlgo();

        //set device variable
        allocDeviceMem();
        //Sync
        cudaDeviceSynchronize();
        ck(cudaGetLastError());
    }

template <typename T>
XlnetLayer<T>::XlnetLayer(XlnetLayer<T> const& xlnet_layer):
    batch_size(xlnet_layer.batch_size),
    seq_len(xlnet_layer.seq_len),
    head_num(xlnet_layer.head_num),
    size_per_head(xlnet_layer.size_per_head),
    stream(xlnet_layer.stream),
    cublas_handle( xlnet_layer.cublas_handle),
    epsilon(xlnet_layer.epsilon),
    hidden_dim(xlnet_layer.hidden_dim),
    hidden_dim_ff(xlnet_layer.hidden_dim_ff),
    buf_size(xlnet_layer.buf_size),
    qk_buf_size(xlnet_layer.qk_buf_size),
    dir(xlnet_layer.dir),
    ifCheck(xlnet_layer.ifCheck),
    gemm_file(xlnet_layer.gemm_file),
    layer_weight_device(xlnet_layer.stream,xlnet_layer.hidden_dim,xlnet_layer.hidden_dim_ff){

        //set cublas Alogrithm
        cudaGetDevice(&gpu_id);
        setCublas();
        copyCublasAlgo(xlnet_layer.cublas_algo,xlnet_layer.cublas_func);

        //set device variable
        allocDeviceMem();

    }


template <typename T>
void XlnetLayer<T>::allocDeviceMem() {
    deviceMalloc(&k_head_r, seq_len*2*hidden_dim);
    deviceMalloc(&qkv_buf,buf_size*3);
    query_buf=qkv_buf;
    key_buf=qkv_buf+buf_size;
    value_buf=qkv_buf+2*buf_size;

    deviceMalloc(&q_buf,buf_size);
    deviceMalloc(&k_buf,buf_size);

    deviceMalloc(&qk_buf,qk_buf_size);

    deviceMalloc(&q_buf_bd, buf_size);
    deviceMalloc(&k_buf_bd, batch_size*seq_len*2*hidden_dim);
    deviceMalloc(&qk_buf_bd, batch_size*seq_len*head_num*seq_len*2);
    deviceMalloc(&qk_buf_bd_shift, batch_size*seq_len*head_num*seq_len);

    deviceMalloc(&q_buf_ef, buf_size);
    deviceMalloc(&k_buf_ef, batch_size*2*hidden_dim);

    deviceMalloc(&qk_buf_ef, batch_size*head_num*seq_len*2);
    deviceMalloc(&qk_buf_ef_trans, batch_size*head_num*seq_len*2);
    deviceMalloc(&qk_buf_ef_seg, batch_size*head_num*seq_len*seq_len);
    deviceMalloc(&qk_buf_ef_seg_trans, batch_size*head_num*seq_len*seq_len);

    deviceMalloc(&attn_score, batch_size*head_num*seq_len*seq_len);
    deviceMalloc(&value_buf_trans,buf_size);
    deviceMalloc(&attn_vec, batch_size*head_num*seq_len*size_per_head);
    deviceMalloc(&attn_vec_trans, batch_size*head_num*seq_len*size_per_head);

    deviceMalloc(&attn_out, batch_size*hidden_dim*seq_len);
    deviceMalloc(&attn_layernorm, batch_size*hidden_dim*seq_len);

    deviceMalloc(&output_fc1, batch_size*seq_len*hidden_dim_ff);
    deviceMalloc(&output_fc2, batch_size*seq_len*hidden_dim);
    deviceMalloc(&output_layernorm, batch_size*seq_len*hidden_dim);
} 


template <typename T>
void XlnetLayer<T>::setLayerWeight(LayerWeightHost<T> & layer_weight_host){
    layer_weight_device.copyFromHost(layer_weight_host);
}

template <typename T>
XlnetLayer<T>::~XlnetLayer() {
    //std::cout << "Deconstruct XlnetLayer" <<std::endl;

    deviceFree(k_head_r);
    deviceFree(qkv_buf);

    deviceFree(q_buf);
    deviceFree(k_buf);
    deviceFree(qk_buf);


    deviceFree(q_buf_bd);
    deviceFree(k_buf_bd);
    deviceFree(qk_buf_bd);
    deviceFree(qk_buf_bd_shift);

    deviceFree(q_buf_ef);
    deviceFree(k_buf_ef);


    deviceFree(qk_buf_ef);
    deviceFree(qk_buf_ef_trans);
    deviceFree(qk_buf_ef_seg);
    deviceFree(qk_buf_ef_seg_trans);

    deviceFree(attn_score);
    deviceFree(value_buf_trans);

    deviceFree(attn_vec);
    deviceFree(attn_vec_trans);

    deviceFree(attn_out);
    deviceFree(attn_layernorm);

    deviceFree(output_fc1);
    deviceFree(output_fc2);
    deviceFree(output_layernorm);

}


//The explicit instantiation part
template class XlnetLayer<__half>; 
template class XlnetLayer<float>;

