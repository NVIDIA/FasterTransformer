
#pragma once
#include "utils.h"


template <typename T>
class XlnetLayer;

template <typename T>
class Xlnet;


/*************************Input********************************/
class InputData{
    protected:
        int batch_size;
        int seq_len;
        int* inp_k;
        float* input_mask;
        int* seg_id;

    public:
        InputData(int batch_size, int seq_len);
        virtual ~InputData()=0;
 };
class InputDataHost;
class InputDataDevice: public InputData{
    private:
        cudaStream_t stream;
    public:
        InputDataDevice(cudaStream_t stream,int batch_size, int seq_len);
        void copyFromHost(InputDataHost& input_data_host);
        ~InputDataDevice();

        template <typename T>
            friend class Xlnet;
};

class InputDataHost: public InputData{
    public:
        InputDataHost(int batch_size, int seq_len);
        void fillInputData(std::string file_name);
        ~InputDataHost();

        friend void InputDataDevice::copyFromHost(InputDataHost& input_data_host);
};

/*************************Pre********************************/
template <typename T>
class PreWeight{
    protected:
        int num_token;
        int hidden_dim;
        T* params_word_emb_k;
    public:
        PreWeight(int hidden_dim, int num_token);
        virtual ~PreWeight()=0;
};

template <typename T>
class PreWeightHost;

template <typename T>
class PreWeightDevice: public PreWeight<T>{
    private:
        cudaStream_t stream;
    public:
        PreWeightDevice<T>(cudaStream_t stream,int hidden_dim, int num_token);
        //PreWeightDevice(PreWeightDevice<T> const& pre_weight_device); 
        void copyFromHost(PreWeightHost<T>& pre_weight_device);
        ~PreWeightDevice<T>();

        friend class Xlnet<T>;
};



template <typename T>
class PreWeightHost: public PreWeight<T>{
    public:
        PreWeightHost<T>(int hidden_dim, int num_token);
        //PreWeightHost(PreWeightHost<T> const& pre_weight_host); 

        void fillPreWeight(std::string file_name);
        ~PreWeightHost<T>();
        friend void PreWeightDevice<T>::copyFromHost(PreWeightHost<T>& pre_weight_host);
};

/*************************Layer********************************/
template <typename T>
class LayerWeight{
    protected:
        int hidden_dim;
        int hidden_dim_ff;

        //Attention Params 
        T* attr_kernel_QKV;
        T* attr_kernel_Q;
        T* attr_kernel_K;
        T* attr_kernel_V;
        T* attr_pos_emb;

        T* attr_bias_Q_w;
        T* attr_bias_Q_r;
        T* attr_bias_Q_s;

        T* attr_seg_embed;

        T* attr_proj_o;

        T* attr_layernorm_gamma;
        T* attr_layernorm_beta;

        T* attr_fc1_bias;
        T* attr_fc1_kernel;

        T* attr_fc2_bias;
        T* attr_fc2_kernel;

        T* attr_ff_gamma;
        T* attr_ff_beta;

    public:
        LayerWeight(int hidden_dim, int hidden_dim_ff);
        virtual ~LayerWeight()=0;

};

template <typename T>
class LayerWeightHost;

template <typename T>
class LayerWeightDevice: public LayerWeight<T>{
    private:
        cudaStream_t stream;
    public:
        LayerWeightDevice(cudaStream_t stream,int hidden_dim, int hidden_dim_ff);
        LayerWeightDevice(LayerWeightDevice<T> const& layer_weight_device); 

        void copyFromHost(LayerWeightHost<T>& layer_weight_host);
        ~LayerWeightDevice();

        friend class XlnetLayer<T>;
};


template <typename T>
class LayerWeightHost: public LayerWeight<T>{
    public:
        LayerWeightHost(int hidden_dim, int hidden_dim_ff);
        void fillLayerWeight(int i_layer,std::string file_name);
        ~LayerWeightHost();
        friend void LayerWeightDevice<T>::copyFromHost(LayerWeightHost<T>& layer_weight_host);
        LayerWeightHost(LayerWeightHost<T> const& layer_weight_host); 
};

