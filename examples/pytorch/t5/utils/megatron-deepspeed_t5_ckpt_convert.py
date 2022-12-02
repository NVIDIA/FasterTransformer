import configparser
import torch
import argparse
import os
import numpy as np

device = torch.device('cpu')

encoder_config_mapping = {
    "num_attention_heads":"num_heads",
    "kv_channels":"d_kv",
    "hidden_size":"d_model",
    "ffn_hidden_size":"d_ff",
    "num_layers":"num_layers",
    "num_experts":"num_experts",
    "padded_vocab_size":"vocab_size",
    "max_position_embeddings":"relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"
}

decoder_config_mapping = {
    "num_attention_heads":"num_heads",
    "kv_channels":"d_kv",
    "hidden_size":"d_model",
    "ffn_hidden_size":"d_ff",
    "num_layers":"num_layers",
    "num_experts":"num_experts",
    "padded_vocab_size":"vocab_size",
    "max_position_embeddings":"relative_attention_num_buckets_or_max_pos_seq_len",
    "relative_position_num_buckets":"relative_attention_num_buckets_or_max_pos_seq_len"
}

decoder_new_config = {
    "decoder_start_token_id":250104, ## need to adjust
    "eos_token_id":1 ## need to adjust
}

model_new_config = {"structure":{"t5_with_bias":1, "t5_with_moe":0, "moe_layers_in_encoder":[], "moe_layers_in_decoder":[], "use_gated_activation": 0, "position_embedding_type":0}}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, help="path to the deepspeed-megratron checkpoint")
    parser.add_argument("--output_dir", type=str, help="path to the output directory to store binary files")
    parser.add_argument("--tensor_para_size", type=int, help="tensor parallelism size")
    parser.add_argument("--weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    tensor_para_size = args.tensor_para_size
    print("=========================")
    print(f"checkpoint: '{checkpoint_dir}'")
    print(f"output: '{output_dir}'")
    print(f"tensor_para_size: '{tensor_para_size}'")
    print("=========================")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # for tencent t5 model which is trained under ZeRO2
    files = [f for f in os.listdir(checkpoint_dir) if f.find('model_states') != -1]

    model_states_file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    #model_states_file = os.path.join(checkpoint_dir, "model_optim_rng.pt")
    model_states = torch.load(model_states_file, map_location=device)

    model_args = vars(model_states['args'])
    num_layers = model_args['num_layers']

    num_experts = 0
    if 'num_experts' in model_args.keys():
        num_experts = model_args['num_experts'][0]

    # update model structure config
    if not hasattr(model_args, 'position_embedding_type') or model_args.position_embedding_type == "absolute":
        model_new_config["structure"]["position_embedding_type"] = 1

    if num_experts != 0:
        model_new_config["structure"]["t5_with_moe"] = 1

    
    
    model = model_states['module']['language_model']
    embedding = model['embedding']
    encoder = model['encoder']
    decoder = model['decoder']

    word_embeddings = embedding['word_embeddings']
    position_embeddings = embedding['position_embeddings']

    word_embeddings_weight = word_embeddings['weight'].float().detach().numpy()
    file_name = os.path.join(output_dir, "word_embeddings.weight.bin")
    print(f"Saving word_embeddings_weight to '{file_name}'")
    print(f"Shape: '{word_embeddings_weight.shape}'")
    word_embeddings_weight.tofile(file_name)

    position_embeddings_weight = position_embeddings['weight'].float().detach().numpy()
    file_name = os.path.join(output_dir, "position_embeddings.weight.bin")
    print(f"Saving position_embeddings_weight to '{file_name}'")
    print(f"Shape: '{position_embeddings_weight.shape}'")
    position_embeddings_weight.tofile(file_name)

    shared_bias = model_states['module']['lm_head']['bias'].float().detach().numpy()
    file_name = os.path.join(output_dir, "shared.bias.bin")
    print(f"Saving shared.bias to '{file_name}'")
    print(f"Shape: '{shared_bias.shape}'")
    shared_bias.tofile(file_name)

    moe_layers_num = 0
    moe_layers_in_encoder = []
    moe_layers_in_decoder = []

    for k, v in encoder.items():
        val = v.T.float().cpu().numpy().astype(np.float32)
        if k.find("attention.query_key_value.weight") != -1:
            num_splits = 3
            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / num_splits)
            head_num = model_args['num_attention_heads']
            size_per_head = model_args['kv_channels']
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)
            val = val.reshape(hidden_dim, num_splits, local_dim)

            factor = 1
            split_vals = np.split(val, factor, axis=-1)
            query = split_vals[0][:, 0, ...]
            key = split_vals[0][:, 1, ...]
            value = split_vals[0][:, 2, ...]

            prefix = k[:-22] + "query.weight"
            query_tensor_para = np.split(query, tensor_para_size, axis=1)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                query_to_save = query_tensor_para[i]
                print(f"Shape: '{query_to_save.shape}'")
                query_to_save.tofile(file_name)

            prefix = k[:-22] + "key.weight"
            key_tensor_para = np.split(key, tensor_para_size, axis=1)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                key_to_save = key_tensor_para[i]
                print(f"Shape: '{key_to_save.shape}'")
                key_to_save.tofile(file_name)

            prefix = k[:-22] + "value.weight"
            value_tensor_para = np.split(value, tensor_para_size, axis=1)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                value_to_save = value_tensor_para[i]
                print(f"Shape: '{value_to_save.shape}'")
                value_to_save.tofile(file_name)

        elif k.find("attention.query_key_value.bias") != -1:
            num_splits = 3
            local_dim = int(val.shape[-1] / num_splits)
            head_num = model_args['num_attention_heads']
            size_per_head = model_args['kv_channels']
            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.transpose(1, 0, 2)

            val = val.reshape(num_splits, local_dim)
            factor = 1
            split_vals = np.split(val, factor, axis=-1)

            q_bias = split_vals[0][0, ...]
            k_bias = split_vals[0][1, ...]
            v_bias = split_vals[0][2, ...]

            prefix = k[:-20] + "query.bias"
            q_bias_tensor_para = np.split(q_bias, tensor_para_size, axis=0)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                q_bias_to_save = q_bias_tensor_para[i]
                print(f"Shape: '{q_bias_to_save.shape}'")
                q_bias_to_save.tofile(file_name)

            prefix = k[:-20] + "key.bias"
            k_bias_tensor_para = np.split(k_bias, tensor_para_size, axis=0)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                k_bias_to_save = k_bias_tensor_para[i]
                print(f"Shape: '{k_bias_to_save.shape}'")
                k_bias_to_save.tofile(file_name)

            prefix = k[:-20] + "value.bias"
            v_bias_tensor_para = np.split(v_bias, tensor_para_size, axis=0)
            for i in range(tensor_para_size):
                file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                v_bias_to_save = v_bias_tensor_para[i]
                print(f"Shape: '{v_bias_to_save.shape}'")
                v_bias_to_save.tofile(file_name)

        elif k.find('experts') == -1:
            if k.find('deepspeed_moe.gate') != -1:
                layer_id = int(k[7:k.find('.', 7)])
                moe_layers_in_encoder.append(layer_id)

                moe_layers_num += 1
            
            prefix = k
            if k.find('layernorm') != -1 or k.find('gate') != -1 or k.find("attention.dense.bias") != -1 or k.find("dense_4h_to_h.bias") != -1:
                file_name = os.path.join(output_dir, "encoder." + prefix + ".bin")
                print(f"Saving '{prefix}' to '{file_name}'")
                print(f"Shape: '{val.shape}'")
                val.tofile(file_name)
            else:
                val_tensor_para = []
                if k.find("attention.dense.weight") != -1 or k.find("dense_4h_to_h.weight") != -1 or k.find("dense_h_to_4h.bias") != -1:
                    val_tensor_para = np.split(val, tensor_para_size, axis=0)
                elif k.find("dense_h_to_4h.weight") != -1:
                    val_tensor_para = np.split(val, tensor_para_size, axis=1)

                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "encoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    val_to_save = val_tensor_para[i]
                    print(f"Shape: '{val_to_save.shape}'")
                    val_to_save.tofile(file_name)

    print('moe_layers_in_encoder: ', moe_layers_in_encoder)
    model_new_config["structure"]["moe_layers_in_encoder"] = moe_layers_in_encoder
    
    for k, v in decoder.items():
        val = v.T.float().cpu().numpy().astype(np.float32)

        if (k.find("attention.query_key_value.weight") != -1
            or k.find("inter_attention.key_value.weight") != -1
            or k.find("inter_attention.query.weight") != -1): #(d_model * 3, d_model)
            num_splits = 3
            if k.find("inter_attention.key_value.weight") != -1:
                num_splits = 2
            if k.find("inter_attention.query.weight") != -1:
                num_splits = 1

            hidden_dim = val.shape[0]
            local_dim = int(val.shape[-1] / num_splits)
            head_num = model_args['num_attention_heads']
            size_per_head = model_args['kv_channels']
            val = val.reshape(hidden_dim, head_num, num_splits, size_per_head)
            val = val.transpose(0, 2, 1, 3)

            val = val.reshape(hidden_dim, num_splits, local_dim)
            factor = 1
            split_vals = np.split(val, factor, axis=-1)

            if k.find("attention.query_key_value.weight") != -1:
                query_key_value = split_vals[0]
                prefix = k
                query_key_value_tensor_para = np.split(query_key_value, tensor_para_size, axis=2)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    query_key_value_to_save = query_key_value_tensor_para[i]
                    print(f"Shape: '{query_key_value_to_save.shape}'")
                    query_key_value_to_save.tofile(file_name)
            if k.find("inter_attention.key_value.weight") != -1:
                key = split_vals[0][:, 0, ...]
                value = split_vals[0][:, 1, ...]

                prefix = k[:-16] + "key.weight"
                key_tensor_para = np.split(key, tensor_para_size, axis=1)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    key_to_save = key_tensor_para[i]
                    print(f"Shape: '{key_to_save.shape}'")
                    key_to_save.tofile(file_name)

                prefix = k[:-16] + "value.weight"
                value_tensor_para = np.split(value, tensor_para_size, axis=1)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    value_to_save = value_tensor_para[i]
                    print(f"Shape: '{value_to_save.shape}'")
                    value_to_save.tofile(file_name)
            if k.find("inter_attention.query.weight") != -1:
                query = split_vals[0]
                prefix = k
                query_tensor_para = np.split(query, tensor_para_size, axis=2)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    query_to_save = query_tensor_para[i]
                    print(f"Shape: '{query_to_save.shape}'")
                    query_to_save.tofile(file_name)

        elif (k.find("attention.query_key_value.bias") != -1
            or k.find("inter_attention.key_value.bias") != -1
            or k.find("inter_attention.query.bias") != -1):
            num_splits = 3
            if k.find("inter_attention.key_value.bias") != -1:
                num_splits = 2
            if k.find("inter_attention.query.bias") != -1:
                num_splits = 1

            local_dim = int(val.shape[-1] / num_splits)
            head_num = model_args['num_attention_heads']
            size_per_head = model_args['kv_channels']
            val = val.reshape(head_num, num_splits, size_per_head)
            val = val.transpose(1, 0, 2)

            val = val.reshape(num_splits, local_dim)
            factor = 1
            split_vals = np.split(val, factor, axis=-1)

            if k.find("attention.query_key_value.bias") != -1:
                query_key_value_bias = split_vals[0]

                prefix = k
                query_key_value_bias_tensor_para = np.split(query_key_value_bias, tensor_para_size, axis=1)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    query_key_value_bias_to_save = query_key_value_bias_tensor_para[i]
                    print(f"Shape: '{query_key_value_bias_to_save.shape}'")
                    query_key_value_bias_to_save.tofile(file_name)

            if k.find("inter_attention.key_value.bias") != -1:
                key_bias = split_vals[0][0, ...]
                value_bias = split_vals[0][1, ...]

                prefix = k[:-14] + "key.bias"
                key_bias_tensor_para = np.split(key_bias, tensor_para_size, axis=0)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    key_bias_to_save = key_bias_tensor_para[i]
                    print(f"Shape: '{key_bias_to_save.shape}'")
                    key_bias_to_save.tofile(file_name)

                prefix = k[:-14] + "value.bias"
                value_bias_tensor_para = np.split(value_bias, tensor_para_size, axis=0)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    value_bias_to_save = value_bias_tensor_para[i]
                    print(f"Shape: '{value_bias_to_save.shape}'")
                    value_bias_to_save.tofile(file_name)

            if k.find("inter_attention.query.bias") != -1:
                query_bias = split_vals[0]
                prefix = k
                query_bias_tensor_para = np.split(query_bias, tensor_para_size, axis=1)
                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) +  ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    query_bias_to_save = query_bias_tensor_para[i]
                    print(f"Shape: '{query_bias_to_save.shape}'")
                    query_bias_to_save.tofile(file_name)

        elif k.find('experts') == -1:
            if k.find('deepspeed_moe.gate') != -1:
                layer_id = int(k[7:k.find('.', 7)])
                moe_layers_in_decoder.append(layer_id)
                moe_layers_num += 1

            prefix = k
            if k.find('layernorm') != -1 or k.find('gate') != -1 or k.find("attention.dense.bias") != -1 or k.find("dense_4h_to_h.bias") != -1 or \
                k.find("inter_attention.dense.bias") != -1:
                file_name = os.path.join(output_dir, "decoder." + prefix + ".bin")
                print(f"Saving '{prefix}' to '{file_name}'")
                print(f"Shape: '{val.shape}'")
                val.tofile(file_name)
            else:
                val_tensor_para = []
                if k.find("attention.dense.weight") != -1 or k.find("dense_4h_to_h.weight") != -1 or k.find("dense_h_to_4h.bias") != -1:
                    val_tensor_para = np.split(val, tensor_para_size, axis=0)
                elif k.find("dense_h_to_4h.weight") != -1:
                    val_tensor_para = np.split(val, tensor_para_size, axis=1)

                for i in range(tensor_para_size):
                    file_name = os.path.join(output_dir, "decoder." + prefix + "." + str(i) + ".bin")
                    print(f"Saving '{i}' '{prefix}' to '{file_name}'")
                    val_to_save = val_tensor_para[i]
                    print(f"Shape: '{val_to_save.shape}'")
                    val_to_save.tofile(file_name)

    print('moe_layers_in_decoder: ', moe_layers_in_decoder)
    model_new_config["structure"]["moe_layers_in_decoder"] = moe_layers_in_decoder

    # Saving experts weight
    print(f"The number of moe layers is '{moe_layers_num}'")
    for n in range(moe_layers_num):
        experts_in_layer = []
        fc1_weight = []
        fc1_bias = []
        fc2_weight = []
        fc2_bias = []
        prefix = None
        for e in range(num_experts):
            file_name = f"layer_{n}_expert_{e}_mp_rank_00_model_states.pt"
            file_path = os.path.join(checkpoint_dir, file_name)
            expert_dict = torch.load(file_path, map_location=device)
            for k, v in expert_dict.items():
                #val = v.T.float().cpu()#.numpy().astype(np.float32)
                if k.find('dense_h_to_4h.weight') != -1:
                    if prefix is None:
                        prefix = k[15:-22]
                    fc1_weight.append(v)
                if k.find('dense_h_to_4h.bias') != -1:
                    fc1_bias.append(v)
                if k.find('dense_4h_to_h.weight') != -1:
                    fc2_weight.append(v)
                if k.find('dense_4h_to_h.bias') != -1:
                    fc2_bias.append(v)

        stacked_fc1_weight = torch.stack(fc1_weight, 0).transpose(-1, -2).contiguous()
        val = stacked_fc1_weight.float().cpu().numpy()  # (num_experts, d_model, d_ff)
        val_tensor_para = np.split(val, tensor_para_size, axis=2)
        for i in range(tensor_para_size):
            file_name = os.path.join(output_dir, prefix + "dense_h_to_4h.weight." + str(i) + ".bin")
            print(f"Saving '{i}' '{prefix}' to '{file_name}'")
            val_to_save = val_tensor_para[i]
            print(f"Shape: '{val_to_save.shape}'")
            val_to_save.tofile(file_name)

        stacked_fc1_bias = torch.stack(fc1_bias, 0).contiguous()
        val = stacked_fc1_bias.float().cpu().numpy() # (num_experts, d_ff)
        val_tensor_para = np.split(val, tensor_para_size, axis=1)
        for i in range(tensor_para_size):
            file_name = os.path.join(output_dir, prefix + "dense_h_to_4h.bias." + str(i) + ".bin")
            print(f"Saving '{i}' '{prefix}' to '{file_name}'")
            val_to_save = val_tensor_para[i]
            print(f"Shape: '{val_to_save.shape}'")
            val_to_save.tofile(file_name)

        stacked_fc2_weight = torch.stack(fc2_weight, 0).transpose(-1, -2).contiguous()
        val = stacked_fc2_weight.float().cpu().numpy() # (num_experts, d_ff, d_model)
        val_tensor_para = np.split(val, tensor_para_size, axis=1)
        for i in range(tensor_para_size):
            file_name = os.path.join(output_dir, prefix + "dense_4h_to_h.weight." + str(i) + ".bin")
            print(f"Saving '{i}' '{prefix}' to '{file_name}'")
            val_to_save = val_tensor_para[i]
            print(f"Shape: '{val_to_save.shape}'")
            val_to_save.tofile(file_name)

        stacked_fc2_bias = torch.stack(fc2_bias, 0)
        val = stacked_fc2_bias.float().cpu().numpy()
        file_name = os.path.join(output_dir, prefix + "dense_4h_to_h.bias.bin")
        print(f"Saving '{i}' '{prefix}' to '{file_name}'")
        print(f"Shape: '{val_to_save.shape}'")
        val.tofile(file_name)

    config = configparser.ConfigParser()

    config["encoder"] = {}
    config["decoder"] = {}
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"]["weight_data_type"] = args.weight_data_type
    for key, val in model_args.items():
        if key in encoder_config_mapping.keys():
            if key == 'num_experts' and type(val) is list:
                val = val[0]
            config["encoder"][encoder_config_mapping[key]] = f"{val}"
        if key in decoder_config_mapping.keys():
            if key == 'num_experts' and type(val) is list:
                val = val[0]
            config["decoder"][decoder_config_mapping[key]] = f"{val}"

    for key, val in decoder_new_config.items():
        config["decoder"][key] = f"{val}"
    for key, val in model_new_config.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = f"{val_val}"

    config_save_path = os.path.join(output_dir, "config.ini")
    with open(config_save_path, 'w') as configfile:
        config.write(configfile)

