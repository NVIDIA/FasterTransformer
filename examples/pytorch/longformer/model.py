from pdb import set_trace
import torch

from transformers.models.longformer.modeling_longformer import LongformerBaseModelOutput


def from_hf_longformer_weight_to_ft(weights_file, layer_num, fp16):
    weights = torch.load(weights_file)
    all_weights = []
    for i in range(0, layer_num):
        # Need to transpose the kernel for torch.nn.Linear
        # q k v kg vg weights and bias should be continous, required by the ft longformer encoder.
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value.weight".format(i)].transpose(0, 1))
        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.key_global.weight".format(i)].transpose(0, 1))
        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.value_global.weight".format(i)].transpose(0, 1))

        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.query_global.weight".format(i)].transpose(0, 1))

        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key_global.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value_global.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query_global.bias".format(i)])

        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.output.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.LayerNorm.weight".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.LayerNorm.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.intermediate.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.intermediate.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.output.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.output.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.output.LayerNorm.weight".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.output.LayerNorm.bias".format(i)])

    for i in range(0, len(all_weights)):
        all_weights[i] = all_weights[i].flatten()

    all_weights = torch.cat(all_weights).type(torch.float16) if fp16 else torch.cat(all_weights)
    return all_weights.contiguous()


class FTLongformerEncoder(torch.nn.Module):
    def __init__(self, weights_file, layer_num, head_num, size_per_head,
                 intermediate_size, local_attn_window_size,
                 max_global_token_num, batch_size, seq_len,
                 attn_scaler, ft_longformer_lib, fp16=False, hf_plugin_mode=False):
        super().__init__()
        self.fp16 = fp16
        assert seq_len % local_attn_window_size == 0 and seq_len / \
            local_attn_window_size >= 2, "seq_len need to be multiple of local_attn_window_size and at least 2 times big."

        self.hf_plugin_mode = hf_plugin_mode
        all_weight = from_hf_longformer_weight_to_ft(weights_file, layer_num, self.fp16)
        self.all_weight = all_weight.cuda()
        torch.classes.load_library(ft_longformer_lib)
        self.ft_encoder = torch.classes.FasterTransformer.LongformerEncoder(layer_num, head_num * size_per_head,
                                                                            head_num, size_per_head,
                                                                            intermediate_size, local_attn_window_size,
                                                                            max_global_token_num, batch_size, seq_len,
                                                                            attn_scaler)

    def set_hf_plugin_mode(self, is_plugin):
        self.hf_plugin_mode = is_plugin

    def forward(self, *args, **kwargs):
        encoder_in = args[0]

        if self.hf_plugin_mode:
            # In this mode, assume that HuggingFace's LongformerModel.encoder has been
            # substituted to this class's instance
            extended_attention_mask = kwargs['attention_mask']
            local_attn_mask = torch.zeros_like(extended_attention_mask)
            local_attn_mask[extended_attention_mask > -10000.] = 1.0
            global_attn_mask = torch.zeros_like(extended_attention_mask)
            global_attn_mask[extended_attention_mask > 0.] = 1.0
            output = self.ft_encoder.forward(encoder_in, local_attn_mask, global_attn_mask, self.all_weight, 0)
            return LongformerBaseModelOutput(
                last_hidden_state=output,
                hidden_states=None,
                attentions=None,
                global_attentions=None,
            )
        else:
            local_attn_mask = args[1]
            global_attn_mask = args[2]
            return self.ft_encoder.forward(encoder_in, local_attn_mask, global_attn_mask, self.all_weight, 0)
