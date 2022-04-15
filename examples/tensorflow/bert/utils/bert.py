# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import tensorflow as tf
import numpy as np
import math
import six
import os
from examples.tensorflow.bert.utils.common import create_initializer

ACTIVATION_AMAX_NUM = 72
INT8O_GEMM_NUM = 8
TRT_AMAX_NUM = 3
SCALE_RESERVE_NUM  = 21

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    tf_datatype=tf.float32):
    
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                            seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        use_bias=True,
        bias_initializer=create_initializer(initializer_range, tf_datatype),
        kernel_initializer=create_initializer(initializer_range, tf_datatype))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        if tf.rank(attention_mask) == 3:
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            
        adder = (1.0 - tf.cast(attention_mask, tf_datatype)) * -10000.0

        attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)

    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    context_layer = tf.matmul(attention_probs, value_layer)

    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def tf_bert(input_tensor,
            encoder_args,
            attention_mask=None,
            intermediate_act_fn=gelu,
            initializer_range=0.02):
    '''
    Run the bert transformer layer by TensorFlow.
    
    Args:
        inputs: A tf.Tensor with shape [batch_size, seq_len, hidden_dimension]. 
                The inputs tensor of encoder. The rank must be 3. 
        encoder_args: The arguments for encoder. The details are in the class 
                      "TransformerArgument" of common.py
        attention_mask: A tf.Tensor. The attention mask for self attention.
        intermediate_act_fn: A callable function.  
                             The activation function in the FFN. It is gelu in BERT. 
        initializer_range: A float value.     
                           The range of initializer for all weights.
        
    Outputs:
        outputs: A tf.Tensor with shape [batch_size, seq_len, hidden_dimension].
                 The results of encoder.
    '''
    
    if encoder_args.hidden_dim % encoder_args.head_num != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (encoder_args.hidden_dim, encoder_args.head_num))

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    prev_output = reshape_to_matrix(input_tensor)

    for layer_idx in range(encoder_args.num_layer):
        with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
            layer_input = prev_output
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=encoder_args.head_num,
                        size_per_head=encoder_args.size_per_head,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length,
                        tf_datatype=encoder_args.dtype)
                    attention_output = attention_head

                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        encoder_args.hidden_dim,
                        use_bias=True,
                        bias_initializer=create_initializer(
                            initializer_range, encoder_args.dtype),
                        kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))
                    attention_output = layer_norm(
                        attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    encoder_args.inter_size,
                    activation=intermediate_act_fn,
                    use_bias=True,
                    bias_initializer=create_initializer(
                        initializer_range, encoder_args.dtype),
                    kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    encoder_args.hidden_dim,
                    use_bias=True,
                    bias_initializer=create_initializer(
                        initializer_range, encoder_args.dtype),
                    kernel_initializer=create_initializer(initializer_range, encoder_args.dtype))
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output

            # amaxList for int8 quantization
            if encoder_args.int8_mode != 0:
                amaxList = tf.get_variable(name="amaxList", shape=[ACTIVATION_AMAX_NUM + 9*encoder_args.hidden_dim + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM], dtype=tf.float32)

    prev_output = tf.reshape(prev_output, shape=tf.shape(input_tensor))
    return prev_output

def build_sequence_mask(sequence_length,
                        num_heads=None,
                        maximum_length=None,
                        dtype=tf.float32):
    """Builds the dot product mask.
    Args:
        sequence_length: The sequence length.
        num_heads: The number of heads.
        maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
        dtype: The type of the mask tensor.
    Returns:
        A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
        ``[batch_size, 1, max_length, max_length]``.
    """
    mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype) # [batch_size, maximum_length]
    mask = tf.reshape(mask, [-1, 1, 1, maximum_length])
    m_2 = tf.transpose(mask, [0, 1, 3, 2])
    mask = mask * m_2

    return mask

def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def ft_bert(inputs,
            encoder_args,
            encoder_vars_dict,
            sequence_length):
    '''
    Run the bert transformer layer by FasterTransformer.
    Args:
        inputs: A tf.Tensor with shape [batch_size, seq_len, hidden_dimension].
                The inputs tensor of encoder. The rank must be 3.
        encoder_args: The arguments for encoder. The details are in the class "TransformerArgument" of common.py
        attention_mask: A tf.Tensor. The attention mask for self attention.
        encoder_vars_dict: A dict of tf.Tensor or numpy array.
                            The variables for encoder. They can be either some tensor or some numpy array.
                            The key is the name of the tensor, like 'layer_0/attention/self/query/kernel:0'.
                            Teh value is the corresponding tensor or numpy array
        sequence_length: A tf.Tensor or numpy array with shape [batch_size].
                        The sequence length of the sentences
    Outputs:
        outputs: A tensor with shape [batch_size, seq_len, hidden_dimension].
                The results of encoder.
    '''
    transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_bert.so'))
    if encoder_args.int8_mode == 0:
        outputs = transformer_op_module.bert(
            inputs,
            inputs,
            sequence_length,
            [encoder_vars_dict['layer_%d/attention/self/query/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/query/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/key/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/key/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/value/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/value/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/LayerNorm/beta:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/LayerNorm/gamma:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/intermediate/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/intermediate/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/LayerNorm/beta:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/LayerNorm/gamma:0' % id] for id in range(encoder_args.num_layer)],
            head_num = encoder_args.head_num, size_per_head = encoder_args.size_per_head,
            inter_size = encoder_args.inter_size,
            num_layer = encoder_args.num_layer, remove_padding=encoder_args.remove_padding,
            q_scaling = 1.0)
    else:
        outputs = transformer_op_module.bert_int8(
            inputs,
            inputs,
            sequence_length,
            [encoder_vars_dict['layer_%d/attention/self/query/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/query/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/key/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/key/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/value/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/self/value/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/LayerNorm/beta:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/attention/output/LayerNorm/gamma:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/intermediate/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/intermediate/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/dense/kernel:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/dense/bias:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/LayerNorm/beta:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/output/LayerNorm/gamma:0' % id] for id in range(encoder_args.num_layer)],
            [encoder_vars_dict['layer_%d/amaxList:0' % id] for id in range(encoder_args.num_layer)],
            head_num = encoder_args.head_num, 
            size_per_head = encoder_args.size_per_head,
            inter_size = encoder_args.inter_size,
            num_layer = encoder_args.num_layer,
            int8_mode = encoder_args.int8_mode,
            remove_padding=encoder_args.remove_padding,
            q_scaling = 1.0)
    return outputs
