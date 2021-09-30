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
import six
import os
from examples.tensorflow.common_utils.position import SinusoidalPositionEncoder

def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

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

def tf_encoder_opennmt(input_tensor,
                        encoder_args,
                        sequence_length,
                        initializer_range=0.02):
    '''
    Run the bert transformer layer by TensorFlow.
    
    Args:
        input_tensor: A tf.Tensor with shape [batch_size, seq_len, hidden_dimension]. 
                       The inputs tensor of encoder. The rank must be 3. 
        encoder_args: The arguments for encoder. The details are in the class 
                      "TransformerArgument" of common.py
        sequence_length: A tf.Tensor with shape [batch_size], with tf.int type.
                         The sequence length of each sentence in input_tensor.
        initializer_range: A float value.     
                           The range of initializer for all weights.
        
    Outputs:
        output: A tf.Tensor with shape [batch_size, max(sequence_length), hidden_dimension].
                The results of encoder.
    '''
    
    data_type = encoder_args.dtype
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    
    input_tensor *= encoder_args.hidden_dim**0.5
    position_encoder = SinusoidalPositionEncoder()
    input_tensor = position_encoder(input_tensor, position=tf.range(seq_length))
    
    mask = build_sequence_mask(
        sequence_length,
        encoder_args.head_num,
        maximum_length=tf.shape(input_tensor)[1],
        dtype=data_type)
    
    intermediate_size = encoder_args.hidden_dim * 4
    if encoder_args.hidden_dim % encoder_args.head_num != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (encoder_args.hidden_dim, encoder_args.head_num))

    layer_input = input_tensor
    for layer_idx in range(encoder_args.num_layer):
        with tf.variable_scope("layer_%d" % layer_idx, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("multi_head"):
                normed_input = tf.cast(layer_norm(tf.cast(layer_input, tf.float32)), data_type)
                
                queries, keys, values = tf.split(tf.layers.conv1d(normed_input, encoder_args.hidden_dim * 3, 1), 3, axis=2)
                
                # split head
                queries = tf.reshape(queries, [batch_size, seq_length, encoder_args.head_num, encoder_args.size_per_head])
                queries = tf.transpose(queries, [0, 2, 1, 3])
                
                keys = tf.reshape(keys, [batch_size, seq_length, encoder_args.head_num, encoder_args.size_per_head])
                keys = tf.transpose(keys, [0, 2, 1, 3])
                
                values = tf.reshape(values, [batch_size, seq_length, encoder_args.head_num, encoder_args.size_per_head])
                values = tf.transpose(values, [0, 2, 1, 3])
                
                queries *= (encoder_args.size_per_head)**-0.5

                dot = tf.matmul(queries, keys, transpose_b=True)
                
                if mask is not None:
                    dot = tf.cast(tf.cast(dot, data_type) * mask + ((1.0 - mask) * data_type.min), dot.dtype)

                attn = tf.cast(tf.nn.softmax(tf.cast(dot, data_type)), dot.dtype)

                context_1 = tf.matmul(attn, values)
                context_1 = tf.transpose(context_1, [0, 2, 1, 3])
                context_1 = tf.reshape(context_1, [batch_size, seq_length, encoder_args.hidden_dim])
                attention_output = tf.layers.conv1d(context_1, encoder_args.hidden_dim, 1)
                context_2 = attention_output + layer_input
                
            with tf.variable_scope("ffn"):
                normed_context_2 = tf.cast(layer_norm(tf.cast(context_2, tf.float32)), data_type)
                intermediate_output = tf.layers.conv1d(normed_context_2, intermediate_size, 1, activation=tf.nn.relu)
                layer_output_1 = tf.layers.conv1d(intermediate_output, encoder_args.hidden_dim, 1)
                layer_output_2 = layer_output_1 + context_2
                layer_input = layer_output_2
                
    layer_input = tf.cast(layer_input, tf.float32)
    output = layer_norm(layer_input, name="LayerNorm")
    output = tf.cast(output, encoder_args.dtype)
    return output

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


def ft_encoder_opennmt(inputs,
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
    
    q_w_list = []
    q_b_list = []
    k_w_list = []
    k_b_list = []
    v_w_list = []
    v_b_list = []

    for i in range(encoder_args.num_layer):
        q_w, k_w, v_w = tf.split(encoder_vars_dict['transformer/encoder/layer_%d/multi_head/conv1d/kernel:0' % i], 3, axis=-1)
        q_w_list.append(q_w)
        k_w_list.append(k_w)
        v_w_list.append(v_w)
        
        q_b, k_b, v_b = tf.split(encoder_vars_dict['transformer/encoder/layer_%d/multi_head/conv1d/bias:0' % i], 3, axis=-1)
        q_b_list.append(q_b)
        k_b_list.append(k_b)
        v_b_list.append(v_b)
    
    input_shape = get_shape_list(inputs, expected_rank=3)
    seq_length = input_shape[1]    
    inputs *= encoder_args.hidden_dim**0.5
    position_encoder = SinusoidalPositionEncoder()
    inputs = position_encoder(inputs, position=tf.range(seq_length))

    transformer_op_module = tf.load_op_library(os.path.join('./lib/libtf_encoder.so'))
    tf_datatype = inputs.dtype
    outputs = transformer_op_module.encoder(
        inputs,
        inputs,
        sequence_length,
        [tf.cast(encoder_vars_dict['transformer/encoder/layer_%d/multi_head/LayerNorm/beta:0' % id], tf_datatype) for id in range(encoder_args.num_layer)],
        [tf.cast(encoder_vars_dict['transformer/encoder/layer_%d/multi_head/LayerNorm/gamma:0' % id], tf_datatype) for id in range(encoder_args.num_layer)],
        q_w_list, q_b_list,
        k_w_list, k_b_list,
        v_w_list, v_b_list,
        [encoder_vars_dict['transformer/encoder/layer_%d/multi_head/conv1d_1/kernel:0' % id] for id in range(encoder_args.num_layer)],
        [encoder_vars_dict['transformer/encoder/layer_%d/multi_head/conv1d_1/bias:0' % id] for id in range(encoder_args.num_layer)],
        [tf.cast(encoder_vars_dict['transformer/encoder/layer_%d/ffn/LayerNorm/beta:0' % id], tf_datatype) for id in range(encoder_args.num_layer)],
        [tf.cast(encoder_vars_dict['transformer/encoder/layer_%d/ffn/LayerNorm/gamma:0' % id], tf_datatype) for id in range(encoder_args.num_layer)],
        [encoder_vars_dict['transformer/encoder/layer_%d/ffn/conv1d/kernel:0' % id] for id in range(encoder_args.num_layer)],
        [encoder_vars_dict['transformer/encoder/layer_%d/ffn/conv1d/bias:0' % id] for id in range(encoder_args.num_layer)],
        [encoder_vars_dict['transformer/encoder/layer_%d/ffn/conv1d_1/kernel:0' % id] for id in range(encoder_args.num_layer)],
        [encoder_vars_dict['transformer/encoder/layer_%d/ffn/conv1d_1/bias:0' % id] for id in range(encoder_args.num_layer)],
        tf.cast(encoder_vars_dict['transformer/encoder/LayerNorm/beta:0'], tf_datatype),
        tf.cast(encoder_vars_dict['transformer/encoder/LayerNorm/gamma:0'], tf_datatype),
        head_num = encoder_args.head_num, size_per_head = encoder_args.size_per_head,
        inter_size = encoder_args.inter_size,
        num_layer = encoder_args.num_layer, remove_padding=encoder_args.remove_padding,
        q_scaling = 1.0)

    return outputs