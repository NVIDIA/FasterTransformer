# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes
import sys
sys.path.insert(0, "../../pytorch/vit")
from VisionTransformerINT8WeightLoader import ViTINT8WeightLoader
import numpy as np
import os
import os.path
import tensorrt as trt
import torch

def load_weights(weight_path:str):
    suffix = weight_path.split('.')[-1]
    if suffix != 'pth':
        print("Unsupported weight file: Unrecognized format %s " % suffix)
        exit(-1)
    return torch.load(weight_path, map_location="cpu")

class ViTINT8PluginLoader:
    def __init__(self, plugin_path) -> None:

        handle = ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        if not handle:
            raise RuntimeError("Fail to load plugin library: %s" % plugin_path)

        self.logger_ = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger_, "")
        plg_registry = trt.get_plugin_registry()

        self.plg_creator = plg_registry.get_plugin_creator("CustomVisionTransformerINT8Plugin", "1", "")

    def load_model_config(self, config, args):
        self.patch_size_        = config.patches.size[0]
        self.num_heads_         = config.transformer.num_heads
        self.layer_num_         = config.transformer.num_layers
        self.inter_size_        = config.transformer.mlp_dim
        self.embed_dim_         = config.hidden_size
        self.max_batch_         = args.batch_size
        self.img_size_          = args.img_size
        self.with_class_token_  = (config.classifier == 'token')
        self.seq_len_           = pow(self.img_size_//self.patch_size_, 2) + 1 if self.with_class_token_ else 0
        self.in_chans_          = 3
        self.int8_mode_           = args.int8_mode
        self.serial_name_       = "ViTINT8Engine_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.patch_size_,
                                                                      self.num_heads_ ,
                                                                      self.layer_num_ ,
                                                                      self.inter_size_,
                                                                      self.embed_dim_ ,
                                                                      self.max_batch_ ,
                                                                      self.img_size_  ,
                                                                      self.seq_len_,
                                                                      self.int8_mode_)
        self.value_holder = []


    def build_plugin_field_collection(self, weights):
        field_type = trt.PluginFieldType.FLOAT16 
        arr_type = np.float16

        self.value_holder = [np.array([self.max_batch_ ]).astype(np.int32),
                             np.array([self.img_size_  ]).astype(np.int32),
                             np.array([self.patch_size_]).astype(np.int32),
                             np.array([self.in_chans_  ]).astype(np.int32),
                             np.array([self.embed_dim_ ]).astype(np.int32),
                             np.array([self.num_heads_ ]).astype(np.int32),
                             np.array([self.inter_size_]).astype(np.int32),
                             np.array([self.layer_num_ ]).astype(np.int32),
                             np.array([self.int8_mode_ ]).astype(np.int32),
                             np.array([self.with_class_token_]).astype(np.int32)        
        ]

        max_batch   = trt.PluginField("max_batch",  self.value_holder[0], trt.PluginFieldType.INT32)
        img_size    = trt.PluginField("img_size",   self.value_holder[1], trt.PluginFieldType.INT32)
        patch_size  = trt.PluginField("patch_size", self.value_holder[2], trt.PluginFieldType.INT32)
        in_chans    = trt.PluginField("in_chans",   self.value_holder[3], trt.PluginFieldType.INT32)
        embed_dim   = trt.PluginField("embed_dim",  self.value_holder[4], trt.PluginFieldType.INT32)
        num_heads   = trt.PluginField("num_heads",  self.value_holder[5], trt.PluginFieldType.INT32)
        inter_size  = trt.PluginField("inter_size", self.value_holder[6], trt.PluginFieldType.INT32)
        layer_num   = trt.PluginField("layer_num",  self.value_holder[7], trt.PluginFieldType.INT32)
        int8_mode   = trt.PluginField("int8_mode",  self.value_holder[8], trt.PluginFieldType.INT32)
        with_cls_token = trt.PluginField("with_cls_token", self.value_holder[9], trt.PluginFieldType.INT32)
        
        vit_weights = ViTINT8WeightLoader(self.layer_num_, self.img_size_, self.patch_size_, weights, 
            classifier='token' if self.with_class_token_ else '' )
        vit_weights.to_int8(ths_path='../../../build/lib/libpyt_vit.so')
        vit_weights.to_cuda()
        weights = vit_weights.listed_weight_to_dict()
        
        part_fc = []
        for name in weights.keys():
            if name == 'transformer.embeddings.cls_token' and (not self.with_class_token_):
                continue
            elif name.split('.')[-1] == 'amaxList' or name.split('.')[-1] == 'h_amaxList':
                self.value_holder.append(weights[name].cpu().numpy().astype(np.float32))
                part_fc.append(trt.PluginField(name, self.value_holder[-1], trt.PluginFieldType.FLOAT32))
            else:
                self.value_holder.append(weights[name].cpu().numpy().astype(np.float16))
                part_fc.append(trt.PluginField(name, self.value_holder[-1], trt.PluginFieldType.FLOAT16))

        return trt.PluginFieldCollection([max_batch, img_size, patch_size, in_chans, embed_dim, num_heads, inter_size, layer_num, int8_mode, with_cls_token] + part_fc)


    def build_network(self, weights):
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # weights = load_weights(weights_path)

        with trt.Builder(self.logger_) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
            builder_config.max_workspace_size = 8 << 30
            builder_config.set_flag(trt.BuilderFlag.FP16)
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Create the network
            input_tensor = network.add_input(name="input_img", dtype=trt.float16, shape=(-1, self.in_chans_, self.img_size_, self.img_size_))
   
            # Specify profiles 
            profile = builder.create_optimization_profile()
            min_shape = (1, self.in_chans_, self.img_size_, self.img_size_)
            ##TODO: There is a bug in TRT when opt batch is large
            max_shape = (self.max_batch_, self.in_chans_, self.img_size_, self.img_size_)
            profile.set_shape("input_img", min=min_shape, opt=min_shape, max=max_shape)
            builder_config.add_optimization_profile(profile)

            #import pdb;pdb.set_trace()
            print("Generate plugin field collection...")
            pfc = self.build_plugin_field_collection(weights)


            fn = self.plg_creator.create_plugin("vision_transformer", pfc)
            inputs = [input_tensor]
            vit = network.add_plugin_v2(inputs, fn) 

            output_tensor = vit.get_output(0)
            output_tensor.name = "visiont_transformer_output"

            vit.precision = trt.float16 
            vit.set_output_type(0, trt.float16)
            network.mark_output(output_tensor)

            print("Building TRT engine....")
            engine = builder.build_engine(network, builder_config)
            return engine

    def serialize_engine(self, engine, file_folder='./'):
        if not os.path.isdir(file_folder):
            self.logger_.log(self.logger_.VERBOSE, "%s is not a folder." % file_folder)
            exit(-1)

        file_path =os.path.join(file_folder, self.serial_name_)

        self.logger_.log(self.logger_.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        self.logger_.log(self.logger_.INFO, "Saving Engine to {:}".format(file_path))
        with open(file_path, "wb") as fout:
            fout.write(serialized_engine)
        self.logger_.log(self.logger_.INFO, "Done.")

    def deserialize_engine(self, file_folder='./'):
        if not os.path.isdir(file_folder):
            self.logger_.log(self.logger_.VERBOSE, "%s is not a folder." % file_folder)
            exit(-1)

        file_path =os.path.join(file_folder, self.serial_name_)
        if not os.path.isfile(file_path):
            self.logger_.log(self.logger_.VERBOSE, "%s not exists. " % file_path)
            return None
        
        filename = os.path.basename(file_path)
        info = filename.split('_')
        self.patch_size_ = int(info[1])
        self.num_heads_  = int(info[2])
        self.layer_num_  = int(info[3])
        self.inter_size_ = int(info[4])
        self.embed_dim_  = int(info[5])
        self.max_batch_  = int(info[6])
        self.img_size_   = int(info[7])
        self.seq_len_    = int(info[8])
        self.int8_mode_  = int(info[9])
        self.in_chans_   = 3
        with open(file_path, 'rb') as f: 
            runtime = trt.Runtime(self.logger_)
            return runtime.deserialize_cuda_engine(f.read())


        

