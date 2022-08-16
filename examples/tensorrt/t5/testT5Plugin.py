#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
#

import configparser
import os
import sys
import ctypes
import argparse
import math
import numpy as np
import tensorrt as trt
import torch
from datetime import datetime

from transformers import PreTrainedTokenizerFast
from transformers import T5Tokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.decoding.utils.recover_bpe import recover_bpe

npToTrt = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}
npToPFT = {np.int8: trt.PluginFieldType.INT8, np.float16: trt.PluginFieldType.FLOAT16, np.int32: trt.PluginFieldType.INT32, np.float32: trt.PluginFieldType.FLOAT32}
npToTorch = {np.dtype('float16'): torch.float16, np.dtype('int32'): torch.int32, np.dtype('float32'): torch.float32}
device = 0

# global variables with default value
globalNMaxBatchSize = 128
globalNMaxSeqLen = 384
globalNBeamSize = 4
globalNUseFP16 = 0

globalNSM = (lambda x: x[0] * 10 + x[1])(torch.cuda.get_device_capability())

globalFBeamDiversity = 0.0
globalFTemperature = 1.0
globalFLenPenalty = 0.0
globalFRepPenalty = 1.0

nMinBatchSize = 1
nOptBatchSize = globalNMaxBatchSize
nMaxBatchSize = globalNMaxBatchSize
nMinSeqLen = 1
nOptSeqLen = globalNMaxSeqLen
nMaxSeqLen = globalNMaxSeqLen

def bleu_score(pred, ref):
    from sacrebleu import corpus_bleu
    bleu = corpus_bleu(pred, [ref], force=True)
    print("       bleu score: {:6.2f}".format(bleu.score))
    print("       bleu counts: {}".format(bleu.counts))
    print("       bleu totals: {}".format(bleu.totals))
    print("       bleu precisions: {}".format(bleu.precisions))
    print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
    return bleu

def getT5EncoderPlugin(arg):
    nBatchSize = arg['batch_size']
    nMaxSeqLen = arg['max_seq_len']
    nBeamSize = arg['beam_width'],
    nSM = globalNSM
    useFP16 = int(arg['data_type'] == 'fp16')
    ckpt_path = arg['ckpt_path'].encode()
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'T5EncoderPlugin':
            pList = [
                trt.PluginField('max_batch_size', np.int32(nBatchSize), npToPFT[np.int32]),
                trt.PluginField('max_seq_len', np.int32(nMaxSeqLen), npToPFT[np.int32]),
                trt.PluginField('beam_width', np.int32(nBeamSize), npToPFT[np.int32]),
                trt.PluginField('sm', np.int32(nSM), npToPFT[np.int32]),
                trt.PluginField('useFP16', np.int32(useFP16), npToPFT[np.int32]),
                trt.PluginField('ckpt_path', ckpt_path, trt.PluginFieldType.CHAR),
            ]
            return c.create_plugin(c.name, trt.PluginFieldCollection(pList))
    return None

def getT5DecodingPlugin(arg):
    nBatchSize = arg['batch_size']
    nMaxSeqLen = arg['max_seq_len']
    nMemMaxSeqLen = arg['max_seq_len']
    nBeamSize = arg['beam_width']
    useFP16 = int(arg['data_type'] == 'fp16')
    ckpt_path = arg['ckpt_path'].encode()
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'T5DecodingPlugin':
            pList = [
                trt.PluginField('max_batch_size', np.int32(nBatchSize), npToPFT[np.int32]),
                trt.PluginField('max_seq_len', np.int32(nMaxSeqLen), npToPFT[np.int32]),
                trt.PluginField('mem_max_seq_len', np.int32(nMaxSeqLen), npToPFT[np.int32]),
                trt.PluginField('beam_width', np.int32(nBeamSize), npToPFT[np.int32]),
                trt.PluginField('useFP16', np.int32(useFP16), npToPFT[np.int32]),
                trt.PluginField('ckpt_path', ckpt_path, trt.PluginFieldType.CHAR),
            ]
            return c.create_plugin(c.name, trt.PluginFieldCollection(pList))
    return None

def buildEngine(logger, arg, trtFileName):
    builder = trt.Builder(logger)
    network = builder.create_network(1)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.flags = int(arg['data_type'] == 'fp16')

    inputT0 = network.add_input('inputId', npToTrt[np.int32], [-1, -1])
    inputT1 = network.add_input('inputSeqLen', npToTrt[np.int32], [-1])
    inputT2 = network.add_input('inputTopK', npToTrt[np.int32], [-1])
    inputT3 = network.add_input('inputTopP', npToTrt[np.int32], [-1])
    inputT4 = network.add_input('inputBeam_search_diversity_rate', npToTrt[np.float32], [-1])
    inputT5 = network.add_input('inputTemperature', npToTrt[np.float32], [-1])
    inputT6 = network.add_input('inputLen_penalty', npToTrt[np.float32], [-1])
    inputT7 = network.add_input('inputRepetition_penalty', npToTrt[np.float32], [-1])

    profile.set_shape(inputT0.name, [nMinBatchSize, nMinSeqLen], [nOptBatchSize, nOptSeqLen], [nMaxBatchSize, nMaxSeqLen])
    profile.set_shape(inputT1.name, [nMinBatchSize], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT2.name, [1], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT3.name, [1], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT4.name, [1], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT5.name, [1], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT6.name, [1], [nOptBatchSize], [nMaxBatchSize])
    profile.set_shape(inputT7.name, [1], [nOptBatchSize], [nMaxBatchSize])
    config.add_optimization_profile(profile)

    model_config = configparser.ConfigParser()
    model_config_path = os.path.join(arg["ckpt_path"], 'config.ini')
    if os.path.isfile(model_config_path):
        model_config.read(model_config_path)

    encoderPlugin = getT5EncoderPlugin(arg)
    decodingPlugin = getT5DecodingPlugin(arg)
    if encoderPlugin == None:
        print("Failed making encoder plugin!")
        return None
    if decodingPlugin == None:
        print("Failed making decoding plugin!")
        return None

    encoderLayer = network.add_plugin_v2([inputT0, inputT1], encoderPlugin)
    decodingLayer = network.add_plugin_v2([encoderLayer.get_output(0), inputT1, inputT2, inputT3, inputT4, inputT5, inputT6, inputT7], decodingPlugin)
    decodingLayer.get_output(0).name = "decodingOutput0"
    decodingLayer.get_output(1).name = "decodingOutput1"
    decodingLayer.get_output(0).dtype = npToTrt[np.int32]
    decodingLayer.get_output(1).dtype = npToTrt[np.int32]

    network.mark_output(decodingLayer.get_output(0))
    network.mark_output(decodingLayer.get_output(1))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed getting serialized engine!")
        return None
    print("Succeeded getting serialized engine!")
    with open(trtFileName, "wb") as f:
        f.write(engineString)
        print("Succeeded saving .plan file!")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    return engine

def testBoth(arg, stream):
    useFP16 = int(arg['data_type'] == 'fp16')
    nBatchSize = arg['batch_size']
    nSeqLen = arg['max_seq_len']
    testCase = "<fp%s,bs=%d,sl=%d>" % (['32', '16'][useFP16], nBatchSize, nSeqLen)
    print("Test both Encoder and Decoding", testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    ctypes.cdll.LoadLibrary(arg['lib_path'])

    trtFile = 'T5Engine-fp' + ['32', '16'][useFP16] + '.plan'
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineString = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        engine = buildEngine(logger, arg, trtFile)

    context = engine.create_execution_context()
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    #for i in range(engine.num_bindings):
    #    print("Bind[%2d]:i[%d]->"%(i,i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->"%(i,i-nInput),
    #            engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i),engine.get_binding_name(i))

    tokenizer = T5Tokenizer.from_pretrained(arg['model'])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(arg['model'])

    with open(arg['source'], 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]

    with open(arg['target'], 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    sys.stdout.flush()

    outputId = []
    outputSeqLen = []
    prev = 0
    needWarmUp = True
    torch.cuda.synchronize()
    start_time = datetime.now()
    while prev < len(src_text):
        input_texts = src_text[prev:prev + nBatchSize]
        prev += nBatchSize

        input_token = tokenizer(input_texts, return_tensors='pt', padding=True)
        inputId = np.ascontiguousarray(input_token['input_ids'].numpy().astype(np.int32))
        inputMask = np.ascontiguousarray(np.sum(input_token['attention_mask'].numpy(), 1).astype(np.int32))
        nRealBatchSize, nRealSeqLen = np.shape(inputId)

        context.set_binding_shape(0, [nRealBatchSize, nRealSeqLen])
        context.set_binding_shape(1, [nRealBatchSize])
        context.set_binding_shape(2, [nRealBatchSize])
        context.set_binding_shape(3, [nRealBatchSize])
        context.set_binding_shape(4, [nRealBatchSize])
        context.set_binding_shape(5, [nRealBatchSize])
        context.set_binding_shape(6, [nRealBatchSize])
        context.set_binding_shape(7, [nRealBatchSize])

        inputTopK = np.full([nRealBatchSize], arg['sampling_topk'], dtype=np.int32)
        inputTopP = np.full([nRealBatchSize], arg['sampling_topp'], dtype=np.float32)

        inputFBeamDiversity = np.full([nRealBatchSize], globalFBeamDiversity, dtype=np.float32)
        inputFTemperature = np.full([nRealBatchSize], globalFTemperature, dtype=np.float32)
        inputFLenPenalty = np.full([nRealBatchSize], globalFLenPenalty, dtype=np.float32)
        inputFRepPenalty = np.full([nRealBatchSize], globalFRepPenalty, dtype=np.float32)

        bufferD = []
        bufferD.append(torch.from_numpy(inputId).to(device))
        bufferD.append(torch.from_numpy(inputMask).to(device))
        bufferD.append(torch.from_numpy(inputTopK).to(device))
        bufferD.append(torch.from_numpy(inputTopP).to(device))
        bufferD.append(torch.from_numpy(inputFBeamDiversity).to(device))
        bufferD.append(torch.from_numpy(inputFTemperature).to(device))
        bufferD.append(torch.from_numpy(inputFLenPenalty).to(device))
        bufferD.append(torch.from_numpy(inputFRepPenalty).to(device))
        bufferD.append(torch.empty(tuple(context.get_binding_shape(8)), dtype=torch.int32, device=device))
        bufferD.append(torch.empty(tuple(context.get_binding_shape(9)), dtype=torch.int32, device=device))
        torch.cuda.synchronize()

        if needWarmUp:
            for i in range(5):
                context.execute_async_v2([b.data_ptr() for b in bufferD], stream)
            prev = 0
            needWarmUp = False
            torch.cuda.synchronize()
            start_time = datetime.now()
            continue

        context.execute_async_v2([b.data_ptr() for b in bufferD], stream)
        torch.cuda.synchronize()

        outputId.append(bufferD[nInput + 0].cpu().numpy())
        outputSeqLen.append(bufferD[nInput + 1].cpu().numpy())

        if len(outputId) >= arg["max_iteration"]:
            break

    stop_time = datetime.now()
    execution_time = (stop_time - start_time).total_seconds()

    outputText = []
    for batch_token, batch_seq_len in zip(outputId, outputSeqLen):
        for j in range(len(batch_token)):
            outputText.append(fast_tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))

    bleuScore = bleu_score(outputText, tgt_text[:len(outputText)])
    with open("output.txt", 'w') as f:
        for line in outputText:
            f.write(line + '\n')
    print("[INFO] FT translates {} batches taking {:.2f} sec to translate {} tokens, BLEU score: {:.2f}, {:.0f} tokens/sec.".format(len(outputText) // nBatchSize, execution_time, bleuScore.sys_len, bleuScore.score, bleuScore.sys_len / execution_time))

    if arg["ft_BLEU_threshold"] != None:
        assert bleuScore.score >= arg["ft_BLEU_threshold"], f"[ERROR] T5Plugin Test FAIL !"
        print(f"[INFO] T5Plugin Test PASS !")
    print(f"[INFO] Test both Encoder and Decoding {testCase} finish!")

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    torch.cuda.set_device(device)
    stream = 0  #torch.cuda.Stream(device).cuda_stream
    #os.system('rm -f ./*.plan ./*.in')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, metavar='NUMBER', default=32, help='batch size (default: 32)')
    parser.add_argument('-beam', '--beam_width', type=int, metavar='NUMBER', default=4, help='beam width (default: 4)')
    parser.add_argument('-s', '--max_seq_len', type=int, metavar='NUMBER', default=128, help='max sequence length (default: 200)')
    parser.add_argument('--source', type=str, metavar='STRING', default="../examples/pytorch/decoding/utils/translation/test.en", help="Path to the source file.")
    parser.add_argument('--target', type=str, metavar='STRING', default="../examples/pytorch/decoding/utils/translation/test.de", help="Path to the target file.")
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, metavar='NUMBER', default=0.0, help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, metavar='NUMBER', default=4, help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, metavar='NUMBER', default=0.0, help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, metavar='STRING', default="fp32", help='data type (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, metavar='STRING', default="lib/libtrt_t5.so", help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-model', '--model', type=str, metavar='STRING', default="t5-small", help='T5 model size.', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument(            '--ckpt_path',      type=str, metavar='STRING', help='path to the checkpoint file.')
    parser.add_argument('-max_ite', '--max_iteration',  type=int,   metavar='NUMBER',   default=100000,     help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    parser.add_argument('--ft_BLEU_threshold', type=float, help='Threshold of FT BLEU score')
    arg = vars(parser.parse_args())
    testBoth(arg, stream)
    print("Test finish!")
