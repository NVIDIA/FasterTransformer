import os
from typing import Dict
import argparse
import timeit
import logging
import numpy as np
from together_worker.fast_inference import FastInferenceInterface
from together_web3.computer import RequestTypeLanguageModelInference
from together_web3.together import TogetherWeb3, TogetherClientOptions
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from utils.bloom import Bloom
from utils.word_list import to_word_list_format
from transformers import AutoTokenizer, AutoConfig

# logging.setLevel(int(os.environ.get('LOG_LEVEL', logging.DEBUG)))


def get_int(input_: str, default=0) -> int:
    try:
        my_num = int(input_)
        return my_num
    except ValueError:
        print(f'Invalid int {input_} set to default: {default}')
        return default


def get_float(input_: str, default=0.0) -> float:
    try:
        my_num = float(input_)
        return my_num
    except ValueError:
        print(f'Invalid float {input_} set to default: {default}')
        return default


def post_processing_text(output_text, stop_tokens):
    print(f"<post_processing_text> output_text: {output_text}")

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)
            
    print(f"<post_processing_text> stop_tokens: {filtered_stop_tokens}.")

    end_pos = len(output_text)
    print(f"<post_processing_text>1 end_pos: {end_pos}.")
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    print(f"<post_processing_text>2 end_pos: {end_pos}.")
    print(f"<post_processing_text> text: {output_text}, end_pos: {end_pos}")
    post_processed_text = output_text[:end_pos]
    print(f"<post_processing_text> input: {output_text}")
    print(f"<post_processing_text> output: {post_processed_text}")
    return post_processed_text


class FastBloomInference(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        
        args['worker_name'] = 'worker'+str(dist.get_rank())
        args['workers'] = dist.get_world_size()
        args['rank'] = dist.get_rank()
        args['world_size'] = dist.get_world_size()
        
        super().__init__(model_name, args if args is not None else {})
        print("\n=============== Arguments ===============")
        print(args.keys())
        print(args)
        #for key in args.keys():
        #    print("{}: {}".format(arg, getattr(args, arg)))
        print("=========================================\n")
        self.tensor_para_size = args['tensor_para_size']
        self.pipeline_para_size = 1
        self.max_batch_size = args['max_batch_size']
        self.random_seed_tensor = torch.zeros([self.max_batch_size], dtype=torch.int64)
        self.task_info={
            "prompt_seqs": None,
            "output_len":16,
            "beam_width": 1,
            "top_k": 50,
            "top_p": 0,
            "beam_search_diversity_rate": 0,
            "temperature": 0.1,
            "len_penalty": 0,
            "repetition_penalty": 1.0,
            "return_cum_log_probs": 0,
            "return_output_length":0,
        }
        

        hf_config = vars(AutoConfig.from_pretrained(args['hf_model_path']))
        head_num = hf_config['num_attention_heads']
        layer_num = hf_config['n_layer']
        size_per_head = hf_config['n_embed'] // head_num
        self.tokenizer = AutoTokenizer.from_pretrained(args['hf_model_path'], use_fast=False)
        start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id
        vocab_size = hf_config['vocab_size']
        layernorm_eps = 1e-5
        lib_path = args["lib_path"]
        ckpt_path = args['ckpt_path']
        self.tokenizer.pad_token = self.tokenizer.eos_token
        torch.manual_seed(0)
        with torch.no_grad():
            # Prepare model.
            self.bloom_model = Bloom(head_num, size_per_head, 
                        vocab_size, start_id, self.end_id, layer_num,
                        self.tensor_para_size, 
                        self.pipeline_para_size, 
                        lib_path,
                        inference_data_type="fp16",
                        weights_data_type=np.float16,
                        layernorm_eps=layernorm_eps,
                        int8_mode=0)
            if not self.bloom_model.load(ckpt_path=ckpt_path):
                print("[WARNING] Checkpoint file not found. Model loading is skipped.")
               
        print(f"<FastBloomInference.__init__> rank {dist.get_rank()} initialization done")

    def _sync_task_info(self):
        print(f"<FastBloomInference._sync_task_info> enter rank-<{dist.get_rank()}>")
        dist.barrier()
        if dist.get_rank() == 0:
            dist.broadcast_object_list([self.task_info], src=0)
        else:
            info = [None]
            torch.distributed.broadcast_object_list(info, src=0)
            self.task_info = info[0]
        dist.barrier()
        print(f"<FastBloomInference._sync_task_info> leave rank-<{dist.get_rank()}, task_info:{self.task_info}>")
        
    def dispatch_request(self, args, env) -> Dict:
        print(f"Rank {dist.get_rank()} get {args}")
        args = args[0]
        args = {k: v for k, v in args.items() if v is not None}
        # Inputs
        self.task_info["prompt_seqs"] = [args['prompt']]
        self.task_info["output_len"] = get_int(args.get("max_tokens", 16), default=16)
        self.task_info["beam_width"] = get_int(args.get("beam_width", 1), default=1)
        self.task_info["top_k"] = get_int(args.get("top_k", 50), default=50)
        self.task_info["top_p"] = get_float(args.get("top_p", 0.0), default=0.0)
        self.task_info["beam_search_diversity_rate"] = get_float(args.get("beam_search_diversity_rate", 0.0), default=0.0)
        self.task_info["temperature"] = get_float(args.get("temperature", 0.8), default=0.1)
        self.task_info["len_penalty"] = get_float(args.get("len_penalty", 0.0), default=0.0)
        self.task_info["repetition_penalty"] = get_float(args.get("repetition_penalty", 1.0), default=1.0)
        self.task_info["stop"] = args.get("stop", [])
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        self.task_info["return_cum_log_probs"] = args.get("return_cum_log_probs", 0)
        self.task_info["return_output_length"] = args.get("return_output_length", 0)
        self.task_info["stream_tokens"] = args.get("stream_tokens", False)
        
        if len(self.task_info["prompt_seqs"][0]) == 0 or self.task_info["output_len"] == 0:
            inferenece_result = []
            item = {'choices': [], }
            for beam_id in range(self.task_info["beam_width"]):
                choice = {
                    "text": '',
                    "index": beam_id,
                    "finish_reason": "length"
                }
            item['choices'].append(choice)
            inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            result = {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": 0.0
            }
            print(f"<FastBloomInference.dispatch_request> (not FT runs, 0 input or output) return: {result}")
            return result
        else:
            self._sync_task_info()
            result = self._run_inference()
            print(f"<FastBloomInference.dispatch_request> return: {result}")
            return result

    def _run_inference(self):
        print(f"<FastBloomInference._run_inference> enter rank-<{dist.get_rank()}>")
        
        with torch.no_grad():
            contexts = self.task_info["prompt_seqs"]
            start_ids = [torch.IntTensor(self.tokenizer.encode(c)) for c in contexts]
            start_lengths = [len(ids) for ids in start_ids]
            
            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)
            start_lengths = torch.IntTensor(start_lengths)
            print(f"start_ids: length ({start_ids.shape[0]}) ids: {start_ids}")
            
            time = timeit.default_timer()
            max_batch_size = self.max_batch_size
            tokens_batch = self.bloom_model(start_ids,
                                    start_lengths,
                                    self.task_info["output_len"],
                                    self.task_info["beam_width"],
                                    self.task_info["top_k"] * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                    self.task_info["top_p"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["beam_search_diversity_rate"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["temperature"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["len_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    self.task_info["repetition_penalty"] * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                    presence_penalty = None,
                                    min_length = None,
                                    random_seed = self.random_seed_tensor,
                                    bad_words_list = to_word_list_format(np.array([self.task_info["stop"]])) if self.task_info["stop"] else None,
                                    return_output_length = self.task_info["return_output_length"],
                                    return_cum_log_probs = self.task_info["return_cum_log_probs"],
                                    request_id=self.served,
                                    stream_tokens_pipe = self.stream_tokens_pipe_w if self.task_info["stream_tokens"] else -1)
            # only a thread (rank 0) gets the output, while the others are supposed to return None.
            time_elapsed = timeit.default_timer() - time
        print("[INFO] Bloom time costs: {:.2f} ms. <rank-{}>".format(time_elapsed * 1000, dist.get_rank()))
        
        if dist.get_rank() == 0:
            assert tokens_batch is not None
        
            if self.task_info["return_cum_log_probs"] > 0:
                tokens_batch, _, cum_log_probs = tokens_batch
                print('[INFO] Log probs of sentences:', cum_log_probs)

            inferenece_result = []
            tokens_batch = tokens_batch.cpu().numpy()
            
            for i, (context, tokens) in enumerate(zip(self.task_info["prompt_seqs"], tokens_batch)):
                item = {'choices': [], }
                for beam_id in range(self.task_info["beam_width"]):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    output = self.tokenizer.decode(token)
                    print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")
                    choice = {
                        "text": post_processing_text(output, self.task_info["stop"]),
                        "index": beam_id,
                        "finish_reason": "length"
                    }
                item['choices'].append(choice)
                inferenece_result.append(item)
            #  So far coordinator does not support batch. 
            return {
                "result_type": RequestTypeLanguageModelInference,
                "choices": inferenece_result[0]['choices'],
                "raw_compute_time": time_elapsed
            }
        else:
            return None
        
    def worker(self):
        while True:
            self._sync_task_info()
            self._run_inference()
        

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--together_model_name', type=str, default=os.environ.get('SERVICE', 'Together-Bloomchat'),
                        help='worker name for together coordinator.')
    parser.add_argument('--ckpt_path', type=str, default='/workspace/FasterTransformer/build/model/ft-bloom-ock-dolly-oasst1-tp8/8-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--hf_model_path', type=str, default='bigscience/bloom',
                        help='hugging face model name (used to load config).')
    parser.add_argument('--lib_path', type=str, default='/workspace/FasterTransformer/build/lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=8,
                        help='tensor parallel size')
    parser.add_argument('--worker_name', type=str, default=os.environ.get('WORKER','worker1'),
                        help='worker name for together coordinator.')
    parser.add_argument('--group_name', type=str, default=os.environ.get('GROUP', 'group1'),
                        help='group name for together coordinator.')
    
    args = parser.parse_args()
    
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coord_http_port = os.environ.get("COORD_HTTP_PORT", "8092")
    coord_ws_port = os.environ.get("COORD_WS_PORT", "8093")

    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=f"http://{coord_url}:{coord_http_port}",
        websocket_url=f"ws://{coord_url}:{coord_ws_port}/websocket"
    )
    fip = FastBloomInference(model_name=args.together_model_name, args={
        "coordinator": coordinator,
        "hf_model_path": args.hf_model_path,
        "worker_name": args.worker_name,
        "group_name": args.group_name,
        "ckpt_path": args.ckpt_path,
        "lib_path": args.lib_path,
        "tensor_para_size":args.tensor_para_size,
        "stream_tokens_pipe": True,
        "gpu_num": 8,
        "gpu_type": "A100-80G",
        "gpu_mem": 8000000,
        "max_batch_size": 1
    })
    fip.start()
