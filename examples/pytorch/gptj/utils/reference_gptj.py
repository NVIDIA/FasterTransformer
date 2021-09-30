import torch
from transformers import GPTNeoForCausalLM, AutoConfig
from pathlib import Path

# GPT-J 6B config
config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
config.attention_layers = ["global"] * 28
config.attention_types = [["global"], 28]
config.num_layers = 28
config.num_heads = 16
config.hidden_size = 256 * config.num_heads
config.vocab_size = 50400
config.rotary = True
config.rotary_dim = 64
config.jax = True

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping


class Checkpoint(MutableMapping):
    def __init__(self, chkpt_path, device="cpu"):
        self.device = device
        self.chkpt_path = chkpt_path
        self.checkpoint = torch.load(chkpt_path)
    def __len__(self):
        return len(self.checkpoint)
    def __getitem__(self, key):
        return self.checkpoint[key]
    def __setitem__(self, key, value):
        return
    def __delitem__(self, key, value):
        return
    def keys(self):
        return self.checkpoint.keys()
    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))
    def __copy__(self):
        return Checkpoint(self.chkpt_path, device=self.device)
    def copy(self):
        return Checkpoint(self.chkpt_path, device=self.device)

model = GPTNeoForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=Checkpoint("j6b_ckpt.pt")
)

input_ids = torch.as_tensor([
[ 818,   198,   464,  464,  818,   464,  198,  464],
[ 262,   464,   968,  968,  257,   968,  198,  717],
[ 938,   968,  1971, 1971, 1445,  1971,  464,  640],
[3155,  8221, 12056, 3782,  326, 12056, 5398,  314],
[ 286,  2732,   423,  468,  481,     6, 4332, 2497],
[1528,   286,   257, 3199, 1884,  5859,  628,  262],
[  11, 15198,   649,  663,  787, 41683,  628, 3807],
[ 257,   318,  1182, 5079,  340,   423,  198,   11],
]).T.cuda()
o = model(input_ids)

