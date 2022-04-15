import torch

def print_memory_usage(info=""):
    t = torch.cuda.get_device_properties(0).total_memory / 1024**2
    r = torch.cuda.memory_reserved(0) / 1024**2
    a = torch.cuda.memory_allocated(0) / 1024**2
    f = r-a  # free inside reserved
    print(f"[INFO][{info}] total_memory: {t}, reversed: {r}, allocated: {a}")
