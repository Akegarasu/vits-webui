import torch
from modules import options

cpu = torch.device("cpu")

def get_cuda_device():
    if options.cmd_opts.device_id is not None:
        return f"cuda:{options.cmd_opts.device_id}"

    return "cuda"

def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device(get_cuda_device())
    return cpu

device = cpu if options.cmd_opts.cpu else get_optimal_device()
