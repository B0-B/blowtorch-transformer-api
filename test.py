import subprocess as sp
import torch
import os

def get_gpu_memory(id: int=0) -> int:

    '''
    Trys to get the currently allocated memory size by torch.
    https://stackoverflow.com/a/58216793
    '''

    try:
        vram_usage = torch.cuda.memory_allocated(0)
    except:
        vram_usage = 0

    
    return memory_free_values

print(get_gpu_memory())