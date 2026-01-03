import numpy as np
import torch
import random
import os

final_lr = -3
final_layer = 10

def SetSeed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)                  
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
def Normalization(data, MIN, MAX, inverse=False):
    if not inverse:
        data_N = (data - MIN) / (MAX - MIN)
        return data_N
    else:
        data_D = data * (MAX - MIN) + MIN
        return data_D