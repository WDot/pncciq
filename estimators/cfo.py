import torch
import numpy as np
from torch.nn import functional as F
def cfo(x1,x2,freqs,device='cuda',mask=None):
    N = x1.shape[-1]
    ts = torch.reshape(torch.arange(N,device=device),[1,1,-1])
    shifts = x2 * torch.exp(-1j*2*np.pi*freqs*ts)
    
    if mask == None:
        mask = torch.ones(x1.shape,device=device)
    x1 = x1 * mask
    x1 = x1 / torch.linalg.norm(x1,ord=2,dim=-1,keepdim=True)
    shifts = shifts * mask
    shifts = shifts / torch.linalg.norm(shifts,ord=2,dim=-1,keepdim=True)
    correlation = torch.squeeze(torch.permute(torch.matmul(x1,torch.permute(shifts,(0,2,1))),(0,2,1)),-1)
    return torch.abs(correlation)