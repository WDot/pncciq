import torch
import numpy as np
from torch.nn import functional as F
def fcc(x1,x2,taus,device='cuda',mask=None):
    N = x1.shape[-1]
    fft1 = torch.fft.fft(x1,dim=-1)
    conjFft1 = torch.conj(fft1)
    fft2 = torch.fft.fft(x2,dim=-1)
    freqs = torch.reshape(torch.fft.fftfreq(N,1,device=device),[1,1,-1])
    shifts = fft2 * torch.exp(1j*2*np.pi*freqs*taus)
    
    if mask == None:
        mask = torch.ones(x1.shape,device=device)
    conjFft1 = conjFft1 * mask
    conjFft1 = conjFft1 / torch.linalg.norm(conjFft1,ord=2,dim=-1,keepdim=True)
    shifts = shifts * mask
    shifts = shifts / torch.linalg.norm(shifts,ord=2,dim=-1,keepdim=True)
    correlation = torch.squeeze(torch.permute(torch.matmul(conjFft1,torch.permute(shifts,(0,2,1))),(0,2,1)),-1)
    return torch.abs(correlation)