import torch
import numpy as np
from modulators.bpsk import bpsk_signal
from modulators.qpsk import qpsk_signal
from modulators.psk8 import psk8_signal
from modulators.msk import msk_signal
from modulators.qam16 import qam16_signal
from modulators.qam64 import qam64_signal
from modulators.qam256 import qam256_signal
from modulators.pi4_dqpsk import pi4_dqpsk_signal
from modulators.awgn import add_awgn
from modulators.rician import rician
import time
NUM_SAMPLES = int(125e3)
SIGNAL_LENGTH = 16384
SPAN = 6 #for RRCF

label_dict = {'16qam': 0, '64qam': 1, '256qam': 2, 'bpsk': 3, 'qpsk': 4, '8psk': 5, 'dqpsk': 6, 'msk': 7}

modulations = {
            "bpsk": bpsk_signal,
            "qpsk": qpsk_signal,
            "8psk": psk8_signal,
            "16qam": qam16_signal,
            "64qam": qam64_signal,
            "256qam": qam256_signal,
            "dqpsk": pi4_dqpsk_signal,
            "msk": msk_signal
        }

modulation_names = list(label_dict.keys())

def normalize(x):
    xReal = torch.real(x)
    xImag = torch.imag(x)
    
    muReal = torch.mean(xReal)
    muImag = torch.mean(xImag)
    sigmaReal = torch.std(xReal)
    sigmaImag = torch.std(xImag)
    xNormReal = (xReal - muReal)/(sigmaReal + 1e-8)
    xNormImag = (xImag - muImag)/(sigmaImag + 1e-8)
    return xNormReal + 1j*xNormImag
def random_segment(signal):
    max_start_idx = signal.shape[0] - SIGNAL_LENGTH

    start_idx = np.random.randint(0,max_start_idx) if max_start_idx > 0 else 0
    return signal[start_idx:(start_idx + SIGNAL_LENGTH)]

def time_delay(x, delay):
    sample_index = torch.arange(x.shape[-1])
    fftSignal = torch.fft.fftshift(torch.fft.fft(x,dim=-1),dim=-1)
    output = torch.fft.ifft(torch.fft.ifftshift(fftSignal * \
            torch.exp(-1j * 2.0 * np.pi * delay / x.shape[-1] * sample_index),dim=-1),dim=-1) 
    return output

def freq_delay(x, delay):
    sample_index = torch.arange(x.shape[-1])
    delayFunc = torch.exp(1j * 2.0 * np.pi * delay * sample_index)
    output = x * delayFunc
    return output

def forward_delays(rx_signals,time_delays):
        #print(phase_delays.shape)
        return time_delay(rx_signals,time_delays)

def getitem(seed):
    generator = np.random.default_rng(seed)
    torch_generator = torch.Generator().manual_seed(seed)
    mod_choice = np.random.choice(len(modulation_names))
    mod_name = modulation_names[mod_choice]
    mod_label = label_dict[mod_name]
    mod_func = modulations[mod_name]

    # 2) Draw random parameters
    #    - T in [1..20] integer
    T = generator.integers(2, 15 + 1)
    if generator.random() > 0.5:
        up = 1
        down = 1
        interp_scale = 1.0
    else:
        down = generator.integers(1,9 + 1)
        up = down + 1
        interp_scale = float(up) / float(down)

    true_T = float(T*interp_scale)

    num_symbols = int(np.ceil((SIGNAL_LENGTH / true_T))) + 1

    #    - fc in [0,1]
    fc = generator.uniform(-2e-1,2e-1)
    fc2 = generator.uniform(-2e-1,2e-1)
    #    - snr in [0..13] dB
    snr_dB = generator.uniform(-20,20.0)
    #    - beta in [0,1]
    beta = generator.uniform(0.1,1.0)

    signal = mod_func(num_symbols, T, fc, up, down, beta=beta, span=SPAN, passband=True,generator=torch_generator)
    signal = random_segment(signal)
    signal_baseband = freq_delay(signal,-fc)
    signal_passband_shifted = freq_delay(signal,fc2)
    tdelay = generator.uniform(-30.0,30)
    signal_delayed = forward_delays(signal,tdelay)
    rician_min = -0.00005
    rician_max = 0.00005
    #ricianKdb = generator.uniform(0,10)
    rayleigh_num_sinusoids = generator.integers(8,20+1)


    rician_passband = torch.from_numpy(rician(rician_min,rician_max,SIGNAL_LENGTH,rayleigh_num_sinusoids,generator))
    rician_passband_shifted = torch.from_numpy(rician(rician_min,rician_max,SIGNAL_LENGTH,rayleigh_num_sinusoids,generator))
    rician_passband_delayed = torch.from_numpy(rician(rician_min,rician_max,SIGNAL_LENGTH,rayleigh_num_sinusoids,generator))


    # 4) Add noise for the chosen SNR
    noisy_signal_baseband = normalize(add_awgn(signal_baseband,snr_dB,'cpu',torch_generator))
    noisy_signal_baseband = torch.unsqueeze(noisy_signal_baseband,0)
    noisy_signal_passband = normalize(add_awgn(signal*rician_passband, snr_dB,'cpu',torch_generator))
    noisy_signal_passband = torch.unsqueeze(noisy_signal_passband,0)
    noisy_signal_passband_shifted = normalize(add_awgn(signal_passband_shifted*rician_passband_shifted,snr_dB,'cpu',torch_generator))
    noisy_signal_passband_shifted = torch.unsqueeze(noisy_signal_passband_shifted,0)
    noisy_signal_passband_delayed = normalize(add_awgn(signal_delayed*rician_passband_delayed, snr_dB,'cpu',torch_generator))
    noisy_signal_passband_delayed = torch.unsqueeze(noisy_signal_passband_delayed,0)

    return noisy_signal_baseband,noisy_signal_passband,noisy_signal_passband_shifted,noisy_signal_passband_delayed, mod_label,tdelay,snr_dB,fc,fc2,rayleigh_num_sinusoids

#noisy_signals_baseband = np.zeros((NUM_SAMPLES,SIGNAL_LENGTH),dtype=np.complex64)
noisy_signals_passband = np.zeros((NUM_SAMPLES,SIGNAL_LENGTH),dtype=np.complex64)
noisy_signals_passband_shifted = np.zeros((NUM_SAMPLES,SIGNAL_LENGTH),dtype=np.complex64)
noisy_signals_passband_delayed = np.zeros((NUM_SAMPLES,SIGNAL_LENGTH),dtype=np.complex64)
mod_labels = np.zeros(NUM_SAMPLES,dtype=np.int64)
tdelays = np.zeros(NUM_SAMPLES,dtype=np.float32)
snrs = np.zeros(NUM_SAMPLES,dtype=np.float32)
fcs = np.zeros(NUM_SAMPLES,dtype=np.float32)
fc2s = np.zeros(NUM_SAMPLES,dtype=np.float32)
num_sinusoids = np.zeros(NUM_SAMPLES,dtype=np.int64)
ricianKs = np.zeros(NUM_SAMPLES,dtype=np.float32)

print('Initialized Buffers')

times = []
for i in range(NUM_SAMPLES):
    t = time.time()
    noisy_signal_baseband,noisy_signal_passband,noisy_signal_passband_shifted,noisy_signal_passband_delayed, mod_label,tdelay,snr_dB,fc,fc2,rayleigh_num_sinusoids = getitem(i)
    times.append(time.time() - t)
    #noisy_signals_baseband[i,:] = noisy_signal_baseband.detach().cpu().numpy()
    noisy_signals_passband[i,:] = noisy_signal_passband.detach().cpu().numpy()
    noisy_signals_passband_shifted[i,:] = noisy_signal_passband_shifted.detach().cpu().numpy()
    noisy_signals_passband_delayed[i,:] = noisy_signal_passband_delayed.detach().cpu().numpy() 
    mod_labels[i] = mod_label
    tdelays[i] = tdelay
    snrs[i] = snr_dB
    fcs[i] = fc
    fc2s[i] = fc2
    num_sinusoids[i] = rayleigh_num_sinusoids
    #ricianKs[i] = ricianK

    if (i % 10000) == 0:
         print(i)
         print(np.sum(times))
         times = []
print('Done generating, now saving!')
np.savez_compressed('tdelays20250506.npz',
                                          noisy_signals_passband=noisy_signals_passband,\
                                          noisy_signals_passband_shifted=noisy_signals_passband_shifted,\
                                          noisy_signals_passband_delayed=noisy_signals_passband_delayed,\
                                          mod_labels=mod_labels,\
                                          tdelays=tdelays,\
                                          snrs=snrs,\
                                          fcs=fcs,\
                                          fc2s=fc2s,\
                                          num_sinusoids=num_sinusoids)

print('Saved!')

