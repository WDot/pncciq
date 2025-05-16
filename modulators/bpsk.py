import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample_poly
from modulators.rrcf import root_raised_cosine_filter

def bpsk_signal(num_symbols,T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate a BPSK signal using an impulse train + root-raised-cosine (RRC) pulse shaping.

    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 <= fc <= 1).
    - num_symbols (int): Number of BPSK symbols to generate.
    - beta (float): RRC filter roll-off factor (0 <= beta <= 1).
    - span (int): RRC filter span in symbol durations.
    - passband (bool): If True, multiply the baseband signal by a carrier at fc.

    Returns:
    - waveform (torch.Tensor): Complex-valued BPSK waveform, length ~ (T*num_symbols + filter_delay).
                               filter_delay = 2*span*T - 1 for 'full' convolution.
    """
    # Step 1: Generate random BPSK symbols.
    #   bit = 0 -> symbol = -1
    #   bit = 1 -> symbol = +1
    bits = torch.randint(0, 2, (num_symbols,),generator=generator)
    bpsk_symbols = 2.0 * bits - 1.0  # maps 0->-1, 1->+1
    bpsk_symbols = bpsk_symbols.to(torch.complex64)

    # Step 2: Create an impulse train (complex) of length T*num_symbols, 
    # placing the BPSK symbols at multiples of T.
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    #for i in range(num_symbols):
    impulse_train[::T] = bpsk_symbols

    # Step 3: Generate the RRC filter.
    rrc = root_raised_cosine_filter(T, beta, span=span)

    # Step 4: Convolve the impulse train with the RRC filter.
    # We'll split real/imag since PyTorch doesn't have complex conv1d.
    real_train = impulse_train.real.view(1, 1, -1)  # shape (N=1, C=1, L=...)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)  # shape (1,1,K)

    # 'full' convolution, output length = L + K - 1
    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')

    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear')
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear')

    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]

    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))
    # Step 5: Optionally up-convert (passband).
    if passband:
        out_length = shaped_baseband.shape[0]
        n = torch.arange(out_length, dtype=torch.float32)
        carrier_phase = 2.0 * np.pi * fc * n
        carrier = torch.exp(1j * carrier_phase)
        waveform = shaped_baseband * carrier
    else:
        waveform = shaped_baseband

    return waveform

# ----------------- Example usage -----------------
if __name__ == "__main__":
    T = 8            # Samples per symbol
    fc = 0.1         # Normalized carrier frequency
    num_symbols = 50 # Number of BPSK symbols
    beta = 0.25      # RRC roll-off factor
    span = 6         # Filter span in symbols

    # Generate the shaped BPSK signal
    signal = bpsk_signal(num_symbols, T, fc, beta=beta, span=span, passband=True)
    print(f"Generated RRC-shaped BPSK signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped BPSK Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped BPSK Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
