import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample_poly
from modulators.rrcf import root_raised_cosine_filter

def qpsk_signal(num_symbols, T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate a QPSK signal using an impulse train + root-raised-cosine pulse shaping.

    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 to 1).
    - num_symbols (int): Number of QPSK symbols to generate.
    - beta (float): RRC filter roll-off factor (0 <= beta <= 1).
    - span (int): RRC filter span in symbols.
    - passband (bool): If True, multiply the baseband signal by carrier (fc).

    Returns:
    - waveform (torch.Tensor): Complex-valued QPSK waveform of length (T*num_symbols + filter_delay).
                               filter_delay = (2*span*T), if 'full' convolution is kept.
    """
    # Step 1: Generate random QPSK symbols (complex baseband values).
    #    bits -> 2 bits = 1 QPSK symbol
    total_bits = 2 * num_symbols
    bits = torch.randint(0, 2, (total_bits,),generator=generator)
    bit_pairs = bits.view(num_symbols, 2)
    # Gray-coded QPSK mapping
    # 00 -> +1 + j*+1
    # 01 -> +1 + j*-1
    # 11 -> -1 + j*-1
    # 10 -> -1 + j*+1
    symbol_ints = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]  # values in [0..3]
    qpsk_map = (torch.tensor([
        1 + 1j,   # 00
        1 - 1j,   # 01
       -1 - 1j,   # 11
       -1 + 1j    # 10
    ], dtype=torch.complex64))*np.exp(1j*(np.pi/4.0)) #Rotate this
    qpsk_symbols = qpsk_map[symbol_ints]  # shape: (num_symbols,)

    # Step 2: Create an upsampled impulse train of length T * num_symbols.
    #    Place each symbol at multiples of T, zeros elsewhere.
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    #for i in range(num_symbols):
    impulse_train[::T] = qpsk_symbols

    # Step 3: Create RRC filter (real-valued)
    rrc = root_raised_cosine_filter(T, beta, span=span)

    # Step 4: Convolve (complex) impulse train with (real) RRC filter.
    # PyTorch doesn't have native complex conv1d, so we separate real & imag.
    real_train = impulse_train.real.view(1, 1, -1)  # shape (N=1, C=1, L=...)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)  # shape (1,1,K)

    # 'full' convolution: output length = L + K - 1
    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')
    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    # shape: (1,1,L + K -1)

    # Combine real & imaginary back into a single complex waveform
    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]  # shape (L+K-1,)
    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))

    # Step 5: If passband=True, multiply by the carrier e^{j 2π fc n}, else keep baseband
    if passband:
        out_length = shaped_baseband.shape[0]
        n = torch.arange(out_length, dtype=torch.float32)
        carrier_phase = 2.0 * np.pi * fc * n
        carrier = torch.exp(1j * carrier_phase)
        waveform = shaped_baseband * carrier
    else:
        waveform = shaped_baseband

    return waveform

# ------------------- Example usage -------------------
if __name__ == "__main__":
    T = 8            # Samples per symbol
    fc = 0.1         # Normalized carrier frequency
    num_symbols = 50 # Number of QPSK symbols
    beta = 0.25      # RRC roll-off
    span = 6         # RRC span in symbols

    # Generate the shaped QPSK signal
    signal = qpsk_signal(num_symbols,T, fc, beta=beta, span=span, passband=True)
    print(f"Generated shaped QPSK signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped QPSK Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped QPSK Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
