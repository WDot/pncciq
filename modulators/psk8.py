import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample_poly
from modulators.rrcf import root_raised_cosine_filter

def psk8_signal(num_symbols,T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate an 8-PSK signal using an impulse train + root-raised-cosine (RRC) pulse shaping.
    
    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 <= fc <= 1).
    - num_symbols (int): Number of 8-PSK symbols to generate.
    - beta (float): RRC filter roll-off factor (0 <= beta <= 1).
    - span (int): RRC filter span in symbols.
    - passband (bool): If True, multiply the baseband signal by a carrier at fc.

    Returns:
    - waveform (torch.Tensor): Complex-valued 8-PSK waveform, length ~ (T*num_symbols + 2*span*T - 1).
    """
    # 1) Generate random bit sequence (3 bits per symbol => 8 possible symbols).
    total_bits = 3 * num_symbols
    bits = torch.randint(0, 2, (total_bits,),generator=generator)  # shape: (3*num_symbols,)

    # 2) Group bits into integers [0..7].
    bit_groups = bits.view(num_symbols, 3)
    symbol_ints = bit_groups[:, 0] * 4 + bit_groups[:, 1] * 2 + bit_groups[:, 2]  # range [0..7]

    # 3) Map each integer to an 8-PSK constellation point.
    #    We define equally spaced phases around the unit circle:
    #    0 -> 0, 1 -> pi/4, 2 -> pi/2, 3 -> 3pi/4, 4 -> pi, 5 -> 5pi/4, 6 -> 3pi/2, 7 -> 7pi/4
    #    phases[k] = 2*pi * (k/8)
    phases = 2.0 * np.pi * torch.arange(8, dtype=torch.float32) / 8.0
    constellation = torch.exp(1j * phases)  # shape (8,) in the complex plane

    # 4) Create the 8-PSK symbols from the mapping.
    psk8_symbols = constellation[symbol_ints]  # shape (num_symbols,)

    # 5) Create an impulse train of length T*num_symbols (complex).
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    #for i in range(num_symbols):
    impulse_train[::T] = psk8_symbols

    # 6) Create RRC filter.
    rrc = root_raised_cosine_filter(T, beta, span=span)

    # 7) Convolve the impulse train with RRC (separately real/imag).
    real_train = impulse_train.real.view(1, 1, -1)  # shape (1,1,L)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)  # shape (1,1,K)

    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')  # full conv
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')
    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]
    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))

    # 8) Optional carrier multiplication
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
    T = 8             # Samples per symbol
    fc = 0.1          # Normalized carrier frequency
    num_symbols = 50  # Number of 8-PSK symbols
    beta = 0.25       # RRC roll-off
    span = 6          # Filter span (in symbols)

    # Generate the shaped 8-PSK signal
    signal = psk8_signal(num_symbols, T, fc, beta=beta, span=span, passband=True)
    print(f"Generated RRC-shaped 8-PSK signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped 8-PSK Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped 8-PSK Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
