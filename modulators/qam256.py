import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample_poly
from modulators.rrcf import root_raised_cosine_filter


def _map_to_256qam(symbols):
    """
    Map the symbols to the corresponding 256QAM constellation.
    Each symbol is an 8-bit number, so we map it to one of 256 possible points in the complex plane.
    The points are arranged on a square grid.
    """
    # Define the possible values for the I and Q components of 256QAM
    # The values are selected from {-15, -13, ..., 15}
    in_phase = torch.arange(-15, 16, 2)/15.0
    quadrature = torch.arange(-15, 16, 2)/15.0

    # Create the 256QAM constellation by combining in-phase and quadrature components
    I, Q = torch.meshgrid(in_phase, quadrature,indexing='ij')
    constellation = I.flatten() + 1j * Q.flatten()

    return constellation[symbols]


def qam256_signal(num_symbols, T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate a 256-QAM signal using an impulse train + root-raised-cosine (RRC) pulse shaping.

    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 <= fc <= 1).
    - num_symbols (int): Number of 256-QAM symbols to generate.
    - beta (float): RRC filter roll-off factor (0 <= beta <= 1).
    - span (int): RRC filter span in symbols.
    - passband (bool): If True, multiply the baseband signal by a carrier at fc.

    Returns:
    - waveform (torch.Tensor): Complex-valued 256-QAM waveform of length 
      approximately (T * num_symbols + 2 * span * T - 1).
    """
    # 1) Generate random bits (8 bits per symbol).
    total_bits = 8 * num_symbols
    bits = torch.randint(0, 2, (total_bits,),generator=generator)  # shape: (8 * num_symbols,)

    # 2) Group bits into integers [0..255].
    bit_groups = bits.view(num_symbols, 8)
    # symbol_ints in [0..255] using bit shifts
    symbol_ints = (bit_groups[:, 0] << 7) | (bit_groups[:, 1] << 6) | (bit_groups[:, 2] << 5) \
                  | (bit_groups[:, 3] << 4) | (bit_groups[:, 4] << 3) | (bit_groups[:, 5] << 2) \
                  | (bit_groups[:, 6] << 1) |  bit_groups[:, 7]

    
    qam256_symbols = _map_to_256qam(symbol_ints)

    # 4) Create an impulse train of length T*num_symbols (complex).
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    #for i in range(num_symbols):
    impulse_train[::T] = qam256_symbols

    # 5) Create the RRC filter.
    rrc = root_raised_cosine_filter(T, beta, span=span)

    # 6) Convolve (real & imag) separately with RRC filter.
    real_train = impulse_train.real.view(1, 1, -1)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)

    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')  # 'full' convolution
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')
    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]
    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))

    # 7) Optional passband multiplication.
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
    num_symbols = 50  # Number of 256-QAM symbols
    beta = 0.25       # RRC roll-off
    span = 6          # Filter span (in symbols)

    # Generate the shaped 256-QAM signal
    signal = qam256_signal(num_symbols,T, fc,  beta=beta, span=span, passband=True)
    print(f"Generated RRC-shaped 256-QAM signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped 256-QAM Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped 256-QAM Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
