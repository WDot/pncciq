import torch
import numpy as np
import torch.nn.functional as F
from modulators.rrcf import root_raised_cosine_filter
from scipy.signal import resample_poly

def _map_to_16qam(symbols):
    """
    Map the symbols to the corresponding 16QAM constellation.
    Each symbol is a 4-bit number, so we map it to one of 16 possible points in the complex plane.
    """
    # 16QAM constellation points (arranged in a square grid)
    # In-phase (I) and Quadrature (Q) are in the set {-3, -1, 1, 3}
    # 16QAM uses 4 bits per symbol, creating 16 combinations
    in_phase = torch.tensor([-3, -1, 1, 3])/3.0
    quadrature = torch.tensor([-3, -1, 1, 3])/3.0

    # Mapping each symbol to a 2D point in the complex plane
    I, Q = torch.meshgrid(in_phase, quadrature,indexing='ij')
    constellation = I.flatten() + 1j * Q.flatten()
    
    return constellation[symbols]


def qam16_signal(num_symbols, T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate a 16-QAM signal using an impulse train + root-raised-cosine (RRC) pulse shaping.

    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 <= fc <= 1).
    - num_symbols (int): Number of 16-QAM symbols to generate.
    - beta (float): RRC filter roll-off factor (0 <= beta <= 1).
    - span (int): RRC filter span in symbols.
    - passband (bool): If True, multiply the baseband signal by a carrier at fc.

    Returns:
    - waveform (torch.Tensor): Complex-valued 16-QAM waveform of length ~ (T*num_symbols + (2*span*T - 1)).
    """

    # 1) Generate random bits (4 bits per 16-QAM symbol).
    total_bits = 4 * num_symbols
    bits = torch.randint(0, 2, (total_bits,),generator=generator)  # shape: (4*num_symbols,)

    # 2) Group bits into integers [0..15].
    bit_groups = bits.view(num_symbols, 4)
    symbol_ints = (bit_groups[:, 0] << 3) | (bit_groups[:, 1] << 2) | (bit_groups[:, 2] << 1) | bit_groups[:, 3]
    

    # 4) Use symbol_ints to index into the constellation.
    qam16_symbols = _map_to_16qam(symbol_ints)  # shape (num_symbols,)

    # 5) Create an impulse train (complex) of length T*num_symbols.
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    #for i in range(num_symbols):
    impulse_train[::T] = qam16_symbols
    
    # 6) Create the RRC filter.
    rrc = root_raised_cosine_filter(T, beta, span=span)

    # 7) Convolve (real & imag) separately with RRC filter.
    real_train = impulse_train.real.view(1, 1, -1)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)

    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')  # 'full' conv
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')
    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    
    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]
    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))

    # 8) Optional passband multiplication.
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
    num_symbols = 50  # Number of QAM16 symbols
    beta = 0.25       # RRC roll-off
    span = 6          # Filter span (in symbols)

    # Generate the shaped 16-QAM signal
    signal = qam16_signal(num_symbols,T, fc, beta=beta, span=span, passband=True)
    print(f"Generated RRC-shaped 16-QAM signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped 16-QAM Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped 16-QAM Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
