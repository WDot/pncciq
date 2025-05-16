import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample_poly
from modulators.rrcf import root_raised_cosine_filter

def pi4_dqpsk_signal(num_symbols, T, fc, up,down, beta=0.25, span=6, passband=True,generator=torch.default_generator):
    """
    Generate a π/4-DQPSK signal using differential encoding + impulse train + RRC pulse shaping.

    Parameters:
    - T (int): Samples per symbol.
    - fc (float): Normalized carrier frequency (0 <= fc <= 1).
    - num_symbols (int): Number of π/4-DQPSK symbols to generate.
    - beta (float): RRC filter roll-off factor.
    - span (int): RRC filter span in symbols.
    - passband (bool): If True, multiply the baseband signal by a carrier at fc.

    Returns:
    - waveform (torch.Tensor): Complex-valued π/4-DQPSK waveform of length
      approximately (T*num_symbols + 2*span*T - 1).
    """
    # 1) Generate random bits and map to phase increments (2 bits per symbol => 4 increments).
    #    00 -> +π/4
    #    01 -> +3π/4
    #    10 -> -π/4
    #    11 -> -3π/4

    total_bits = 2 * num_symbols
    bits = torch.randint(0, 2, (total_bits,),generator=generator)  # shape: (2*num_symbols,)
    bit_pairs = bits.view(num_symbols, 2)
    # Convert to integer 0..3
    symbol_ints = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]

    # Map each integer to a phase increment
    # increments[i]: in {+π/4, +3π/4, -π/4, -3π/4}
    increment_map = {
        0: np.pi/4,
        1: 3*np.pi/4,
        2: -np.pi/4,
        3: -3*np.pi/4
    }
    increments = torch.tensor([increment_map[int(x)] for x in symbol_ints], dtype=torch.float32)

    # 2) Differentially encode the absolute phase across symbols.
    #    phase[i] = phase[i-1] + increments[i], with phase[0] = increments[0].
    #phase_symbols = torch.zeros(num_symbols, dtype=torch.float32)
    #if num_symbols > 0:
    #    phase_symbols[0] = increments[0]
    #for i in range(1, num_symbols):
    #    phase_symbols[i] = phase_symbols[i-1] + increments[i]
    phase_symbols = torch.cumsum(increments,-1)

    # Optionally wrap phase to [-π, π] (not strictly required, but can help keep angles bounded)
    phase_symbols = torch.fmod(phase_symbols + np.pi, 2*np.pi) - np.pi

    # 3) Convert each symbol's phase to a complex point on the unit circle: e^{j*phase_symbols[i]}.
    #    This is the "symbol-level" representation, ignoring pulse shape so far.
    dqpsk_symbols = torch.exp(1j * phase_symbols)

    # 4) Create an impulse train of length T*num_symbols, placing each symbol at multiples of T.
    total_samples = T * num_symbols
    impulse_train = torch.zeros(total_samples, dtype=torch.complex64)
    impulse_train[::T] = dqpsk_symbols
    #for i in range(num_symbols):
    #    impulse_train[i * T] = dqpsk_symbols[i]

    # 5) Convolve with the RRC filter.
    rrc = root_raised_cosine_filter(T, beta, span=span)

    real_train = impulse_train.real.view(1, 1, -1)
    imag_train = impulse_train.imag.view(1, 1, -1)
    rrc_2d = rrc.view(1, 1, -1)

    real_shaped = F.conv1d(real_train, rrc_2d, padding='same')  # 'full' conv => L_out = L + K - 1
    imag_shaped = F.conv1d(imag_train, rrc_2d, padding='same')

    #real_shaped = F.interpolate(real_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    #imag_shaped = F.interpolate(imag_shaped,scale_factor=interp_scale,mode='linear',recompute_scale_factor=True)
    shaped_baseband = real_shaped[0,0,:] + 1j * imag_shaped[0,0,:]
    shaped_baseband = torch.tensor(resample_poly(shaped_baseband,up,down))
    # 6) Optionally multiply by carrier for passband.
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
    num_symbols = 50  # Number of π/4-DQPSK symbols
    beta = 0.25       # RRC roll-off factor
    span = 6          # RRC filter span (in symbols)

    signal = pi4_dqpsk_signal(num_symbols, T, fc, beta=beta, span=span, passband=True)
    print(f"Generated RRC-shaped π/4-DQPSK signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('RRC-Shaped π/4-DQPSK Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('RRC-Shaped π/4-DQPSK Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
