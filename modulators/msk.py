import torch
import numpy as np
from scipy.signal import resample_poly
import torch.nn.functional as F
def msk_signal(num_symbols,T, fc, up, down, beta=0.25, span=6, passband=True,generator=torch.default_generator):

    #Note the other args are dummies to match everything else
    """
    Generate a random MSK (Minimum Shift Keying) signal.

    Parameters:
    - T (int): Symbol period in samples.
    - fc (float): Normalized carrier frequency, from 0 to 1.
    - num_symbols (int): Number of symbols (bits) to generate.

    Returns:
    - msk_waveform (torch.Tensor): Complex-valued MSK signal of length num_symbols * T (in samples).
    """
    # Generate a random bit sequence (0 and 1)
    bit_seq = torch.randint(0, 2, (num_symbols,),generator=generator)
    interp_scale = float(up)/float(down)
    # MSK uses continuous-phase FSK with frequency separation of 1/(2*T)
    # Frequencies for bit 0 and bit 1:
    #   f0 = fc - 1/(4*T)
    #   f1 = fc + 1/(4*T)
    # (Some references define it differently, but this is a common parameterization.)
    f0 = fc - 1 / (4.0 * T*interp_scale)
    f1 = fc + 1 / (4.0 * T*interp_scale)

    #t = torch.arange(total_samples, dtype=torch.float32)

    # Phase accumulator to ensure continuous phase
    #phase = torch.zeros(total_samples, dtype=torch.float32)
    impulse_train = bit_seq.repeat_interleave(T)

    #print(impulse_train)
    #ramp = torch.arange(total_samples)#.repeat(num_symbols)
    #print(ramp)

    phase_raw = (2.0*np.pi*(f0*(1-impulse_train) + f1*impulse_train))
    phase_raw = torch.tensor(resample_poly(phase_raw,up,down))
    phase = torch.cumsum(phase_raw,-1)

    msk_waveform = torch.exp(1j * phase)

    return msk_waveform

# Example usage:
if __name__ == "__main__":
    T = 10           # Symbol period in samples
    fc = 0.1         # Normalized carrier frequency
    num_symbols = 50 # Number of symbols

    signal = msk_signal(T, fc, num_symbols)
    print(f"Generated MSK signal with {len(signal)} samples.")

    # Plot the real and imaginary parts
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real.numpy(), label='Real part')
    plt.title('MSK Signal (Real Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal.imag.numpy(), label='Imag part', color='orange')
    plt.title('MSK Signal (Imag Part)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
