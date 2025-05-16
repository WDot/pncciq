import torch
import math

def add_awgn(signal, snr_dB,device,generator=torch.default_generator):
    """
    Add AWGN (complex Gaussian noise) to a complex signal so that
    the resulting signal has the specified SNR (in dB).

    Parameters:
    - signal (torch.Tensor): Complex-valued tensor representing the signal.
                             Shape can be (...), but must be complex dtype.
    - snr_dB (float): Desired signal-to-noise ratio in decibels.

    Returns:
    - noisy_signal (torch.Tensor): The input signal plus AWGN, 
      with average SNR ~= snr_dB.
    """
    # 1) Compute signal power (mean of |signal|^2).
    #    For a complex tensor, signal.abs() gives magnitude per element.
    signal_power = signal.abs().pow(2).mean()

    # 2) Convert snr_dB to linear scale.
    snr_linear = 10 ** (snr_dB / 10.0)

    # 3) Desired noise power is signal_power / snr_linear.
    #    For complex AWGN, we typically split noise power equally into real & imag parts.
    noise_power = signal_power / snr_linear

    # 4) Generate complex Gaussian noise with zero mean and variance = noise_power.
    #    Each real/imag part has variance = noise_power/2.
    #    So standard deviation = sqrt(noise_power/2).
    sigma = torch.sqrt(noise_power / 2)
    
    # Create real and imaginary parts of the noise
    noise_real = sigma * torch.randn(signal.real.shape,device=device,generator=generator)
    noise_imag = sigma * torch.randn(signal.imag.shape,device=device,generator=generator)
    noise = noise_real + 1j * noise_imag

    # 5) Add noise to signal
    noisy_signal = signal + noise
    return noisy_signal

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Create a sample complex signal
    torch.manual_seed(0)
    N = 1000
    # Example signal: random QPSK symbols
    bits = torch.randint(0,2,(2*N,))
    symbols = 2*bits[:N] - 1 + 1j*(2*bits[N:] - 1)  # shape (N,)

    # Desired SNR in dB
    snr_dB = 10.0

    # Add noise
    noisy = add_awgn(symbols, snr_dB)

    # Compute resulting SNR
    signal_power = symbols.abs().pow(2).mean()
    noise_power = (noisy - symbols).abs().pow(2).mean()
    resulting_snr = 10 * torch.log10(signal_power / noise_power)
    print(f"Resulting SNR: {resulting_snr:.2f} dB (target was {snr_dB} dB)")
