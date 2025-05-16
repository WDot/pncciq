import numpy as np

def rician(fd_norm_min,fd_norm_max, n_samples, N=100,generator=np.random.default_rng()):
    """
    Generate a Rayleigh fading channel coefficient time series using a sum-of-sinusoids method.

    Parameters
    ----------
    fd_norm : float
        Normalized maximum Doppler shift (i.e., fd / Fs in cycles per sample).
    Fs : float
        Sampling frequency in Hz.
    v : float
        Velocity in m/s. (Not directly used in this function when fd_norm is provided.)
    n_samples : int
        Number of samples for the simulation (can be longer or shorter than Fs).
    N : int, optional
        Number of sinusoids to sum (default is 100).

    Returns
    -------
    z : ndarray of complex
        Complex channel coefficients representing the Rayleigh fading channel.
    """
    K = generator.uniform(0,10,[1,N])#10**(Kdb/10.0)
    fd_norm = generator.uniform(fd_norm_min,fd_norm_max,[1,N])
    fd_norm_los = generator.uniform(fd_norm_min,fd_norm_max,[1,N])
    # Convert normalized Doppler frequency to radians per sample.
    omega_d = 2 * np.pi * fd_norm

    # Create a discrete time index (n = 0, 1, 2, ..., n_samples-1)
    n = np.expand_dims(np.arange(n_samples),-1)


    # Sum N sinusoids with random phase and random angle of arrival.
    # Random angle of arrival, uniformly distributed between -pi and pi.
    alpha = (generator.random(size=[1,N]) - 0.5) * 2 * np.pi
    # Random initial phase, uniformly distributed between -pi and pi.
    phi = (generator.random(size=[1,N]) - 0.5) * 2 * np.pi
    # Random amplitude scaling factors from a standard normal distribution.
    a = generator.normal(size=[1,N])
    b = generator.normal(size=[1,N])
    # Accumulate contributions for the cosine and sine components.
    x = a * np.cos(omega_d * n * np.cos(alpha) + phi)
    y = b * np.sin(omega_d * n * np.cos(alpha) + phi)

    # Form the complex channel coefficient and normalize by 1/sqrt(N)
    #rayleigh coponent
    z = (1 / np.sqrt(N)) * (x + 1j * y)
    #print(np.mean(np.abs(z)))

    #rician component
    rician = np.exp(1j*2*np.pi*fd_norm_los*n)

    combined = np.sum(z /np.sqrt(K + 1) + np.sqrt(K/(K + 1))*rician,-1)
    return combined

# Example usage:
if __name__ == "__main__":
    # Example parameters:
    #v = 60 * 0.44704       # Convert 60 mph to m/s
    #Fs = 1e5               # Sample rate (Hz)
    n_samples = 150000     # Total number of samples (can be different from Fs)
    
    # Suppose you already computed a Doppler frequency fd (Hz) and now have:
    # For example, let's assume a Doppler shift computed via:
    #   fd = v * center_freq / 3e8,
    # then normalized frequency is:
    #   fd_norm = fd / Fs.
    # For demonstration, let's set a normalized Doppler frequency directly:
    fd_norm_min = -0.005  # normalized Doppler shift in cycles per sample
    fd_norm_max = 0.005
    # Generate the channel coefficients:
    z = rician(fd_norm_min,fd_norm_max, n_samples, N=100)
    
    # (Optional) Inspect the magnitude in dB:
    import matplotlib.pyplot as plt
    z_mag = np.abs(z)
    z_mag_dB = 10 * np.log10(z_mag)
    n = np.arange(n_samples)
    
    plt.plot(n, z_mag_dB)
    plt.plot([0, n_samples], [0, 0], ':r')  # reference line at 0 dB (no fading)
    plt.xlabel('Sample Index')
    plt.ylabel('Magnitude (dB)')
    plt.title('Simulated Rayleigh Fading Channel')
    plt.legend(['Rayleigh Fading', 'No Fading'])
    plt.axis([0, n_samples, -15, 5])
    plt.show()
