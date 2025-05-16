import torch

def root_raised_cosine_filter(T, beta, span=6):
    """
    Generate a Root-Raised Cosine (RRC) filter impulse response.
    
    Parameters:
    - T: Symbol period in samples (integer). 
         e.g., if your system uses 8 samples per symbol, T=8.
    - beta: Rolloff factor (float, 0 <= beta <= 1).
    - span: Filter span in symbols (integer). 
            The total filter length will be (2 * span * T + 1) taps.
    
    Returns:
    - rrc_filter: 1D PyTorch tensor containing the RRC filter coefficients.
    """
    # Number of taps: for `span` symbols on each side, plus the center tap
    num_taps = 2 * span * T + 1
    
    # Time index (in samples) from -span*T to +span*T
    n = torch.arange(-span * T, span * T + 1, dtype=torch.float32)
    
    # Convert sample index to "symbol-time" units
    # i.e., t = n / T  means t in units of symbol times
    t = n / T
    
    # Allocate filter
    rrc_filter = torch.zeros(num_taps, dtype=torch.float32)
    
    # Define small epsilon to avoid division by zero
    eps = 1e-8
    
    # Precompute pi * t to avoid repeated multiplication
    pi_t = torch.pi * t

    near_zero_mask = (torch.abs(t) < eps).type(torch.float32)
    denominator_near_zero_mask = (torch.abs(1.0 - (4.0 * beta * t)**2) < eps).type(torch.float32)
    
    near_zero_vector = ((1.0 - beta) + 4.0 * beta / torch.pi) * torch.ones(num_taps, dtype=torch.float32)
    denominator_near_zero_vector_numerator = (1.0 + beta) * torch.sin((1.0 + beta) * pi_t) \
                        - (1.0 - beta) * torch.cos((1.0 - beta) * pi_t) * (4.0 * beta / (torch.pi))
    denominator_near_zero_vector_denominator = 8.0 * beta * t
    denominator_near_zero_vector = denominator_near_zero_vector_numerator / (denominator_near_zero_vector_denominator + eps)
    else_vector_numerator_1 = torch.sin((1.0 - beta) * pi_t)
    else_vector_numerator_2 = 4.0 * beta * t * torch.cos((1.0 + beta) * pi_t)
    else_vector_denominator = torch.pi * t * (1.0 - (4.0 * beta * t)**2)
    else_vector = (else_vector_numerator_1 + else_vector_numerator_2)/(else_vector_denominator + eps)

    rrc_filter_vector = near_zero_mask*near_zero_vector + (1-near_zero_mask)*denominator_near_zero_mask*denominator_near_zero_vector + (1-near_zero_mask)*(1 - denominator_near_zero_mask)*else_vector
    energy = torch.sum(rrc_filter_vector**2)
    rrc_filter_vector = rrc_filter_vector / torch.sqrt(energy + eps)
    #print(rrc_filter_vector)
    '''
    for i in range(num_taps):
        # Handle the special case where t = 0
        # Also handle near-zero denominators carefully
        if abs(t[i]) < eps:
            # limit of RRC filter as t -> 0
            rrc_filter[i] = (1.0 - beta) + 4.0 * beta / torch.pi
        # Handle the case where 1 - (4*beta*t)^2 is close to zero
        elif abs(1.0 - (4.0 * beta * t[i])**2) < eps:
            # limit of RRC filter as denominator -> 0
            # This comes from L'Hopital's rule or known RRC expansions
            numerator = (1.0 + beta) * torch.sin((1.0 + beta) * pi_t[i]) \
                        - (1.0 - beta) * torch.cos((1.0 - beta) * pi_t[i]) * (4.0 * beta / (torch.pi))
            denominator = 8.0 * beta * t[i]
            rrc_filter[i] = numerator / denominator
        else:
            # General RRC formula
            numerator_1 = torch.sin((1.0 - beta) * pi_t[i])
            numerator_2 = 4.0 * beta * t[i] * torch.cos((1.0 + beta) * pi_t[i])
            denominator = torch.pi * t[i] * (1.0 - (4.0 * beta * t[i])**2)
            rrc_filter[i] = (numerator_1 + numerator_2) / denominator

    # Normalize filter to have unit energy
    energy = torch.sum(rrc_filter**2)
    rrc_filter = rrc_filter / torch.sqrt(energy + eps)
    '''

    return rrc_filter_vector

# Example usage:
if __name__ == "__main__":
    T = 8        # 8 samples per symbol
    beta = 0.25  # Rolloff factor
    span = 6     # Span in symbols

    rrc_taps = root_raised_cosine_filter(T, beta, span)
    print(f"RRC filter length = {len(rrc_taps)}")
    print("First 10 taps:", rrc_taps[:10])
