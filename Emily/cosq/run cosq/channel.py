import numpy as np
from functools import lru_cache

lru_cache(maxsize=None)
def channel_with_memory(num_levels, epsilon, delta):

    # number of bits needed to represent each index
    num_levels = int(num_levels)
    n = num_levels.bit_length() - 1

    # transition probabilities of bit-error process
    P_z = np.array([
        [(1 - epsilon + delta) / (1 + delta), epsilon / (1 + delta)],
        [(1 - epsilon) / (1 + delta), (epsilon + delta) / (1 + delta)]
    ])

    # every error-pattern index
    z = np.arange(num_levels)

    # convert error-pattern index into binary, most to least significant
    bit_positions = np.arange(n - 1, -1, -1)
    bits = (z[:, None] >> bit_positions) & 1

    # probability of first bit error state
    p_z = np.where(bits[:, 0] == 1, epsilon, 1 - epsilon).astype(float)

    # probabilities of all remaining bit error states
    for i in range(1, n):
        p_z *= P_z[bits[:, i - 1], bits[:, i]]

    # codeword transition matrix
    P = p_z[z[:, None] ^ z[None, :]]

    # read-only
    P.setflags(write=False)

    return P


def transmit_bits_through_channel(bitstream, epsilon, delta, rng):

    bits = np.fromiter((int(bit) for bit in bitstream), dtype=np.uint8)

    if len(bits) == 0:
        return ""

    P_z = np.array([
        [(1 - epsilon + delta) / (1 + delta), epsilon / (1 + delta)],

        [(1 - epsilon) / (1 + delta), (epsilon + delta) / (1 + delta)]
    ])

    errors = np.zeros(len(bits), dtype=np.uint8)

    # initial stationary probability epsilon
    errors[0] = rng.binomial(1, epsilon)

    # subsequent error bits
    for i in range(1, len(bits)):
        errors[i] = rng.choice([0, 1], p=P_z[errors[i - 1]])

    received = bits ^ errors

    return "".join(received.astype(str))