import numpy as np


def interleave(bitstream, rows=8):

    bits = np.array([int(bit) for bit in bitstream], dtype=np.uint8)

    N = len(bits)

    if N == 0:
        return "", (N, np.array([], dtype=int))

    cols = int(np.ceil(N / rows))

    # input positions, row-wise
    positions = np.arange(rows * cols)

    # positions corresponding to bits
    valid = positions < N

    # matrix, column-wise
    permutation = positions.reshape(rows, cols).T.flatten()
    permutation = permutation[valid.reshape(rows, cols).T.flatten()]

    interleaved = bits[permutation]

    return "".join(interleaved.astype(str)), (N, permutation)


def deinterleave(bitstream, metadata):

    N, permutation = metadata

    received = np.array([int(bit) for bit in bitstream], dtype=np.uint8)

    if len(received) != N:
        raise ValueError("Received bitstream has incorrect length.")

    original = np.zeros(N, dtype=np.uint8)

    original[permutation] = received

    return "".join(original.astype(str))