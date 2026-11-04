import numpy as np

BIT_ALLOCATION_76 = np.array([
    [8, 7, 6, 4, 3, 0, 0, 0],
    [7, 6, 5, 4, 0, 0, 0, 0],
    [6, 5, 4, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

BIT_ALLOCATION_58 = np.array([
    [8, 7, 6, 4, 0, 0, 0, 0],
    [7, 6, 5, 0, 0, 0, 0, 0],
    [6, 5, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

BIT_ALLOCATION_24 = np.array([
    [8, 8, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])


def get_bit_allocation(rate):
    if rate == 76:
        return BIT_ALLOCATION_76
    elif rate == 58:
        return BIT_ALLOCATION_58
    elif rate == 24:
        return BIT_ALLOCATION_24
    else:
        raise ValueError("Unsupported bit rate")


def zigzag_order(block):
    r, c = block.shape
    result = []

    for diagonal in range(r + c - 1):
        for i in range(r):
            j = diagonal - i

            if 0 <= j < c:
                result.append(block[i, j])

    return np.array(result)


def inverse_zigzag_order(flat_array, block_size=8):
    block = np.zeros((block_size, block_size))
    index = 0

    for diagonal in range(2 * block_size - 1):
        for i in range(block_size):
            j = diagonal - i

            if 0 <= j < block_size:
                block[i, j] = flat_array[index]
                index += 1

    return block