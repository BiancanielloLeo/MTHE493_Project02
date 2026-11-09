import json
import numpy as np

from bit_allocation import get_bit_allocation, zigzag_order, inverse_zigzag_order


def load_codebook(path):
    with open(path, "r") as file:
        return json.load(file)


def get_codebooks(codebook, rate, set_index, system):

    if system == "SQ":
        dc_key = f"dc_rate_{rate}"
        ac_key = f"ac_rate_{rate}"

    else:
        dc_key = f"dc_rate_{rate}_set_{set_index}"
        ac_key = f"ac_rate_{rate}_set_{set_index}"

    dc_codebook = np.array(codebook[dc_key], dtype=np.float64)
    ac_codebook = np.array(codebook[ac_key], dtype=np.float64)

    return dc_codebook, ac_codebook


def get_codebook_value(value, codebook):
    index = np.argmin(np.abs(codebook - value))

    return codebook[index], index


def quantize_dct_block(block_values, codebook, set_index, bit_rate, system):
    allocation = get_bit_allocation(bit_rate)
    bit_allocation = zigzag_order(allocation)

    quantized_dct_block = np.zeros(64, dtype=np.float64)
    encoded_bits = []

    for position, bits in enumerate(bit_allocation):
        
        if bits == 0:
            continue

        # get dc codebook for first coefficient
        if position == 0:
            dc_codebook, _ = get_codebooks(codebook, bits, set_index, system)
            selected_codebook = dc_codebook

        # get ac codebook for all other coefficients
        else:
            _, ac_codebook = get_codebooks(codebook, bits, set_index, system)
            selected_codebook = ac_codebook

        # quantize coefficient
        value, index = get_codebook_value(block_values[position], selected_codebook)

        quantized_dct_block[position] = value

        # convert into binary
        binary_index = format(index, f"0{bits}b")
        encoded_bits.extend(int(bit) for bit in binary_index)

    return quantized_dct_block, encoded_bits


def normalize_quantized_blocks(zigzag_blocks):

    normalized_blocks = np.zeros_like(zigzag_blocks, dtype=np.float64)

    # dc coefficients
    dc_coefficients = zigzag_blocks[:,0]

    dc_mean = np.mean(dc_coefficients)
    dc_std = np.std(dc_coefficients)

    normalized_blocks[:, 0] = (dc_coefficients - dc_mean) / dc_std    

    # ac coefficients
    ac_coefficients = zigzag_blocks[:,1:]

    ac_std = np.std(ac_coefficients)

    normalized_blocks[:, 1:] = ac_coefficients / ac_std

    return normalized_blocks, dc_mean, dc_std, ac_std


def denormalize_quantized_blocks(normalized_blocks, dc_mean, dc_std, ac_std):

    denormalized_blocks = np.zeros_like(normalized_blocks, dtype=np.float64)

    # dc coefficients
    denormalized_blocks[:, 0] = (normalized_blocks[:, 0] * dc_std) + dc_mean

    # ac coefficients
    denormalized_blocks[:, 1:] = (normalized_blocks[:, 1:] * ac_std)

    return denormalized_blocks


def compress_image(dct_blocks, codebook, set_index, bit_rate, system):
    # convert into zigzag order
    zigzag_blocks = np.array([zigzag_order(block) for block in dct_blocks])

    # normalize coefficients
    normalized_blocks, dc_mean, dc_std, ac_std = (normalize_quantized_blocks(zigzag_blocks))

    # quantize each normalized block
    bitstream = []

    for block in normalized_blocks:
        _, encoded_bits = quantize_dct_block(block, codebook, set_index, bit_rate, system)
        bitstream.extend(encoded_bits)

    # normalization parameters for decoding
    normalization_params = {"dc_mean": dc_mean, "dc_std": dc_std, "ac_std": ac_std}

    return bitstream, normalization_params, len(dct_blocks)


def decompress_image(bitstream, codebook, set_index, bit_rate, normalization_params, num_blocks, system):

    # get bit allocation
    allocation = get_bit_allocation(bit_rate)
    bit_allocation = zigzag_order(allocation)

    # decode blocks
    normalized_blocks = np.zeros((num_blocks, 64), dtype=np.float64)

    bit_position = 0

    for block_index in range(num_blocks):

        for position, bits in enumerate(bit_allocation):

            if bits == 0:
                continue

            # binary codeword
            codeword = bitstream[bit_position:bit_position + bits]

            bit_position += bits

            # convert binary codeword to index
            index = int("".join(map(str, codeword)), 2)

            # codebook
            if position == 0:
                dc_codebook, _ = get_codebooks(codebook, bits, set_index, system)
                selected_codebook = dc_codebook
            else:
                _, ac_codebook = get_codebooks(codebook, bits, set_index, system)
                selected_codebook = ac_codebook

            # get quantized coefficient
            normalized_blocks[block_index, position] = (selected_codebook[index])

    # denormalize coefficients
    reconstructed_zigzag = denormalize_quantized_blocks(normalized_blocks, normalization_params["dc_mean"], normalization_params["dc_std"], normalization_params["ac_std"])

    # convert into 8x8 blocks
    reconstructed_dct_blocks = np.array([inverse_zigzag_order(block) for block in reconstructed_zigzag])

    return reconstructed_dct_blocks
