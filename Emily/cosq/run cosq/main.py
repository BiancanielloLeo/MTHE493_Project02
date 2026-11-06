import numpy as np
import csv
import os

from image_processing import load_image, divide_into_blocks, reconstruct_from_blocks
from dct import apply_dct, apply_idct
from compression import load_codebook, compress_image, decompress_image
from channel import channel_with_memory, transmit_bits_through_channel
from interleaver import interleave, deinterleave
from bit_allocation import get_bit_allocation, zigzag_order

BIT_RATES = [76, 58, 24]
EPSILONS = [0, 0.005, 0.01, 0.05, 0.1]
COSQ_SETS = {0: [0], 0.005: [1, 7], 0.01: [2, 6], 0.05: [3, 5], 0.1: [4]}
DELTAS = [0,5,10]

BLOCK_SIZE = 8
INTERLEAVER_ROWS = 8

IMAGE_PATH = "images/lena_512_gray.jpg"

OUTPUT_FOLDER = "results"


def calculate_psnr(original, reconstructed):

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10(255 ** 2 / mse)


def transmit_cosq(bitstream, bit_allocation, epsilon, delta, rng):
    received = []
    position = 0

    while position < len(bitstream):
        for width in bit_allocation:
            if width == 0:
                continue

            if position >= len(bitstream):
                break

            index_bits = bitstream[position:position + width]
            position += width
            index = int("".join(map(str, index_bits)), 2)

            P = channel_with_memory(2 ** width, epsilon, delta)

            received_index = rng.choice(2 ** width, p=P[index])

            received_bits = format(received_index, f"0{width}b")
            received.extend(int(bit) for bit in received_bits)

    return received


def transmit_sq_il(bitstream, epsilon, delta, rng):

    interleaved, metadata = interleave(bitstream, rows=INTERLEAVER_ROWS)
    received = transmit_bits_through_channel(interleaved, epsilon, delta, rng)

    return deinterleave(received, metadata)


def reconstruct_image(bitstream, codebook, set_index, bit_rate, params, num_blocks, image_shape, system):

    reconstructed_dct = decompress_image(bitstream, codebook, set_index, bit_rate, params, num_blocks, system)
    reconstructed_blocks = apply_idct(reconstructed_dct)
    reconstructed_image = reconstruct_from_blocks(reconstructed_blocks, image_shape, BLOCK_SIZE)

    return reconstructed_image


def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image = load_image(IMAGE_PATH)

    blocks = divide_into_blocks(image, BLOCK_SIZE)

    dct_blocks = apply_dct(blocks)

    sq_codebook = load_codebook("codebooks/SQ_Codebook.json")

    results = []

    rng = np.random.default_rng(None)

    for bit_rate in BIT_RATES:

        print(f"\nBIT RATE: {bit_rate}")

        allocation = get_bit_allocation(bit_rate)
        bit_allocation = zigzag_order(allocation)

        sq_bitstream, sq_params, num_blocks = compress_image(dct_blocks, sq_codebook, None, bit_rate, "SQ")

        for epsilon in EPSILONS:
            for delta in DELTAS:

                # sq-il
                received_sq = transmit_sq_il(sq_bitstream, epsilon, delta, rng)
                reconstructed_sq = reconstruct_image(received_sq, sq_codebook, None, bit_rate, sq_params, num_blocks, image.shape, "SQ")
                sq_psnr = calculate_psnr(image, reconstructed_sq)

                results.append({"system": "SQ-IL", "bit_rate": bit_rate, "epsilon": epsilon, "delta": delta, "set_index": None, "psnr": sq_psnr})

                # cosq
                cosq_codebook = load_codebook(f"codebooks/COSQ_Codebook_Delta{delta}.json")

                for set_index in COSQ_SETS[epsilon]:

                    cosq_bitstream, cosq_params, _ = compress_image(dct_blocks, cosq_codebook, set_index, bit_rate, "COSQ")
                    received_cosq = transmit_cosq(cosq_bitstream, bit_allocation, epsilon, delta, rng)
                    reconstructed_cosq = reconstruct_image(received_cosq, cosq_codebook, set_index, bit_rate, cosq_params, num_blocks, image.shape, "COSQ")
                    cosq_psnr = calculate_psnr(image, reconstructed_cosq)

                    results.append({"system": "COSQ", "bit_rate": bit_rate, "epsilon": epsilon, "delta": delta, "set_index": set_index, "psnr": cosq_psnr})

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    csv_path = os.path.join(
        OUTPUT_FOLDER,
        "psnr_results.csv"
    )

    with open(csv_path, "w", newline="") as file:

        fieldnames = [
            "system",
            "bit_rate",
            "epsilon",
            "delta",
            "set_index",
            "psnr"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()