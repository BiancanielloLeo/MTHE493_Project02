
import json
import os
import numpy as np

from bit_allocation import get_bit_allocation, zigzag_order

from train_codebook_helpers import generate_source_signal, train_codebook


def get_required_bits(bit_rates):

    required_bits = set()

    for rate in bit_rates:

        allocation = get_bit_allocation(rate)

        allocation_zigzag = zigzag_order(allocation)

        required_bits.update(
            int(bits)
            for bits in allocation_zigzag
            if bits > 0
        )

    return sorted(required_bits)


def generate_codebooks(bit_rates, set_index=0, num_samples=500000):

    # generate gaussian distributions
    dc_source = generate_source_signal("gaussian", num_samples)

    # generate laplacian source distributions
    ac_source = generate_source_signal("laplace", num_samples)

    required_bits = get_required_bits(bit_rates)

    codebooks = {}

    for bits in required_bits:

        # dc codebook
        dc_codebook, _, dc_mse = train_codebook(dc_source, bits)

        # ac codebook
        ac_codebook, _, ac_mse = train_codebook(ac_source, bits)

        codebooks[f"dc_rate_{bits}_set_{set_index}"] = dc_codebook.tolist()

        codebooks[f"ac_rate_{bits}_set_{set_index}"] = ac_codebook.tolist()

    # save codebook
    os.makedirs("codebook", exist_ok=True)

    output_path = ("codebook/gaussian_laplacian_codebook.json")

    with open(output_path, "w") as file:
        json.dump(codebooks, file, indent=4)


if __name__ == "__main__":

    BIT_RATES = [24]

    generate_codebooks(
        bit_rates=BIT_RATES,
        set_index=0
    )
