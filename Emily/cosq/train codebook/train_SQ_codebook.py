import json
import os
import numpy as np

from bit_allocation import get_bit_allocation, zigzag_order

from train_SQ_codebook_helpers import generate_source_signal, train_codebook


def generate_codebooks(bit_rates, num_samples=500000):

    # generate gaussian distributions
    dc_source = generate_source_signal("gaussian", num_samples)

    # generate laplacian source distributions
    ac_source = generate_source_signal("laplace", num_samples)

    codebooks = {}

    for bits in bit_rates:

        # dc codebook
        dc_codebook, _, dc_mse = train_codebook(dc_source, bits)

        # ac codebook
        ac_codebook, _, ac_mse = train_codebook(ac_source, bits)

        codebooks[f"dc_rate_{bits}"] = dc_codebook.tolist()
        codebooks[f"ac_rate_{bits}"] = ac_codebook.tolist()

    # save codebook
    os.makedirs("codebooks", exist_ok=True)

    with open("codebooks/SQ_Codebook.json", "w") as file:
        json.dump(codebooks, file, indent=4)


if __name__ == "__main__":

    rates = [3, 4, 5, 6, 7, 8]

    generate_codebooks(
        bit_rates=rates
    )
