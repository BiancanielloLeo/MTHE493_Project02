import json
import os
import numpy as np

from bit_allocation import get_bit_allocation, zigzag_order

from train_COSQ_codebook_helpers import generate_source_signal, channel_with_memory, get_initial_partitions, simulated_annealing, train_codebook

def load_lloyd_max_codebook(input_path):
    with open(input_path, "r") as file:
        return json.load(file)


def generate_one_codebook(source, initial_codebook, epsilon_values, delta=10):

    # existing Lloyd-Max codebook
    codebook = initial_codebook.copy()

    # initial partitions
    partition_indices = get_initial_partitions(source, codebook)

    # identity mapping
    mapping = np.arange(len(codebook))

    codebooks = []
    distortions = []

    # optimize for each channel error rate
    for epsilon in epsilon_values:

        # channel transition matrix
        P = channel_with_memory(len(codebook), epsilon, delta)

        # simulated annealing
        mapping = simulated_annealing(source, partition_indices, codebook, P, mapping)

        # optimize partitions and centroids
        codebook, partition_indices, mse = train_codebook(source, codebook, P, mapping)

        codebooks.append(codebook.copy())
        distortions.append(mse)

    return codebooks, distortions

def generate_all_codebooks(initial_codebook, bit_rates, epsilon_values, delta, num_samples=500000):
    # generate gaussian distributions
    dc_source = generate_source_signal("gaussian", num_samples)

    # generate laplacian source distributions
    ac_source = generate_source_signal("laplace", num_samples)

    codebooks = {}

    for bits in bit_rates:

        # Lloyd-Max codebooks
        dc_initial = np.array(initial_codebook[f"dc_rate_{bits}"])
        ac_initial = np.array(initial_codebook[f"ac_rate_{bits}"])

        # Lloyd-Max codebooks set 0
        codebooks[f"dc_rate_{bits}_set_0"] = dc_initial.tolist()
        codebooks[f"ac_rate_{bits}_set_0"] = ac_initial.tolist()

        # dc cosq codebooks
        dc_results, dc_distortions = generate_one_codebook(dc_source, dc_initial, epsilon_values, delta)

        # ac cosq codebooks
        ac_results, ac_distortions = generate_one_codebook(ac_source, ac_initial, epsilon_values, delta)

        for stage in range(len(epsilon_values)):
            codebooks[f"dc_rate_{bits}_set_{stage + 1}"] = (dc_results[stage].tolist())
            codebooks[f"ac_rate_{bits}_set_{stage + 1}"] = (ac_results[stage].tolist())

    # save codebook
    os.makedirs("codebooks", exist_ok=True)

    with open(f"codebooks/COSQ_Codebook_Delta{delta}.json", "w") as file:
        json.dump(codebooks, file, indent=4)


if __name__ == "__main__":

    # existing Lloyd-Max codebooks
    with open("codebooks/SQ_Codebook.json", "r") as file:
        sq_codebook = json.load(file)

    rates = [3, 4, 5, 6, 7, 8]
    epsilons = [0.005, 0.01, 0.05, 0.1, 0.05, 0.01, 0.005]
    deltas = [0, 5, 10]

    for delta in deltas:
        generate_all_codebooks(initial_codebook=sq_codebook, bit_rates=rates, epsilon_values=epsilons, delta=delta)
