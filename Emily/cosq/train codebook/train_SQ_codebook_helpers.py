import numpy as np


def generate_source_signal(distribution, num_samples):
    if distribution == "laplace":
        source = np.random.laplace(loc=0, scale=1 / np.sqrt(2), size=num_samples)

    elif distribution == "gaussian":
        source = np.random.normal(loc=0, scale=1, size=num_samples)

    else:
        raise ValueError("Unknown source distribution")

    return (source - np.mean(source)) / np.std(source)


def generate_initial_codebook(source, num_centroids):
    # initialize centroids
    quantiles = (np.arange(num_centroids) + 0.5) / num_centroids

    codebook = np.quantile(source, quantiles)

    return codebook


def assign_partitions(source, codebook):
    # boundaries are halfway between centroids
    boundaries = (codebook[:-1] + codebook[1:]) / 2

    # closest centroid
    partition_indices = np.searchsorted(boundaries, source)

    return partition_indices


def update_centroids(source, partition_indices, codebook):
    num_centroids = len(codebook)
    new_codebook = codebook.copy()

    for i in range(num_centroids):
        cluster = source[partition_indices == i]
        if len(cluster) > 0:
            new_codebook[i] = np.mean(cluster)

    return new_codebook


def calculate_distortion(source, codebook, partition_indices):
    reconstructed = codebook[partition_indices]

    mse = np.mean((source - reconstructed) ** 2)

    return mse


def optimize_codebook(source, initial_codebook, tolerance=1e-6, max_iterations=200):

    codebook = np.asarray(initial_codebook, dtype=np.float64).copy()

    for iteration in range(max_iterations):
        old_codebook = codebook.copy()

        # assign partitions
        partition_indices = assign_partitions(source, codebook)

        # update centroids
        codebook = update_centroids(source, partition_indices, codebook)
        codebook = np.sort(codebook)

        # convergence
        if np.max(np.abs(codebook - old_codebook)) < tolerance:
            break

    # reassign samples using final codebook
    partition_indices = assign_partitions(source, codebook)

    # distortion
    mse = calculate_distortion(source, codebook, partition_indices)

    return codebook, partition_indices, mse


def train_codebook(source, num_bits, tolerance=1e-6, max_iterations=200):

    num_centroids = 2 ** num_bits

    initial_codebook = generate_initial_codebook(source, num_centroids)

    codebook, partition_indices, mse = optimize_codebook(source, initial_codebook, tolerance, max_iterations)

    return codebook, partition_indices, mse
