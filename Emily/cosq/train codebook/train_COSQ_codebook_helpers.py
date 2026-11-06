import numpy as np
from functools import lru_cache

def generate_source_signal(distribution, num_samples):
    if distribution == "laplace":
        source = np.random.laplace(loc=0, scale=1 / np.sqrt(2), size=num_samples)

    elif distribution == "gaussian":
        source = np.random.normal(loc=0, scale=1, size=num_samples)

    else:
        raise ValueError("Unknown source distribution")

    return (source - np.mean(source)) / np.std(source)


@lru_cache(maxsize=None)
def channel_with_memory(num_levels, epsilon, delta):

    # number of bits needed to represent each index
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
    p_z = np.where(bits[:, 0] == 1, epsilon, 1 - epsilon)

    # probabilities of all remaining bit error states
    for i in range(1, n):
        p_z *= P_z[bits[:, i - 1], bits[:, i]]

    # codeword transition matrix
    P = p_z[z[:, None] ^ z[None, :]]

    # read-only
    P.setflags(write=False)

    return P


def get_initial_partitions(source, codebook):

    # midpoints between centroid pairs
    boundaries = (codebook[:-1] + codebook[1:]) / 2

    # integer partition index for samples
    partition_indices = np.searchsorted(boundaries, source)

    return partition_indices


def assign_partitions(source, codebook, P, mapping):
    num_partitions = len(mapping)

    # select channel transition probabilities for each partition
    P_mapping = P[mapping, :]

    # quantities that do not depend on samples
    a = P_mapping @ codebook
    b = P_mapping @ (codebook ** 2)

    # expected distortion for samples and partitions
    source_squared = source ** 2

    best_distortion = np.full(len(source), np.inf)
    partition_indices = np.zeros(len(source), dtype=np.int64)

    for i in range(num_partitions):

        # distortion assigning all samples to partition i
        distortion = source_squared - 2 * source * a[i] + b[i]

        # best partition has lower distortion
        better = distortion < best_distortion

        best_distortion[better] = distortion[better]
        partition_indices[better] = i

    return partition_indices


def update_centroids(source, partition_indices, codebook, P, mapping):
    num_partitions = len(mapping)
    num_centroids = len(codebook)
    new_codebook = codebook.copy()

    # count samples per partition
    counts = np.bincount(partition_indices, minlength=num_partitions)

    # sum of samples per partition
    sums = np.bincount(partition_indices, weights=source, minlength=num_partitions)

    # channel rows corresponding to current mapping
    P_mapping = P[mapping, :]

    # weighted sum and count for every received index
    numerator = P_mapping.T @ sums
    denominator = P_mapping.T @ counts

    valid = denominator > 0

    # update centroids
    new_codebook[valid] = numerator[valid] / denominator[valid]

    return new_codebook


def calculate_distortion(source, codebook, partition_indices, P, mapping):

    num_partitions = len(mapping)

    # sufficient statistics for partitions
    counts = np.bincount(partition_indices, minlength=num_partitions)
    sums = np.bincount(partition_indices, weights=source, minlength=num_partitions)
    squared_sums = np.bincount(partition_indices, weights=source ** 2, minlength=num_partitions)

    # channel rows corresponding to mapping
    P_mapping = P[mapping, :]

    # total squared error for every combination
    squared_errors = (squared_sums[:, None] - 2 * sums[:, None] * codebook[None, :] + counts[:, None] * (codebook[None, :] ** 2))

    # weight  error by channel probability
    total_distortion = np.sum(P_mapping * squared_errors)

    # average MSE
    return max(0.0, total_distortion / len(source))


def optimize_codebook(source, initial_codebook, P, mapping, tolerance=1e-6, max_iterations=200):

    codebook = np.asarray(initial_codebook, dtype=np.float64).copy()

    for iteration in range(max_iterations):
        old_codebook = codebook.copy()

        # assign partitions
        partition_indices = assign_partitions(source, codebook, P, mapping)

        # update centroids
        codebook = update_centroids(source, partition_indices, codebook, P, mapping)

        # convergence
        if np.max(np.abs(codebook - old_codebook)) < tolerance:
            break

    # reassign samples using final codebook
    partition_indices = assign_partitions(source, codebook, P, mapping)

    # distortion
    mse = calculate_distortion(source, codebook, partition_indices, P, mapping)

    return codebook, partition_indices, mse


def train_codebook(source, initial_codebook, P, mapping, tolerance=1e-6, max_iterations=200):

    codebook, partition_indices, mse = optimize_codebook(source, initial_codebook, P, mapping, tolerance, max_iterations)

    return codebook, partition_indices, mse


def simulated_annealing(source, partition_indices, codebook, P, initial_mapping, initial_temperature=0.01, cooling_rate=0.95, minimum_temperature=1e-5, iterations_per_temperature=200):

    rng = np.random.default_rng(None)

    num_partitions = len(initial_mapping)

    # partition statistics
    counts = np.bincount(partition_indices, minlength=num_partitions)
    sums = np.bincount(partition_indices, weights=source, minlength=num_partitions)
    squared_sums = np.bincount(partition_indices, weights=source ** 2, minlength=num_partitions)


    # total squared error for every combination
    squared_errors = (squared_sums[:, None] - 2 * sums[:, None] * codebook[None, :] + counts[:, None] * (codebook[None, :] ** 2))

    # cost[i, k] is expected distortion partition i for transmitted index k
    cost = squared_errors @ P.T

    # supplied mapping
    mapping = np.asarray(initial_mapping, dtype=np.int64).copy()

    # total average distortion
    current_distortion = np.sum(cost[np.arange(num_partitions), mapping]) / len(source)

    best_mapping = mapping.copy()
    best_distortion = current_distortion

    temperature = initial_temperature

    # cool until temperature reaches minimum
    while temperature > minimum_temperature:
        for _ in range(iterations_per_temperature):

            # select two different partitions
            i, j = rng.choice(num_partitions, size=2, replace=False)

            # change in distortion if transmitted indices swapped
            old_cost = cost[i, mapping[i]] + cost[j, mapping[j]]
            new_cost = cost[i, mapping[j]] + cost[j, mapping[i]]

            delta_distortion = (new_cost - old_cost) / len(source)

            # accept improvements, accept worse to avoid a local minimum
            if (delta_distortion < 0 or rng.random() < np.exp(-delta_distortion / temperature)):
                mapping[i], mapping[j] = mapping[j], mapping[i]

                current_distortion += delta_distortion

                # best mapping so far
                if current_distortion < best_distortion:
                    best_distortion = current_distortion
                    best_mapping = mapping.copy()

        # reduce temperature after all swap attempts
        temperature *= cooling_rate

    return best_mapping