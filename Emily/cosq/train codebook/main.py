import numpy as np

from train_COSQ_codebook_helpers import channel_with_memory

# Generate a channel matrix for an 8-level codebook.
P = channel_with_memory(
    num_levels=8,
    epsilon=0.005,
    delta=10
)

# Every row represents a probability distribution over received indices.
print("Row sums:", P.sum(axis=1))

# The matrix should be symmetric because x XOR y equals y XOR x.
print("Symmetric:", np.allclose(P, P.T))

# All transition probabilities must be non-negative.
print("Non-negative:", np.all(P >= 0))

# Confirm that the matrix cannot be accidentally modified.
print("Read-only:", not P.flags.writeable)