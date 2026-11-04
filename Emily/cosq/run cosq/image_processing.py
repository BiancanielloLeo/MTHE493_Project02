import numpy as np
from PIL import Image

def load_image(path):
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.float64)

def divide_into_blocks(image, block_size=8):
    h, w = image.shape

    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)

    return np.array(blocks)

def reconstruct_from_blocks(blocks, image_shape, block_size=8):
    h, w = image_shape

    reconstructed_image = np.zeros((h,w), dtype=np.float64)
    block_index = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            reconstructed_image[i:i+block_size, j:j+block_size] = blocks[block_index]
            block_index += 1

    return reconstructed_image

def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)

    if image_max == image_min:
        return np.zeros_like(image, dtype=np.float64)

    return (image - image_min) / (image_max - image_min)  # Scale to 0-1 range