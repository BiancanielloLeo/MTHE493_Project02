import numpy as np

from image_processing import load_image, divide_into_blocks, reconstruct_from_blocks

from dct import apply_dct, apply_idct

from cosq import load_codebook, compress_image_cosq, decompress_image_cosq

BIT_RATE = 24
SET_INDEX = 0
BLOCK_SIZE = 8

# Load codebooks
codebook = load_codebook("codebook/cosq_codebook.json")

image = load_image("images/lena_512_gray.jpg")

blocks = divide_into_blocks(image, BLOCK_SIZE)

print("Number of blocks:", len(blocks))

dct_blocks = apply_dct(blocks)

bitstream, params, num_blocks = compress_image_cosq(
    dct_blocks,
    codebook,
    SET_INDEX,
    BIT_RATE
)

print("Compressed bitstream length:", len(bitstream))

reconstructed_dct = decompress_image_cosq(
    bitstream,
    codebook,
    SET_INDEX,
    BIT_RATE,
    params,
    num_blocks
)

reconstructed_blocks = apply_idct(
    reconstructed_dct
)

reconstructed_image = reconstruct_from_blocks(
    reconstructed_blocks, image.shape,
    BLOCK_SIZE
)


mse = np.mean(
    (image - reconstructed_image) ** 2
)

if mse == 0:
    psnr = float("inf")
else:
    psnr = 10 * np.log10(255**2 / mse)

print("Image reconstruction MSE:", mse)
print("Image reconstruction PSNR:", psnr, "dB")
print("Reconstructed image saved.")