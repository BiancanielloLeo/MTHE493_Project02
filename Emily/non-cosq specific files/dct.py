from scipy.fft import dctn, idctn
import numpy as np

def apply_dct(blocks):
    return np.array([
        dctn(block, norm="ortho")
        for block in blocks
    ])

def apply_idct(coefficients):
    return np.array([
        idctn(block, norm="ortho")
        for block in coefficients
    ])
