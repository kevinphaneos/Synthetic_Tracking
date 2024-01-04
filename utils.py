from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib import patches, pyplot as plt

import numpy as np


def zscale(data, contrast=0.2):
    norm = ZScaleInterval(contrast=contrast)
    return norm(data)


def histogram_equalization(img_in, bit_depth=2**16, img_dtype=np.uint16):
    """
    Shamlessly stolen from
    https://towardsdatascience.com/histogram-equalization-a-simple-way-to-improve-the-contrast-of-your-image-bcd66596d815
    """
    cast_img = img_in.astype(img_dtype)

    # segregate color streams
    h_b, bin_b = np.histogram(cast_img.flatten(), bit_depth, [0, bit_depth - 1])

    # calculate cdf
    cdf_b = np.cumsum(h_b)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (
        (cdf_m_b - cdf_m_b.min()) * (bit_depth - 1) / (cdf_m_b.max() - cdf_m_b.min())
    )
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype(img_dtype)

    # Return the equalized image
    return cdf_final_b[cast_img]
