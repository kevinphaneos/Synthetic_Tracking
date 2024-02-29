from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib import patches, pyplot as plt
from os import listdir
from os.path import isfile, join
import cupy as cp
from natsort import os_sorted
import numpy as np

def load_as_np(file_path):
    file_names = os_sorted([file_path + '/' + f for f in listdir(file_path) if isfile(join(file_path, f))])
    return [fits.open(file)[0].data for file in file_names]

def load_images(file_path):
    images = []
    file_names = os_sorted([file_path + '/' + f for f in listdir(file_path) if isfile(join(file_path, f))])
    for file in file_names:
        unprocessed_img = fits.open(file)[0].data
        images.append(cp.asarray(unprocessed_img))
    return images

def convert_np_to_cp(list):
    cupies = []
    for image in list:
        cupies.append(cp.asarray(image))
    return cupies

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

