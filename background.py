import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

def find_background(image):
    '''
    Calculates the background of an image using photutils.

    Input: The stacked frames as a numpy array

    Output: The background as a 2DBKG of the same dimensions as the input
    '''
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), filter_size=(3, 3),
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg

def apply_correction(image, bkg):
    '''
    Applies background subtraction, prevents negative values in the output
    using clip.

    Inputs: The image as a numpy array and the background as a numpy array of the
            same dimensions

    Outputs: The background correction image, of the same dimensions of the input
    '''
    return (image - bkg.background).clip(min=0)

def apply_bkg_to_images(images, bkg):
    '''
    Applies a given background to a list of images

    Inputs: A list of images (numpy arrays), and a 2DBKG

    Output: A list of background corrected images
    '''
    bkg_images = []
    for image in images:
        bkg_images.append(apply_correction(image, bkg))
    return bkg_images

