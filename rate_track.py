from utils import load_images
from utils import load_as_np
from utils import convert_np_to_cp
from background import apply_bkg_to_images
from background import find_background
from astrometric_localization import detect_stars, match_to_catalogue, skycoord_to_pixels, solution_to_image_coords, apply_starmask
import numpy as np
import math
import cupy as cp
from cupyx.scipy import ndimage
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from natsort import os_sorted
from astropy.io import fits
from astropy.wcs import WCS
import contextlib

class rate_tracked_info:
    def __init__(self, data_path, num_velocities):
        self.num_velocities = int(math.sqrt(num_velocities)) ** 2
        self.data_path = data_path
        self.file_names = os_sorted([data_path + '/' + f for f in listdir(data_path) if isfile(join(data_path, f))])
        self.velocity_grid, self.vel_idx_dict = create_velocity_grid(self.num_velocities)
        self.astrometric_solution, self.preprocessed_frames = preprocess_frames(data_path)
        self.data_cube = build_data_cube(self.preprocessed_frames, self.velocity_grid)
        self.heatmap = create_heatmap(self.data_cube, self.vel_idx_dict, self.num_velocities)
        self.heatmap_peaks = detect_peaks((cp.ndarray.get(self.heatmap) - 2 * cp.ndarray.get(self.heatmap).mean()).clip(min=0))
        self.hits = get_hits(self.num_velocities, self.heatmap_peaks, self.vel_idx_dict, self.data_cube)
        self.hit_info = self.hit_calcs()

    def hit_calcs(self):
        a = int(math.sqrt(self.num_velocities)) 
        b = math.floor(a/2)
        detection_velocities = (np.transpose((self.heatmap_peaks).nonzero()) - b ) 
        formatted_velocities = np.asarray([np.asarray([x, -y]) for y, x in detection_velocities])
        fits_wcs = fits.open(self.file_names[0])
        image_wcs = WCS(fits_wcs[0])
        out = ""
        for det_vel in formatted_velocities:
            image = self.data_cube[self.vel_idx_dict[(det_vel[0], det_vel[1])]]
            index = (cp.unravel_index(cp.argmax(image, axis = None), image.shape))
            x = cp.ndarray.get(index[1])
            y = cp.ndarray.get(index[0])
            start_radec = image_wcs.pixel_to_world(x, y)
            end_radec = image_wcs.pixel_to_world((x + det_vel[0]), (y + det_vel[1]))
            out += "Detected object at moving at velocity <" + str((start_radec.ra.deg - end_radec.ra.deg)/4) + ' ' +  str((start_radec.dec.deg - end_radec.dec.deg)/4)+ "> in RA, Dec degrees/second. \n"
        return out

def preprocess_frames(data_path):  
    '''
    Preprocesses the frames for the brute force velocity search. This function removes
    the background based on the first frame, then stacks all of the background images
    for star extraction. Then, if searches for an astrometric solution, then returns
    the solution as well as the preprocessed frames.

    Input: A string of the folder name in which frames are stored

    Output: Astrometric solution and preprocessed images as CuPy arrays in a tuple
    '''
    unprocessed_images_cupy = load_images(data_path)
    unprocessed_images_numpy = load_as_np(data_path)
    initial_stack = stack_images(unprocessed_images_cupy, np.array([0, 0]))
    bkg = find_background(unprocessed_images_numpy[0])
    backgroundless_images =  convert_np_to_cp(apply_bkg_to_images(unprocessed_images_numpy, bkg))
    stars = detect_stars(cp.ndarray.get(stack_images(backgroundless_images, cp.asarray([0, 0]))))
    astrometric_solution = match_to_catalogue(stars)
    if (astrometric_solution != None and len(stars) >= 15):
        print(len(stars))
        print(len(skycoord_to_pixels(astrometric_solution)))
        with contextlib.suppress(Exception):
            s_list = solution_to_image_coords(stars, skycoord_to_pixels(astrometric_solution))[1]
            return (astrometric_solution, convert_np_to_cp(apply_starmask(s_list, initial_stack, backgroundless_images)))
    return None, backgroundless_images

def stack_images(frames, velocity_vector):
    '''
    Stacks a series of frames at a certain velocity

    Inputs:
        file_path is a list of images that are to be stacked as cupy arrays
        
        velocity_vector is a 2D CuPy array where the first entry is the x velocity 
        and the second is the y velocity. (i.e [-1, 2]). As of now, the units of vel
        are pixel/frame.

    Output:
        a stacked image of all frames.
    '''
    final_dimensions = frames[0].shape
    stacked_image = cp.zeros(shape=final_dimensions)
    iter = 0
    init_x_vel = velocity_vector[0]
    init_y_vel = velocity_vector[1]

    for image in frames:
        img_shifted = ndimage.shift(image, velocity_vector, mode='constant')
        if iter != 0:
            stacked_image += img_shifted
        iter += 1
        velocity_vector = [init_y_vel * iter, init_x_vel * iter]
    return stacked_image

def create_velocity_grid(num_velocities):
    '''
    Inputs: 
        num_velocities is an integer that determines how many velocities will
        be included in the velocity grid
    
    Output:
        a tuple which contains the velocity grid as a CuPy array and a dictionary
        that takes in a tuple of a certain velocity (ie (-1, 3)), and returns the corresponding
        index on the CuPy grid.
    '''
    a = int(math.sqrt(num_velocities)) 
    b = math.floor(a/2)
    if a % 2 == 0:
        a += 1
    velocities = cp.zeros(shape = (a * a, 2))
    vel_to_idx = {}
    idx = 0
    for i in range(-b, b+1):
        for j in range(-b, b+1):
            vel_to_idx[(i,j)] = idx
            velocities[idx] = cp.asarray([i,j])
            idx += 1
    return (velocities, vel_to_idx)

def build_data_cube(images, all_vectors):
    '''
    Creates a list of stacked frames for each velocity vector

    Inputs: file_path is a list of strings refering to the images that are to be stacked

    all_vectors is a list of 2D numpy array where the first entry is the x velocity 
        and the second is the y velocity. (i.e [-1, 2]). As of now, the units of vel
        are pixel/frame.
    
    Output: a dictionary of the final stacked frames where the velocity vector in tuple 
        form is the key for retreiving a stacked image.
    '''
    data_cube = []
    for velocity in all_vectors:
        stacked_image = stack_images(images, velocity_vector=velocity)
        data_cube.append(stacked_image)
    return data_cube

def create_heatmap(data_cube, vel_idx_dict, num_velocities):
    '''
    Based on the maximal values of the images created by the brute velocity search,
    create a heat map to see where signal tends to be concentrated the most.

    Inputs:
        num_velocities : an integer of the number of velocities that are being searched
        vel_idx_dict : a dictionary that contains the corresponding index to a velocity on the data cube
        data_cube: a list that contains all the images stacked at all velocities
    
    Output: a cupy array of that contains a heatmap of the same dimensions of the velocity grid
    '''
    a = int(np.sqrt(num_velocities))
    b = math.floor(a/2)
    heatmap = cp.zeros(shape=(a, a))
    for i in range(a):
        for j in range(a):
            heatmap[i][j] = cp.amax(data_cube[vel_idx_dict[(-b + j, b - i)]])
    return heatmap

def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #Mask of the background
    background = (image==0)
    #Padding for background that gets removed by the maximum_filter
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1) 
    #Remove background
    detected_peaks = local_max ^ eroded_background
    center = int(image.shape[0] / 2)
    detected_peaks[center][center] = 0

    return detected_peaks

def get_hits(num_velocities, heatmap_peaks, vel_idx_dict, data_cube):
    '''
    Based on the peaks of the heatmap, returns the refined images from the data cube

    Inputs:
        num_velocities : an integer of the number of velocities that are being searched
        heatmap_peaks : a np array that contains the points of interest on the velocity grid
        vel_idx_dict : a dictionary that contains the corresponding index to a velocity on the data cube
        data_cube: a list that contains all the images stacked at all velocities
    
    Output: a list of numpy arrays with the same dimensions as the original frames.
    '''
    a = int(math.sqrt(num_velocities)) 
    b = math.floor(a/2)
    detection_velocities = (np.transpose((heatmap_peaks).nonzero()) - b ) 
    formatted_velocities = np.asarray([np.asarray([x, -y]) for y, x in detection_velocities])
    hits = [cp.ndarray.get(data_cube[vel_idx_dict[(detected_vel[0], detected_vel[1])]]) for i, detected_vel in enumerate(formatted_velocities)]
    return hits

