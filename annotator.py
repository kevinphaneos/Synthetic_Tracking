import numpy as np
from utils import zscale
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cupy as cp
from natsort import os_sorted #NEEDED FOR LINUX
import os.path
from PIL import Image
import os
from rate_track import preprocess_frames, stack_images
import json

"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    FILE_PATH = 'Ten_Thirteen'

    save_path = '/home/kevin/Astrophotography_WSL/training_set/'
    save_path_images = save_path + 'images'
    save_path_labels = save_path + 'labels'

    counter = 538
    for folder in listdir(FILE_PATH)[555:]:
        image_folder_path = FILE_PATH + '/' + folder + '/ImageFiles'
        jsons = os_sorted([FILE_PATH + '/' + folder + '/Annotations' + '/' + f for f in os_sorted(listdir(FILE_PATH + '/' + folder + '/Annotations'))])
        #Get first
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        if 'objects' in data['data']:
            point_init = data['data']['objects'][0]['pixels'][0]

        #Get Last point
        with open(jsons[5]) as json_file:
            data = json.load(json_file)
        if 'objects' in data['data']:
            point_last = data['data']['objects'][0]['pixels'][0]

        shift = cp.asarray([round((point_init[1] - point_last[1])/6), round((point_init[0] - point_last[0])/6)])
        solution, frames = preprocess_frames(image_folder_path)
        
        annotation = '0 ' + str(float(point_init[1]/512)) + ' ' + str(float(point_init[0]/512)) + ' ' + str(float(14/512)) +  ' ' + str(float(14/512))
        name_of_file = str(counter)
        completeName = os.path.join(save_path_labels, name_of_file + ".txt")         
        file1 = open(completeName, "w")
        file1.write(annotation)
        file1.close()

        completeName = os.path.join(save_path_images, name_of_file + ".png") 
        solution = cp.ndarray.get(stack_images(frames, shift))
        im = Image.fromarray(solution).convert("L")
        im.save(completeName)
        counter += 1
        print(counter)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()