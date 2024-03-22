import json
from os import listdir
from natsort import os_sorted
from rate_track import rate_tracked_info
import numpy as np
import pandas as pd

"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    PIXEL_ERROR = 3
    point = None
    iteration = 0
    PATH = "HighBright"

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    all_preds = []
    all_points = []
    
    for folder in listdir(PATH)[150:]:
        preds = []
        points = []
        point = None
        first = os_sorted(listdir(PATH + "/" + folder + '/Annotations/'))[0]
        with open(PATH + "/" + folder + '/Annotations/' + first) as json_file:
            data = json.load(json_file)
        if 'objects' in data['data']:
            if len(data['data']['objects']) != 0:
                point = data['data']['objects'][0]['pixels'][0]
                points.append(point)
            else:
                point = None

        results = rate_tracked_info(PATH + '/' + folder +'/ImageFiles', 1000)
        hits = results.hits
        if len(hits) == 0 and point == None:
            TN += 1
        elif len(hits) == 0 and point != None:
            FN += 1
        else:
            for hit in hits:
                idx = np.unravel_index(hit.argmax(), hit.shape)
                preds.append(np.array([idx[0], idx[1]]))
                if np.abs(point[0] - idx[0]) <= PIXEL_ERROR and np.abs(point[1] - idx[1]) <= PIXEL_ERROR:
                    TP += 1
                else:
                    FP += 1
        all_preds.append(preds)
        all_points.append(points)
        print(TP, TN, FP, FN)
    print(TP, TN, FP, FN)
    columns = {'preds': all_preds,
               'labels': all_points}
    df = pd.DataFrame(columns)
    df.to_csv(PATH + '4_Results', sep='\t', encoding='utf-8')


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()