"""
Python script to load pickle file with patch clusters ranked based on
their similarity to the exemplar used to cluster them.

"""

# BASIC
import os
import glob
import random
import operator
import argparse
import itertools
import pickle as pkl
from datetime import datetime
random.seed(42)

# Logging
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np


def save_clusters(pickle_fpath):

    with open(pickle_fpath, 'rb') as handle:
        annotations = pkl.load(handle)

    main_dir = annotations['main_dir']

    ordered_similarity = dict()

    for key, val in annotations.items():
        if key == 'main_dir':
            continue

        img_basename = key
        best_class_id, class_name = annotations[key]['best_match_class']
        mean_similarity = annotations[key][best_class_id]
        img_path = os.path.join(main_dir, str(best_class_id) + "_" + class_name, img_basename)

        if class_name in ordered_similarity:
            ordered_similarity[class_name][img_path] = mean_similarity
        else:
            ordered_similarity[class_name] = dict()
            ordered_similarity[class_name][img_path] = mean_similarity

    for class_name, sim_dicts in ordered_similarity.items():
        print("Class: ", class_name)

        count = 0
        for img_path, value in sorted(ordered_similarity[class_name].items(), key=operator.itemgetter(1), reverse=True):
            if count % 10 == 0 or count in [1, 2, 3]:
                print(img_path, value)
                img = cv2.imread(img_path)
                cv2.imshow(class_name, img)
                cv2.waitKey(0)
                cv2.imwrite(os.path.join(main_dir, 'ordered', class_name + '_' + str(count) +'.png'), img)
            count += 1

    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Save clusters')
    parser.add_argument('-p','--pickle',
                        dest='pickle_file',
                        type=str,
                        help='<Required> Path to pickle file',
                        required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.pickle_file):
        logging.error("File path provided is not a valid file.")
        exit()

    print("Pickle file path: ", args.pickle_file)
    save_clusters(args.pickle_file)
