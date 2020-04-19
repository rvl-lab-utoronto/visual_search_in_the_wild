"""
Python script to load patch sequences that were tracked through feature tracking.
If visualizing, it will show a mosaic of a subset of the patches in the sequence:
    - any key to cycle through different sequences
    - esc to skip to the next folder of sequences

"""

# BASIC
import os
import sys
import glob
import argparse

from utils.logging_config import *

# VISION
import cv2
import numpy as np
from utils.vis_utils import *


def load_patch(cluster_id, input_dir, img_by_img=True, scale_x=1, scale_y=1, patch_dims=(128,128), visualize=True):

    save_summary = False
    patches_path = os.path.join(input_dir, 'patches')
    main_dir = os.path.basename(input_dir)

    longest_patch_seq = 0
    patch_seq_avg_len = 0
    list_of_patch_seq_len = []
    num_patch_folders = 0

    if not os.path.isdir(patches_path):
        logging.error('Patches folder is missing from this directory.')
        return

    patch_folders = [f.path for f in os.scandir(patches_path) if f.is_dir()]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    color = (0,0,255)

    if len(patch_folders) == 0:
        patch_folders = [patches_path]
        print("Saving summary")
        save_summary = True
        # return

    for patch_folder in patch_folders:
        patch_imgs = sorted(glob.glob(os.path.join(patch_folder, '*.png')))
        base_name = os.path.basename(patch_folder)

        images = []

        for patch in patch_imgs:
            image = cv2.imread(patch)
            h, w, d = image.shape
            if h == patch_dims[1] and w == patch_dims[0]:
                resized = cv2.resize(image,(int(w/scale_x),int(h/scale_y)))
                images.append(resized)

                if img_by_img:
                    cv2.imshow(base_name, image)
                    cv2.waitKey(0)

        if img_by_img:
            cv2.destroyAllWindows()

        if len(images) < 2:
            continue

        patch_seq_avg_len += len(images)
        num_patch_folders += 1
        list_of_patch_seq_len.append(len(images))
        if len(images) > longest_patch_seq:
            longest_patch_seq = len(images)

        if len(images) < 4:
            im_array = np.asarray(images)
            h = im_array.shape[1]
            w = im_array.shape[2]

            im_list_2d = im_array.reshape(1, len(images), h, w, 3)
            summary_image = concat_tile_resize(im_list_2d)
        else:
            sqr_len = int(np.sqrt(len(images)))
            total_num = sqr_len*sqr_len
            im_array = np.asarray(images[:total_num])
            h = im_array.shape[1]
            w = im_array.shape[2]

            im_list_2d = im_array.reshape(sqr_len, sqr_len, h, w, 3)
            summary_image = concat_tile_resize(im_list_2d)

        if visualize:
            cv2.putText(summary_image,
                        'patch_id: ' + str(base_name),
                        (25, 25),
                        font,
                        fontScale,
                        color,
                        lineType)
            cv2.putText(summary_image,
                        '# of patches: ' + str(len(images)),
                        (25, 50),
                        font,
                        fontScale,
                        color,
                        lineType)


            cv2.imshow("summary_" + str(main_dir), summary_image)
            k = cv2.waitKey(0)
            if k == 27:
                break
        if save_summary:
            cv2.imwrite(os.path.join(input_dir, "summary_" + str(main_dir) +'.png'), summary_image)

    if num_patch_folders > 0:
        patch_seq_avg_len = patch_seq_avg_len/num_patch_folders

    cv2.destroyAllWindows()

    return longest_patch_seq, patch_seq_avg_len, list_of_patch_seq_len, num_patch_folders

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Load patches in a patch sequence and visualize them')
    parser.add_argument('-p','--path-list',
                        dest='list_of_paths',
                        type=str,
                        nargs='+',
                        help='<Required> Paths to dirs or lists (.txt) of directories containing patches (e.g: /path/to/list1.txt /path/to/dir /path/to/list2.txt)',
                        required=True)
    parser.add_argument('-v --visualize',
                        dest='visualize',
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    PATHS_LIST = args.list_of_paths

    INPUT_PATHS = {}

    for idx, path in enumerate(PATHS_LIST):
        if os.path.isdir(path):
            # directory of images
            INPUT_PATHS[idx] = [path]
        else:
            # video or .txt containing list of videos
            base_name = os.path.basename(path)
            fname, ext = os.path.splitext(base_name)

            if ext == '.txt':
                dir_list = []
                with open(path, "r") as dir_list_file:
                    for cnt, line in enumerate(dir_list_file):
                        line = line.strip()
                        if os.path.isdir(line):
                            # directory of images
                            dir_list.append(line)
                INPUT_PATHS[idx] = dir_list

    print("\nList of directories to process: ")
    for cluster_id, paths in INPUT_PATHS.items():
        print('    Cluster ' + str(cluster_id) + ':')
        for idx, input_path in enumerate(paths):
            print('        ' + str(idx) + '. ' + str(input_path))
    print("\n")

    patch_seqs_lens = dict()
    all_seqs = list()
    patch_seqs_means = list()


    for cluster_id, paths in INPUT_PATHS.items():
        print('Cluster ' + str(cluster_id) + ':')
        patch_seqs_lens[cluster_id] = dict()
        for idx, input_path in enumerate(paths):
            print('    ' + str(idx) + '. ' + str(input_path))
            load_patch(cluster_id, input_path, img_by_img=False, visualize=args.visualize)
