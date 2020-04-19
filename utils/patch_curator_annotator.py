"""
Python script to load patches and allow the user to curate (copy) the patches
into a curated dataset (for evaluation purposes).

Expects an output folder that already contains the destination class_folders.

E.g: python patch_curator_annotator.py -p /path/to/patches -o /output

Directory /output expects to have: ./1_coral, ./2_diver, ./3_sand

Label the patch sequence shown with the number of the folder (by pressing 1 for coral).
Press 'q' to exit curation.

"""

# BASIC
import os
import glob
import argparse

# LOGGING
from utils.logging_config import *

# VISION
import cv2
import numpy as np
from utils.vis_utils import *


def annotate_patch(cluster_id, input_dir, classes, class_dir_paths, img_by_img=True, scale_x=1, scale_y=1, patch_dims=(128,128), patches_to_save='all'):
    patches_path = os.path.join(input_dir, 'patches')
    main_dir = os.path.basename(input_dir)

    class_ids = list(classes.keys())
    get_ord = lambda x: ord(str(x))
    classes_count = {}
    for class_id in class_ids:
        classes_count[class_id] = 0

    if not os.path.isdir(patches_path):
        logging.error('Patches folder is missing from this directory.')
        return

    patch_folders = [f.path for f in os.scandir(patches_path) if f.is_dir()]

    if len(patch_folders) == 0:
        patch_folders = [patches_path]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    color = (0,0,255)

    for patch_idx, patch_folder in enumerate(patch_folders):
        patch_imgs = sorted(glob.glob(os.path.join(patch_folder, '*.png')))
        base_name = os.path.basename(patch_folder)

        already_saved = False
        for class_id in class_ids:
            already_saved_imgs = sorted(glob.glob(os.path.join(class_dir_paths[class_id], '*.png')))
            patch_basename = os.path.basename(patch_imgs[0])
            if len(already_saved_imgs) > 0 and os.path.join(class_dir_paths[class_id], patch_basename) in already_saved_imgs:
                already_saved = True
                classes_count[class_id] += 1
                break

        if already_saved:
            logging.info("\t\tSkipping")
            continue

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


        first_last_array = np.asarray([images[0], images[-1]])
        h_fl = first_last_array.shape[1]
        w_fl = first_last_array.shape[2]

        fl_list_2d = first_last_array.reshape(1, 2, h_fl, w_fl, 3)
        fl_list_img = concat_tile_resize(fl_list_2d)
        cv2.imshow("first_last_" + str(main_dir), fl_list_img)

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

        done = False
        labeled = False
        skip_video = False

        while not done:
            k = cv2.waitKey(0)
            if k == ord('s'):
                done = True
            elif k == ord('q'):
                done = True
                skip_video = True
            elif k in list(map(get_ord, class_ids)):
                class_id = int(chr(k))
                classes_count[class_id] += 1
                logging.info("\t\t" + str(base_name) + " labeled as class: " + str(class_id) + ' | ' + str(classes[class_id]))
                first_patch_fname = os.path.basename(patch_imgs[0])
                last_patch_fname = os.path.basename(patch_imgs[-1])
                print(class_dir_paths[class_id])
                if patches_to_save == 'all':
                    for img_idx, img in enumerate(images):
                        cv2.imwrite(os.path.join(class_dir_paths[class_id], os.path.basename(patch_imgs[img_idx])), img)
                        print(os.path.join(class_dir_paths[class_id], os.path.basename(patch_imgs[img_idx])))
                elif patches_to_save == 'first_last':
                    cv2.imwrite(os.path.join(class_dir_paths[class_id], first_patch_fname), images[0])
                    cv2.imwrite(os.path.join(class_dir_paths[class_id], last_patch_fname), images[-1])

                labeled = True
                done = True

        if patch_idx % 10 == 0:
            print("\t\t" + str(patch_idx) + "/" + str(len(patch_folders)))
            print("\t\tCounts: " + str(classes_count))
        if skip_video:
            break

    cv2.destroyAllWindows()
    return classes_count

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Curate patches manually.')
    parser.add_argument('-p','--path-list',
                        dest='list_of_paths',
                        type=str,
                        nargs='+',
                        help='<Required> Paths to dirs or lists (.txt) of directories containing patches (e.g: /path/to/list1.txt /path/to/dir /path/to/list2.txt)',
                        required=True)
    parser.add_argument('-o', '--output-path',
                        help='Path to output curated patches in (class folders)',
                        dest='output',
                        default='./classes')


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

    if not os.path.isdir(args.output):
        logging.error('Output must be a directory with classes folders already created.')
        exit()
    else:
        class_dirs = [os.path.join(args.output, o) for o in os.listdir(args.output) if os.path.isdir(os.path.join(args.output,o))]
        final_class_folders = {}
        classes = {}
        print("Classifiying patches into: ")
        for class_path in class_dirs:
            class_dir = os.path.basename(class_path)
            if '_' in class_dir:
                class_id = int(class_dir.split('_')[0])
                class_name = class_dir.split('_')[1]
            else:
                class_id = int(class_dir)
                class_name = class_dir
            print("\tClass: " + str(class_id) + ' | ' + str(class_name) + ' | ' + str(class_path))
            classes[class_id] = class_name
            final_class_folders[class_id] = os.path.join(class_path, 'patches')
            if not os.path.isdir(final_class_folders[class_id]):
                os.makedirs(final_class_folders[class_id])


    print("\n")
    classes_count = {}
    for class_id in list(classes.keys()):
        classes_count[class_id] = 0

    for cluster_id, paths in INPUT_PATHS.items():
        print('Cluster ' + str(cluster_id) + ':')
        for idx, input_path in enumerate(paths):
            print('\t' + str(idx) + '. ' + str(input_path))
            counts = annotate_patch(cluster_id, input_path, classes, final_class_folders, img_by_img=False)
            for class_id in list(classes.keys()):
                classes_count[class_id] += counts[class_id]
            print("Overall counts: " + str(classes_count))
