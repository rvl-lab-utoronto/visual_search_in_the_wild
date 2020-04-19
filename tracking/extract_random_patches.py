"""
Python script to cycle through a list of videos or temporal image directories
and extract random patches.
"""

# BASIC
import os
import glob
import random
import argparse

# LOGGING
from utils.logging_config import *

# VISION
import cv2
import numpy as np
from utils.vis_utils import *
from tracking.helpers import *
import tracking.feature_tracker as ft


def extract_patches(input_path='./video.mp4',
                    output_path='./output',
                    is_video=True,
                    visualize=True,
                    framerate=None):
    """

    """

    img_height = 360
    img_width = 640
    input_prefix = ""

    skip_rate = 1

    if is_video:
        base_fname = os.path.basename(input_path)
        video_name, video_ext = os.path.splitext(base_fname)
        input_prefix = video_name

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = 0
        if framerate:
            skip_rate = int(orig_fps/framerate)

        # print("Skipping ahead by " + str(start_frame) + " frames")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    else:
        dir_name = os.path.basename(input_path)
        input_prefix = "DIR_" + dir_name

        img_paths = sorted(glob.glob(os.path.join(input_path, '*.png')))
        img_paths_iter = iter(img_paths)


    count = 0  # current frame #
    prev_frame = None
    patch_dims = (128,128)
    patches = []
    overlap_thresh = 0.4
    similar_sequences = []
    num_patches_per_frame = 10

    while True:

        if is_video:
            ret, image = cap.read()
        else:
            try:
                img_file = next(img_paths_iter)
            except StopIteration:
                logging.warning('Ran out of frames.')
                break
            image = cv2.imread(img_file)
            if image is not None:
                ret = True
            else:
                ret = False

        count += 1

        if count > 10000:
            print("Stopping this video early as requested at count: " + str(count))
            break

        if ret and count % skip_rate == 0:
            logging.info("FRAME #: " + str(count))

            if image.shape[1] > patch_dims[1]*2 and image.shape[0] > patch_dims[0]*2:
                width = int(image.shape[1]/2)
                height = int(image.shape[0]/2)
            else:
                width = int(image.shape[1])
                height = int(image.shape[0])

            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            clone = image.copy()

            if is_video:
                ts = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC if hasattr(cv2, 'cv') else cv2.CAP_PROP_POS_MSEC)
                video_imgs_path = os.path.join(output_path, video_name, 'images')
                img_file = os.path.join(video_imgs_path, video_name + "_" + str(count) + ".png")
                if not os.path.exists(video_imgs_path):
                    os.makedirs(video_imgs_path)
                cv2.imwrite(img_file, image)

            frame_bbox = BBox(0,
                              0,
                              width,
                              height,
                              score=1,
                              label='frame')

            vis_image = np.copy(image)
            new_frame = ft.Frame(image, bboxes=[])

            ft.draw_objects(new_frame, color=(0, 0, 255), visualize=visualize)

            # Start tracking the first sequence
            if len(similar_sequences) == 0:
                logging.info("Starting to track the first sequence..")
                first_sim_seq = SimilarSequence(id="0", w=width, h=height)
                first_sim_seq.sequence.append(img_file)
                similar_sequences.append(first_sim_seq)

            for patch_idx in range(num_patches_per_frame):
                x_center = random.randint(0+patch_dims[0], width-patch_dims[0])
                y_center = random.randint(0+patch_dims[1], height-patch_dims[1])

                xmin = x_center - int(patch_dims[0]/2)
                ymin = y_center - int(patch_dims[1]/2)
                xmax = x_center + int(patch_dims[0]/2)
                ymax = y_center + int(patch_dims[1]/2)

                # build random patch
                random_patch = BBox(xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                    score=1.0,
                                    id=str(count)+'_'+str(patch_idx),
                                    label=str(count)+'_'+str(patch_idx))

                new_frame.objects.append(random_patch)

            new_frame.objects = nms(new_frame.objects)

            # Keep track of patches list:
            for obj in new_frame.objects:
                if str(count) in similar_sequences[-1].frame_dict:
                    similar_sequences[-1].frame_dict[str(count)].append((img_file, obj, dim))
                else:
                    similar_sequences[-1].frame_dict[str(count)] = []
                    similar_sequences[-1].frame_dict[str(count)].append((img_file, obj, dim))

            if visualize:
                k = cv2.waitKey(0)
                if k == 27:
                    break
                k_2 = cv2.waitKey(0)

    # Clean up
    if is_video:
        cap.release()

    cv2.destroyAllWindows()

    patches_annotations_path = os.path.join(output_path, input_prefix, 'patch_annotations')
    patches_path = os.path.join(output_path, input_prefix, 'patches')
    if not os.path.exists(patches_annotations_path):
        os.makedirs(patches_annotations_path)
    if not os.path.exists(patches_path):
        os.makedirs(patches_path)

    for final_seq in similar_sequences:
        final_seq.save_patches(input_prefix, patches_path, min_sighting=0)
        final_seq.write_patches_annotation_file(input_prefix, patches_annotations_path, min_sighting=0)

    print("Finished at count: " + str(count))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract random patches from video/temporal image directory')
    parser.add_argument('-p','--path-list',
                        dest='list_of_paths',
                        type=str,
                        nargs='+',
                        help='<Required> Paths to videos (.mp4, .MP4, .avi), images directory or a list of either (e.g: /path/to/video/video1.mp4 /path/to/images/ /path/to/list.txt)',
                        required=True)
    parser.add_argument('-o', '--output-dir',
                        dest='output_path',
                        type=str,
                        help='<Required> Path to save output in. (e.g: /path/to/output)',
                        required=True)
    parser.add_argument('-v --visualize', dest='visualize', action='store_true')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    PATHS_LIST = args.list_of_paths
    OUTPUT_PATH = args.output_path

    VIDEO_EXTS = ['.mp4', '.MP4', '.avi']

    INPUT_PATHS = []

    for path in PATHS_LIST:
        if os.path.isdir(path):
            # directory of images
            INPUT_PATHS.append(path)
        else:
            # video or .txt containing list of videos
            base_name = os.path.basename(path)
            fname, ext = os.path.splitext(base_name)

            if ext in VIDEO_EXTS:
                INPUT_PATHS.append(path)
            elif ext == '.txt':
                with open(path, "r") as video_list_file:
                    for cnt, line in enumerate(video_list_file):
                        line = line.strip()
                        if os.path.isdir(line):
                            # directory of images
                            INPUT_PATHS.append(line)
                        else:
                            base_name = os.path.basename(line)
                            fname, ext = os.path.splitext(base_name)
                            if ext in VIDEO_EXTS:
                                INPUT_PATHS.append(line)

    print("\nList of videos to process: ")
    for idx, input_path in enumerate(INPUT_PATHS):
        print('    ' + str(idx) + '. ' + input_path)
    print("\n")

    for idx, INPUT_PATH in enumerate(INPUT_PATHS):
        print('[' + str(idx) + ']: ' + INPUT_PATH)
        if os.path.isdir(INPUT_PATH):
            logging.info('Path is a directory.')
            extract_patches(input_path=INPUT_PATH,
                            output_path=OUTPUT_PATH,
                            is_video=False,
                            visualize=args.visualize,
                            framerate=20)
        else:
            filename, file_ext = os.path.splitext(INPUT_PATH)
            if file_ext in VIDEO_EXTS:
                logging.info('Path is a video.')
                extract_patches(input_path=INPUT_PATH,
                                output_path=OUTPUT_PATH,
                                is_video=True,
                                visualize=args.visualize,
                                framerate=5)
            else:
                logging.error('Path is neither a directory of images or a video. Skipping...')
