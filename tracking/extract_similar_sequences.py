"""
Python script to cycle through a list of videos or temporal image directories
and track objects of interest in order to create a dataset of similar
and non-similar images for the purpose of learning an NN embedding of similar
images.

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
from tracking.helpers import *
import tracking.feature_tracker as ft


# initialize the list of reference points
cropping = False
refPt = []

def click_and_crop(event, x, y, flags, params):
    # grab references to the global variables
    global refPt, cropping

    window_name, clone_img, logging = params

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = []
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if x <= refPt[0][0] and y <= refPt[0][1]:
            refPt = []
            pass
        else:
            refPt.append((x, y))
            cropping = False

            # draw a rectangle around the region of interest
            logging.info("Drawing ROI on clone image.")
            cv2.rectangle(clone_img, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow(window_name, clone_img)


def track_similarity(input_path='./video.mp4',
                     output_path='./output',
                     is_video=True,
                     visualize=True,
                     roi_select=False,
                     save_video=False,
                     framerate=None):
    """
        Feature track through video, recording patch sequences
    """

    img_height = 360
    img_width = 640
    input_prefix = ""

    skip_rate = 1

    if is_video:
        base_fname = os.path.basename(input_path)
        video_name, video_ext = os.path.splitext(base_fname)
        input_prefix = video_name
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(os.path.join(output_path, str(video_name)+'_results.avi'), fourcc, 10, (img_width, img_height))

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = 0
        if framerate:
            skip_rate = int(orig_fps/framerate)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    else:
        dir_name = os.path.dirname(input_path)
        dir_name = os.path.basename(dir_name)
        input_prefix = "DIR_" + dir_name

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(os.path.join(output_path, str(dir_name)+'_results.avi'), fourcc, 10, (img_width, img_height))

        img_paths = sorted(glob.glob(os.path.join(input_path, '*.png')))
        img_paths_iter = iter(img_paths)


    count = 0  # current frame #
    prev_frame = None
    prev_img_file = None
    roi = None
    patch_dims = (128,128)
    patches = []
    matches_fwd = None
    prev_matches = None
    close_matches = None
    best_k_matches = 20
    # best_k_matches = 30 # scotts
    match_dist_thresh = 15
    # match_dist_thresh = 35 # scotts
    overlap_thresh = 0.4
    close_matches_percent_thresh = 0.8
    # close_matches_percent_thresh = 0.6
    sighting_thresh = 0
    # not_seen_thresh = 20
    not_seen_thresh = 20
    similar_sequences = []

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
        if count > 5000:
            print("Stopping this video early as requested at count: " + str(count))
            break

        if ret and count % skip_rate == 0:
            logging.info("FRAME #: " + str(count))

            width = int(image.shape[1]/2)
            height = int(image.shape[0]/2)
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
                              (width/2, height/2),
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

            if prev_frame:
                matches_fwd = ft.match(prev_frame.kp, prev_frame.descriptors, new_frame.kp, new_frame.descriptors)
                ft.display_matches(prev_frame.img,
                                   new_frame.img,
                                   prev_frame.kp,
                                   new_frame.kp,
                                   matches_fwd,
                                   window_name='features_tracking_fwd',
                                   visualize=visualize,
                                   best_k=best_k_matches)

                disp = ft.compute_displacement(prev_frame.kp, new_frame.kp, matches_fwd)
                disp_x, disp_y = disp

                logging.info('Displacement [x,y]: ' + str(disp))

                close_matches = []
                close_matches_count = 0
                close_matches_percentage = 0

                for m in matches_fwd[:best_k_matches]:
                    if m.distance <= match_dist_thresh:
                        close_matches.append(m)
                        close_matches_count += 1

                close_matches = matches_nms(close_matches, new_frame.kp)
                close_matches_count = len(close_matches)

                close_matches_percentage = close_matches_count/float(best_k_matches)
                logging.info("Percentage of matches that satisfy threshold: " + str(close_matches_percentage*100) + "%")

                prev_objects = []

                # Build patch-level frame sequence:
                for idx, close_match in enumerate(close_matches):
                    old_match = False
                    prev_center = prev_frame.kp[close_match.queryIdx].pt
                    new_center = new_frame.kp[close_match.trainIdx].pt

                    for old_bbox in prev_frame.objects:
                        old_center = old_bbox.kp_center

                        # if this match features a previously seen keypoint:
                        if isclose(old_center[0], prev_center[0]) and isclose(old_center[1], prev_center[1]):
                            old_match = True

                            new_fr_patch = BBox(int(new_center[0]-patch_dims[0]/2), # xmin
                                                int(new_center[1]-patch_dims[1]/2), # ymin
                                                int(new_center[0]+patch_dims[0]/2), # xmax
                                                int(new_center[1]+patch_dims[1]/2), # ymax
                                                new_center,
                                                score=close_match.distance,
                                                sightings=old_bbox.sightings+1,
                                                id=old_bbox.id,
                                                label=old_bbox.label)

                            new_frame.objects.append(new_fr_patch)
                            continue

                    if not old_match:
                        id = str(count) + '_' + str(idx)

                        prev_fr_patch = BBox(int(prev_center[0]-patch_dims[0]/2), # xmin
                                             int(prev_center[1]-patch_dims[1]/2), # ymin
                                             int(prev_center[0]+patch_dims[0]/2), # xmax
                                             int(prev_center[1]+patch_dims[1]/2), # ymax
                                             prev_center,
                                             score=close_match.distance,
                                             sightings=2,
                                             id=id,
                                             label=id)

                        new_fr_patch = BBox(int(new_center[0]-patch_dims[0]/2), # xmin
                                            int(new_center[1]-patch_dims[1]/2), # ymin
                                            int(new_center[0]+patch_dims[0]/2), # xmax
                                            int(new_center[1]+patch_dims[1]/2), # ymax
                                            new_center,
                                            score=close_match.distance,
                                            sightings=2,
                                            id=id,
                                            label=id)

                        prev_objects.append(prev_fr_patch)
                        new_frame.objects.append(new_fr_patch)

                # Keep track of patches list:
                for obj in new_frame.objects:
                    if obj.label in similar_sequences[-1].frame_dict:
                        similar_sequences[-1].frame_dict[obj.label].append((img_file, obj, dim))
                    else:
                        similar_sequences[-1].frame_dict[obj.label] = []
                        similar_sequences[-1].frame_dict[obj.label].append((img_file, obj, dim))

                for obj in prev_objects:
                    if obj.label in similar_sequences[-1].frame_dict:
                        similar_sequences[-1].frame_dict[obj.label].append((prev_img_file, obj, dim))
                    else:
                        similar_sequences[-1].frame_dict[obj.label] = []
                        similar_sequences[-1].frame_dict[obj.label].append((prev_img_file, obj, dim))

                ft.draw_objects(new_frame, visualize=visualize)

                if save_video:
                    out.write(final_img)

            if visualize:
                k = cv2.waitKey(0)
                if k == 27:
                    break

                k_2 = cv2.waitKey(0)

            if save_video:
                out.write(final_img)

            prev_frame = new_frame
            prev_img_file = img_file
            if close_matches:
                prev_matches = close_matches

    # Clean up
    if is_video:
        cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

    annotations_path = os.path.join(output_path, input_prefix, 'sequences')
    patches_annotations_path = os.path.join(output_path, input_prefix, 'patch_annotations')
    patches_path = os.path.join(output_path, input_prefix, 'patches')
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)
    if not os.path.exists(patches_annotations_path):
        os.makedirs(patches_annotations_path)
    if not os.path.exists(patches_path):
        os.makedirs(patches_path)

    for final_seq in similar_sequences:
        final_seq.write_annotation_file(input_prefix, annotations_path)
        final_seq.save_patches(input_prefix, patches_path, min_sighting=2, patch_dims=patch_dims)
        final_seq.write_patches_annotation_file(input_prefix, patches_annotations_path, min_sighting=2)

    print("Finished at count: " + str(count))
    print("Number of sequences: " + str(len(similar_sequences)))

    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract feature tracked patch sequences from video')
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
    parser.add_argument('-f', '--framerate',
                        dest='framerate',
                        help='Framerate (int)',
                        default=10)
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
            track_similarity(input_path=INPUT_PATH,
                             output_path=OUTPUT_PATH,
                             is_video=False,
                             visualize=args.visualize,
                             roi_select=False,
                             save_video=False,
                             framerate=int(args.framerate))
        else:
            filename, file_ext = os.path.splitext(INPUT_PATH)
            if file_ext in VIDEO_EXTS:
                logging.info('Path is a video.')
                track_similarity(input_path=INPUT_PATH,
                                 output_path=OUTPUT_PATH,
                                 is_video=True,
                                 visualize=args.visualize,
                                 roi_select=False,
                                 save_video=False,
                                 framerate=int(args.framerate))
            else:
                logging.error('Path is neither a directory of images or a video. Skipping...')
