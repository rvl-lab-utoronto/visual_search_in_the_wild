"""
Python script to test on reference images (using triplet network)
"""

# BASIC
import os
import glob
import random
import argparse
import itertools

# Logging
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
from utils.vis_utils import visualize_pairs, generate_heatmaps

import tensorflow as tf
from tensorflow.keras import Model
from network.layers.losses import *
from network.layers.metrics import *
from tensorflow.keras import backend as K
from network.siamese_nets import SiameseNetTriplet
from classification_models.tfkeras import Classifiers
from generators.similar_trios_generator import image_trio_generator, build_lbld_trios

from tqdm import tqdm

TARGET_WIDTH = 128
TARGET_HEIGHT = 128
RANDOM_SEED = 42


def setup_network(weights, network, sliding=True, imagenet=False, freeze_until=None):
    """ Function to setup  model and load its weights
        Args:
            network (str): String identifying the network architecture to use.
            weights (str): Path string to a .h5 weights file.
    """

    model = None
    siamese_net = None

    if network == 'SiameseNetTriplet':
        if imagenet:
            print("    Loading imagenet weights")
        else:
            print("    Not loading imagenet weights")
        siamese_net = SiameseNetTriplet((None,None,3), arch='resnet18', sliding=sliding, imagenet=imagenet, freeze_until=freeze_until)
        model, vector_model = siamese_net.setup_weights(weights=weights, lr=0.0006, optimizer='adam')

    return siamese_net

def forward_pass(siamese_net, inputs, visualize=True, finetuning=False):
    """ Function to predict with the model.
    Args:
        model (Keras model): Loaded and compiled keras model.
        inputs (list): List of two batches of patches, each batch with shape (None, 128, 128, 3).
    """

    vector_model = siamese_net.vector_model
    model = siamese_net.model

    test_x1 = inputs[0]
    test_x2 = inputs[1]

    patch_w = test_x2.shape[2]
    patch_h = test_x2.shape[1]

    print(test_x1.shape)
    print(test_x2.shape)

    size_multiple = int(test_x1.shape[1]/test_x2.shape[1])
    print("Size multiple: " + str(size_multiple))

    print(siamese_net.sliding)

    if siamese_net.sliding:

        # return 0, siamese_net.forward_pass([test_x1, test_x2], visualize=visualize)
        single_model = siamese_net.single_model

        img_vec_1 = single_model.predict(test_x1)
        img_vec_2 = single_model.predict(test_x2)

        print(img_vec_1.shape)
        print(img_vec_2.shape)

        dot_prod = np.dot(img_vec_1, img_vec_2.T)
        print('Dot products: ' + str(dot_prod))

        y_pred_dot = get_label(None, dot_prod, mode='dot')
        y_pred_dot = y_pred_dot.numpy()
        print('Predictions (dot): ' + str(y_pred_dot))

        img_pairs = []
        for imgs_a, imgs_b in zip(inputs[0], inputs[1]):
            img_pairs.append([imgs_a, imgs_b])

        img_pairs = np.array(img_pairs)
        print(img_pairs.shape)

        if visualize:
            visualize_pairs(img_pairs, predictions=y_pred_dot, window_name='paired_image_dot', normalized=(not finetuning))

        return y_pred_dot, dot_prod

    else:

        return siamese_net.get_similarity([test_x1, test_x2], visualize=visualize)


def sliding_patches(img, patch_size=(128,128), step_size=64):

    # slide a window across the image
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            # yield the current window
            yield (x, y, img[y:y + patch_size[1], x:x + patch_size[0]])


def test(ref_dir, query_dir, network, weights, visualize=False, sliding=True, imagenet=False, freeze_until=None, finetuning=False, save_maps=False):
    """Testing function.
    Args:
        network (str): String identifying the network architecture to use.
        weights (str): Path string to a .h5 weights file.
    """
    if imagenet:
        finetuning = True

    siamese_net = setup_network(weights, network, sliding=sliding, imagenet=imagenet, freeze_until=freeze_until)
    patch_shape = (128, 128)
    patch_w = 128
    patch_h = 128

    if siamese_net is None:
        logging.error('Trouble loading the network.')
        return

    ref_dir_basename = os.path.basename(os.path.normpath(ref_dir))
    print(ref_dir_basename)
    ref_patches = sorted(glob.glob(os.path.join(ref_dir, '*.png')))
    query_images = sorted(glob.glob(os.path.join(query_dir, '*.png')))
    query_images.extend(sorted(glob.glob(os.path.join(query_dir, '*.jpg'))))

    query_images_dict = dict()
    for query in query_images:
        frame_num = query.split('_')[-1].split('.png')[0]
        # frame_num = query.split('frame')[-1].split('.jpg')[0]
        print(query)
        # frame_num = query.split('frame')[-1].split('.png')[0]
        # frame_num = query.split('.png')[0].split('_')[-3]
        query_images_dict[int(frame_num)] = query
        # print(frame_num)

    query_images_keys = sorted(query_images_dict, key=query_images_dict.get)
    query_images_keys.sort()
    query_images = [query_images_dict[x] for x in query_images_keys]
    # print(query_images)

    patch_dict = dict()
    desc_shape = ()

    # loop over reference patches:
    for patch_idx, ref_patch_path in enumerate(ref_patches):
        ref_patch = cv2.imread(ref_patch_path)
        ref_patch = cv2.resize(ref_patch, (patch_w, patch_h))
        cv2.imshow("ref_patch", ref_patch)
        k2 = cv2.waitKey(0)

        if k2 == ord('s'):
            print(patch_idx)
            continue

        query_dict = dict()

        for img_path in query_images:
            # large image
            logging.info("Processing query: " + str(img_path))
            orig_image = cv2.imread(img_path)
            saliency_dict = dict()

            if sliding:

                # image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
                image = cv2.resize(orig_image, (512, 512))
                print("Resized image shape: " + str(image.shape))

                for (x, y, patch) in sliding_patches(image):
                    if patch.shape[0] != patch_h or patch.shape[1] != patch_w:
                        continue

                    # draw current patch:
                    if visualize:
                        clone = image.copy()
                        cv2.rectangle(clone, (x, y), (x + patch_w, y + patch_h), (0, 255, 0), 2)
                        cv2.imshow("query_image", clone)
                        cv2.imshow("query_patch", patch)
                        k = cv2.waitKey(0)
                        if k == ord('n'):
                            break
                        elif k == 27:
                            return

                    patch_np = np.zeros((1, patch_w, patch_h, 3))
                    ref_patch_np = np.zeros((1, patch_w, patch_h, 3))

                    if not finetuning:
                        patch_np[0] = patch/255.
                        ref_patch_np[0] = ref_patch/255.
                    else:
                        ResNet18, preprocess_input = Classifiers.get('resnet18')
                        patch_np[0] = preprocess_input(patch)
                        ref_patch_np[0] = preprocess_input(ref_patch)

                    inputs = [patch_np, ref_patch_np]
                    pred_dot = forward_pass(siamese_net, inputs, visualize=visualize, finetuning=finetuning)
                    print("Dot product 3: " + str(pred_dot))
                    saliency_dict[(x,y)] = (0, pred_dot[3])

            else:

                # img_h = (orig_image.shape[0] // patch_h)*patch_h
                # img_w = (orig_image.shape[1] // patch_w)*patch_w
                img_h = 512
                img_w = 512

                if img_w >= img_h:
                    image = cv2.resize(orig_image, (img_h, img_h))
                    image_np = np.zeros((1, img_h, img_h, 3))
                else:
                    image = cv2.resize(orig_image, (img_w, img_w))
                    image_np = np.zeros((1, img_w, img_w, 3))

                # print("Resized image shape: " + str(image.shape))

                ref_patch_np = np.zeros((1, patch_w, patch_h, 3))

                if not finetuning:
                    image_np[0] = image/255.
                    ref_patch_np[0] = ref_patch/255.
                else:
                    ResNet18, preprocess_input = Classifiers.get('resnet18')
                    image_np[0] = preprocess_input(image)
                    ref_patch_np[0] = preprocess_input(ref_patch)

                inputs = [image_np, ref_patch_np]
                # print("Inputs shapes: ")
                # print("Image a: " + str(inputs[0].shape))
                # print("Image b: " + str(inputs[1].shape))
                saliency_dict, desc_shape = forward_pass(siamese_net, inputs, visualize=visualize, finetuning=finetuning)
                # saliency_dict[(x,y)] = (pred_dist, pred_dot)

            query_dict[(img_path)] = saliency_dict

            print("Generating heatmap")
            if not visualize:
                _, _, img_w_hmap = generate_heatmaps(image, saliency_dict, desc_shape=desc_shape, sliding=sliding, get_normed=False, wait_key_dur=-1)
            else:
                _, _, img_w_hmap = generate_heatmaps(image, saliency_dict, desc_shape=desc_shape, sliding=sliding, get_normed=False, wait_key_dur=0)

            if save_maps:
                # print(img_path)
                ext = img_path.split('.')[1]
                img_path_basename = os.path.basename(img_path)
                dir_name = os.path.dirname(img_path)
                img_w_hmap = img_w_hmap[64:448, 64:448]
                hmap_dir_path = os.path.join(dir_name, ref_dir_basename)
                if not os.path.exists(hmap_dir_path):
                    os.makedirs(hmap_dir_path)

                cv2.imwrite(os.path.join(hmap_dir_path, img_path_basename.replace('.' + ext, '_hmap_PATCH' + str(patch_idx) + '.' + ext)), img_w_hmap)

        patch_dict[patch_idx] = query_dict

    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Testing script for generating a heatmap')
    parser.add_argument('-r','--ref-dir',
                        dest='refs_dir',
                        type=str,
                        help='<Required> Path to dir containing reference patches (e.g: /path/to/ref_patches/imgs1/)',
                        required=True)
    parser.add_argument('-q','--query-dirs',
                        dest='query_dir',
                        type=str,
                        help='<Required> Path to dir containing query images (e.g: /path/to/queries/imgs1/)',
                        required=True)
    parser.add_argument('-w', '--weights',
                        dest='weights',
                        help='Keras model',
                        default=None)
    parser.add_argument('-nw', '--network',
                        help='Network name',
                        dest='network',
                        default='SiameseNetTriplet')
    parser.add_argument('-v --visualize',
                        dest='visualize',
                        action='store_true')
    parser.add_argument('-s --sliding',
                        dest='sliding',
                        action='store_true')
    parser.add_argument('-i --imagenet',
                        dest='imagenet',
                        action='store_true')
    parser.add_argument('-fu', '--freeze-until',
                        dest='freeze_until',
                        type=int,
                        default=None)
    parser.add_argument('-ft --finetuning',
                        dest='finetuning',
                        action='store_true')
    parser.add_argument('-sm --save-hmaps',
                        dest='save_maps',
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    REF_PATH = args.refs_dir
    QUERY_PATH = args.query_dir

    print("Reference directory: " + str(REF_PATH))
    print("Query directory: " + str(QUERY_PATH))

    test(REF_PATH,
         QUERY_PATH,
         args.network,
         args.weights,
         visualize=args.visualize,
         sliding=args.sliding,
         imagenet=args.imagenet,
         freeze_until=args.freeze_until,
         finetuning=args.finetuning,
         save_maps=args.save_maps)
