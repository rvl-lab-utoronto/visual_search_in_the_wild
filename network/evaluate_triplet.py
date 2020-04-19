"""
Python script to evaluate patches (similar vs not) trained on triplet siamese

"""

# BASIC
import os
import glob
import random
import argparse
import itertools
import pickle as pkl
from datetime import datetime

# LOGGING
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
from utils.vis_utils import visualize_trios, visualize_pairs

import tensorflow as tf
from tensorflow.keras import Model
from network.layers.losses import *
from network.layers.metrics import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from network.siamese_nets import SiameseNetTriplet
from generators.similar_trios_generator import image_trio_generator, build_lbld_trios

from sklearn.metrics import classification_report
# from sklearn.manifold import TSNE

from tqdm import tqdm

TARGET_WIDTH = 128
TARGET_HEIGHT = 128
RANDOM_SEED = 42

def evaluate(eval_lbls_trios, network, weights, imagenet=False, freeze_until=None, visualize=False, stop=False, finetuning=False):
    """Evaluation function.
    Args:
        network (str): String identifying the network architecture to use.
        weights (str): Path string to a .h5 weights file.
    """

    siamese_net = None
    print("Loading model")
    if imagenet:
        print("    Loading imagenet weights")
    else:
        print("    Not loading imagenet weights")
    if network == 'SiameseNetTriplet':
        siamese_net = SiameseNetTriplet((None,None,3), arch='resnet18', sliding=True, imagenet=imagenet, freeze_until=freeze_until)
        model, vector_model = siamese_net.setup_weights(weights=weights, lr=0.0006, optimizer='adam')
    print("Done loading model")

    # Load data and create data generator:
    eval_ds = tf.data.Dataset.from_generator(
                image_trio_generator,
                args=[eval_lbld_trios, False, True, False, finetuning],
                output_types=((tf.float32, tf.float32, tf.float32)),
                output_shapes=(((TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3))))


    batched_eval_ds = eval_ds.batch(1)
    batched_eval_iter = iter(batched_eval_ds)

    cumm_acc_dot_tot_p = 0
    cumm_acc_dot_p = 0
    cumm_acc_dot_tot_n = 0
    cumm_acc_dot_n = 0

    cumm_acc_csim_tot_p = 0
    cumm_acc_csim_p = 0
    cumm_acc_csim_tot_n = 0
    cumm_acc_csim_n = 0

    cumm_cos_neg_tot = 0
    cumm_cos_neg = 0
    cumm_cos_pos_tot = 0
    cumm_cos_pos = 0
    imgs_processed = 0
    y_true = []
    y_pred = []

    embeddings = []

    while True:
        print("*** Trio #: " + str(imgs_processed) + " ***")
        if stop and imgs_processed > 500:
            break

        eval_inputs = next(batched_eval_iter)
        if imagenet:
            eval_a = eval_inputs[0]*255.
            eval_p = eval_inputs[1]*255.
            eval_n = eval_inputs[2]*255.
        else:
            eval_a = eval_inputs[0]
            eval_p = eval_inputs[1]
            eval_n = eval_inputs[2]

        eval_a_vec, eval_p_vec, eval_n_vec = siamese_net.vector_model.predict([eval_a, eval_p, eval_n])
        embeddings.append(eval_a_vec[0])
        embeddings.append(eval_p_vec[0])
        embeddings.append(eval_n_vec[0])
        dot_prod_p, dot_prod_n = siamese_net.get_similarity_triple([eval_a, eval_p, eval_n])

        similar_np = np.array([1])
        different_np = np.array([0])

        eval_csim_p = cos_sim_pos(None, [eval_a_vec, eval_p_vec], concat=False)
        eval_csim_n = cos_sim_neg(None, [eval_a_vec, eval_n_vec], concat=False)

        eval_acc_dot_p = compute_accuracy(similar_np, dot_prod_p, mode='dot', margin=0.5)
        eval_acc_dot_n = compute_accuracy(different_np, dot_prod_n, mode='dot', margin=0.5)
        eval_acc_csim_p = compute_accuracy(similar_np, np.array(np.abs(eval_csim_p)), mode='dot', margin=0.5)
        eval_acc_csim_n = compute_accuracy(different_np, np.array(np.abs(eval_csim_n)), mode='dot', margin=0.5)

        imgs_processed += 1
        cumm_acc_dot_tot_p += eval_acc_dot_p
        cumm_acc_dot_p = cumm_acc_dot_tot_p/imgs_processed
        cumm_acc_dot_tot_n += eval_acc_dot_n
        cumm_acc_dot_n = cumm_acc_dot_tot_n/imgs_processed

        cumm_acc_csim_tot_p += eval_acc_csim_p
        cumm_acc_csim_p = cumm_acc_csim_tot_p/imgs_processed
        cumm_acc_csim_tot_n += eval_acc_csim_n
        cumm_acc_csim_n = cumm_acc_csim_tot_n/imgs_processed

        cumm_cos_pos_tot += eval_csim_p
        cumm_cos_pos = cumm_cos_pos_tot/imgs_processed
        cumm_cos_neg_tot += eval_csim_n
        cumm_cos_neg = cumm_cos_neg_tot/imgs_processed

        y_pred_dot_p = get_label(similar_np, dot_prod_p, mode='dot')
        y_pred_dot_p = y_pred_dot_p.numpy()
        y_pred_dot_n = get_label(different_np, dot_prod_n, mode='dot')
        y_pred_dot_n = y_pred_dot_n.numpy()

        y_pred_csim_p = get_label(similar_np, np.array(np.abs(eval_csim_p)), mode='dot')
        y_pred_csim_p = y_pred_csim_p.numpy()
        y_pred_csim_n = get_label(different_np, np.array(np.abs(eval_csim_n)), mode='dot')
        y_pred_csim_n = y_pred_csim_n.numpy()

        print('    Dot product (pos): ' + str(dot_prod_p))
        print('    Dot product (neg): ' + str(dot_prod_n))
        print('    Cosine sim  (pos): ' +str(eval_csim_p))
        print('    Cosine sim  (neg): ' +str(eval_csim_n))
        print('    Predictions (dot, pos): ' + str(y_pred_dot_p))
        print('    Predictions (dot, neg): ' + str(y_pred_dot_n))
        print('    Predictions (csim, pos): ' + str(y_pred_csim_p))
        print('    Predictions (csim, neg): ' + str(y_pred_csim_n))

        true_y_p = np.array([str(int(x)) for x in similar_np])
        lbld_pair_p = np.array((eval_a, eval_p, true_y_p)).T
        true_y_n = np.array([str(int(x)) for x in different_np])
        lbld_pair_n = np.array((eval_a, eval_n, true_y_n)).T

        print('* Overall accuracy (dot, pos): %0.2f%%' % (100 * cumm_acc_dot_p))
        print('* Overall accuracy (dot, neg): %0.2f%%' % (100 * cumm_acc_dot_n))
        print('* Overall accuracy (csim, pos): %0.2f%%' % (100 * cumm_acc_csim_p))
        print('* Overall accuracy (csim, neg): %0.2f%%' % (100 * cumm_acc_csim_n))
        print('* Overall cosine similarity (pos): %0.2f' % (cumm_cos_pos))
        print('* Overall cosine similarity (neg): %0.2f' % (cumm_cos_neg))

        y_pred.append(y_pred_dot_p)
        y_pred.append(y_pred_dot_n)
        y_true.append(1)
        y_true.append(0)

        target_names = ['different', 'similar']
        print(classification_report(y_true, y_pred, target_names=target_names))


        print("*********")

        if visualize:
            visualize_pairs(lbld_pair_p, predictions=[y_pred_dot_p], window_name='anchor_pos_dot', normalized=(not finetuning))
            visualize_pairs(lbld_pair_n, predictions=[y_pred_dot_n], window_name='anchor_neg_dot', normalized=(not finetuning))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training script for learning visual similarity.')
    parser.add_argument('-p','--path-list',
                        dest='list_of_paths',
                        type=str,
                        nargs='+',
                        help='<Required> Paths to dirs or lists (.txt) of directories containing patches (e.g: /path/to/list1.txt /path/to/dir /path/to/list2.txt)',
                        required=True)
    parser.add_argument('-e','--tracked-list',
                        dest='list_tracked_vs_random',
                        type=str,
                        nargs='+',
                        help='<Required> list of 0/1 to indicate if patches were feature tracked (1) or randomly extracted (0) (e.g: 0 1 1). Must match number of different input paths were provided.',
                        required=True)
    parser.add_argument('-w', '--weights',
                        dest='weights',
                        help='Keras model',
                        default=None)
    parser.add_argument('-nw', '--network',
                        help='Network name',
                        dest='network',
                        default='SiameseNetTriplet')
    parser.add_argument('-d', '--dataset',
                        dest='dataset',
                        help='underwater or scott',
                        default=None)
    parser.add_argument('-v --visualize',
                        dest='visualize',
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
    parser.add_argument('-s --stop',
                        dest='stop',
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.dataset:
        logging.error("Dataset type not specified.")
        exit()

    PATHS_LIST = args.list_of_paths
    TRACKED_VS_RANDOM_LIST = args.list_tracked_vs_random

    if len(PATHS_LIST) != len(TRACKED_VS_RANDOM_LIST):
        logging.error("Length of paths and tracked_vs_random list do not match. Please provide a '1' or '0' for each input path provided.")
        exit()

    INPUT_PATHS = {}

    for idx, path in enumerate(PATHS_LIST):
        if os.path.isdir(path):
            patches_path = os.path.join(path, 'patches')

            if not os.path.isdir(patches_path):
                logging.error('Patches folder is missing from this directory.')
            else:
                # directory of patches
                INPUT_PATHS[idx] = [patches_path]

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
                            patches_path = os.path.join(line, 'patches')

                            if not os.path.isdir(patches_path):
                                logging.error('Patches folder is missing from this directory.')
                            else:
                                # directory of patches
                                dir_list.append(patches_path)

                INPUT_PATHS[idx] = dir_list

    cluster_ids = list(INPUT_PATHS.keys())
    cluster_params = {}
    for cluster_id in cluster_ids:
        cluster_params[cluster_id] = int(TRACKED_VS_RANDOM_LIST[cluster_id])

    tracked_vs_random_dict = {1: 'tracked', 0: 'random'}

    print("\nList of directories to process: ")
    for cluster_id, paths in INPUT_PATHS.items():
        print('    Cluster ' + str(cluster_id) + ' (' + str(tracked_vs_random_dict[cluster_params[cluster_id]]) + '):')
        for idx, input_path in enumerate(paths):
            print('        ' + str(idx) + '. ' + str(input_path) + ' [eval]')

    print("\n")

    if args.dataset == 'underwater':
        trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=250, num_patch_folders=1000)
    elif args.dataset == 'scott':
        trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=250, num_patch_folders=1000)
    elif args.dataset == 'underwater_curated':
        trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'scott_curated':
        trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'scott_agglomerative':
        trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=250, num_patch_folders=1000, curated=True)

    print("Number of trios generated: " + str(len(trios)))
    eval_lbld_trios = []

    num_examples = 100000
    random.shuffle(trios)

    # Only take a subset for now:
    if len(trios) > int(num_examples):
        trios = trios[:int(num_examples)]

    eval_lbld_trios.extend(trios)

    logging.info("[EVAL] Number of trios: " + str(len(trios)))
    random.shuffle(eval_lbld_trios)

    evaluate(eval_lbld_trios,
             args.network,
             args.weights,
             imagenet=args.imagenet,
             freeze_until=args.freeze_until,
             visualize=args.visualize,
             stop=args.stop,
             finetuning=args.finetuning)
