"""
Python generator for trios of images that are anchor/similar/different.
E.g: (im_query, im_similar, im_diff) also called (anchor, positive, negative).

When building trios, it doesn't match up randomly extracted patches with each other.

"""

# BASIC
import os
import glob
import random
import argparse
import itertools

# LOGGING
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
import tensorflow as tf
from utils.vis_utils import *
from network.layers.losses import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_WIDTH = 128
TARGET_HEIGHT = 128
VECTOR_SIZE = 4*4*512 # resnet18
# VECTOR_SIZE = 512*88 # mobilenetv2
RANDOM_SEED = 42


def get_batch_hard(network, X, seq_ids, batch_size, hard_batch_size=16, norm_batch_size=16):
    """
    Create batch of APN "hard" triplets

    Arguments:
    batch_size -- integer: number of random samples
    hard_batchs_size -- integer : select the number of hardest samples to keep
    norm_batchs_size -- integer : number of random samples to keep
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batch_size+norm_batch_size,w,h,c)
    """
    w = 128
    h = 128
    c = 3

    # compute the loss with current network : d(A,P)-d(A,N)
    loss = np.zeros((batch_size))

    #Compute embeddings for anchors, positive and negatives
    merged_vecs = network.predict_on_batch([X[0], X[1], X[2]])
    anchor = merged_vecs[:, :VECTOR_SIZE]
    positive = merged_vecs[:, VECTOR_SIZE:2*VECTOR_SIZE]
    negative = merged_vecs[:, 2*VECTOR_SIZE:]

    margin = 0.5
    pos_dist = tf.keras.losses.cosine_similarity(anchor, positive)
    neg_dist = tf.keras.losses.cosine_similarity(anchor, negative)
    loss = tf.keras.backend.clip(pos_dist - neg_dist + margin, 0.0, None)
    selection = np.argsort(loss)[::-1][:hard_batch_size]

    # Draw random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(batch_size), selection), norm_batch_size, replace=False)
    selection = np.append(selection, selection2)

    triplets = [X[0].numpy()[selection], X[1].numpy()[selection], X[2].numpy()[selection]]

    return triplets


def get_batch_random(X, seq_ids, batch_size):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    X:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """

    w = 128
    h = 128
    c = 3

    # initialize result
    triplets = [np.zeros((batch_size, h, w, c)) for i in range(3)]

    for i in range(batch_size):
        # Pick one random class for anchor
        anchor_class = random.sample(seq_ids, 1)
        anchor_class = anchor_class[0]
        anchor_class_id = seq_ids.index(anchor_class)
        nb_sample_available_for_class_AP = len(X[anchor_class])

        # pick two different random pics for this class (A and P)
        [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP, size=2, replace=False)

        # pick another class for N, different from anchor_class
        negative_class_id = (anchor_class_id + np.random.randint(1, len(seq_ids))) % len(seq_ids)
        negative_class = seq_ids[negative_class_id]
        nb_sample_available_for_class_N = len(X[negative_class])

        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i,:,:,:] = X[anchor_class][idx_A]
        triplets[1][i,:,:,:] = X[anchor_class][idx_P]
        triplets[2][i,:,:,:] = X[negative_class][idx_N]

    return triplets

def image_trio_generator(img_objs, with_aug=True, inference=False, single=False, imagenet=False):
    """ Keras based image trio generator. Leverages Keras' image transforms.

    The input is a list of [img1_path, img2_path, img3_path] lists where im1 is
    the anchor/query, im2 is similar to im1, and im3 is different than im1.

    Yields a trio of images. E.g: (im1, im2, im3).

    """

    training = True
    if inference:
        training = False

    if with_aug:
        datagen_args = dict(rotation_range=20, # cahnged from 20
                            # width_shift_range=0.2,
                            # height_shift_range=0.2,
                            shear_range=0.15,
                            zoom_range=0.15,
                            horizontal_flip=True,
                            zca_whitening=True)
        # datagen_args = {'theta': 20.,
        #                 'shear': 5.,
        #                 'flip_horizontal': True,
        #                 'brightness':0.8,
        #                 'zx':0.8,
        #                 'zy':0.8}
        # datagen_args = {}
    else:
        datagen_args = {}

    datagen_1 = ImageDataGenerator(**datagen_args)
    if not single:
        datagen_2 = ImageDataGenerator(**datagen_args)
        datagen_3 = ImageDataGenerator(**datagen_args)

    # global image_cache
    image_cache = dict()

    while True:
        for img_obj in img_objs:

            # Ensure same transformations for image trios by controlling random seed
            seed = np.random.randint(low=0, high=1000, size=1)[0]

            image_cache, X1 = preprocess_image(img_obj[0],
                                               seed,
                                               datagen_1,
                                               datagen_args,
                                               image_cache,
                                               width=TARGET_WIDTH,
                                               height=TARGET_HEIGHT,
                                               training=training,
                                               imagenet=imagenet)
            if not single:
                image_cache, X2 = preprocess_image(img_obj[1],
                                                   seed,
                                                   datagen_2,
                                                   datagen_args,
                                                   image_cache,
                                                   width=TARGET_WIDTH,
                                                   height=TARGET_HEIGHT,
                                                   training=training,
                                                   imagenet=imagenet)

                image_cache, X3 = preprocess_image(img_obj[2],
                                                   seed,
                                                   datagen_3,
                                                   datagen_args,
                                                   image_cache,
                                                   width=TARGET_WIDTH,
                                                   height=TARGET_HEIGHT,
                                                   training=training,
                                                   imagenet=imagenet)
            if not inference:
                if single:
                    yield X1[0], np.empty((1, VECTOR_SIZE)), int(float(img_obj[1].decode('utf8')))
                else:
                    # print(img_obj[3])
                    seq_ids = (img_obj[3], img_obj[4], img_obj[5])

                    if X1 is not None and X2 is not None and X3 is not None:
                        yield (X1[0], X2[0], X3[0]), np.empty((1, VECTOR_SIZE * 3)), seq_ids
                        # yield (X1[0], X2[0], X3[0]), y
                    else:
                        pass
            else:
                if X1 is not None and X2 is not None and X3 is not None:
                    yield (X1[0], X2[0], X3[0])
                else:
                    pass





def build_trios(img_seq1, img_seq2, id_1, id_2):
    trios = np.array(np.meshgrid(img_seq1, img_seq1, img_seq2)).T.reshape(-1,3)
    trios = [(x[0],x[1],x[2],id_1,id_1,id_2) for x in trios.tolist() if x[0] != x[1]]
    return trios, len(trios)


def build_lbld_trios(input_dict, cluster_params, max_neg_folder_pairs=None, num_patch_folders=50, first_last=True, curated=False):
    cluster_ids = list(input_dict.keys())
    # print("Cluster ids: " + str(cluster_ids))
    trios_all = []

    # Build trios:
    logging.info("Building trios..")
    cluster_pairs = [(cluster_ids[i], cluster_ids[j]) for i in range(len(cluster_ids)) if cluster_params[i] == 1 for j in range(i, len(cluster_ids))]
    logging.debug("Cluster pairs: " + str(cluster_pairs))
    # print("Cluster pairs: " + str(cluster_pairs))

    for c_idx, pair in enumerate(cluster_pairs):
        logging.debug("C idx: " + str(c_idx))
        seq_pairs = np.array(np.meshgrid(input_dict[pair[0]], input_dict[pair[1]])).T.reshape(-1,2)
        seq_pairs = [tuple(x) for x in seq_pairs.tolist()]
        seq_pairs = [x for x in seq_pairs if x[0] != x[1]]
        # print(seq_pairs)

        for s_idx, seq_pair in enumerate(seq_pairs):
            logging.debug("S idx: " + str(s_idx) + "/" + str(len(seq_pairs)))

            if curated:
                id_1 = seq_pair[0].split('/')
                id_1 = id_1[-2]
                id_2 = seq_pair[1].split('/')
                id_2 = id_2[-2]

                patch_imgs_1 = sorted(glob.glob(os.path.join(seq_pair[0], '*.png')))
                patch_imgs_2 = sorted(glob.glob(os.path.join(seq_pair[1], '*.png')))

                if len(patch_imgs_1) > num_patch_folders:
                    patch_imgs_1 = random.sample(patch_imgs_1, num_patch_folders)
                if len(patch_imgs_2) > max_neg_folder_pairs:
                    patch_imgs_2 = random.sample(patch_imgs_2, max_neg_folder_pairs)

                trios, num_examples = build_trios(patch_imgs_1, patch_imgs_2, id_1, id_2)
                trios_all.extend(trios)
                trios = []

            else:
                patch_folders_1 = [f.path for f in os.scandir(seq_pair[0]) if f.is_dir()]
                patch_folders_2 = [f.path for f in os.scandir(seq_pair[1]) if f.is_dir()]
                if len(patch_folders_1) > num_patch_folders:
                    patch_folders_1 = random.sample(patch_folders_1, num_patch_folders)
                if len(patch_folders_2) > num_patch_folders:
                    patch_folders_2 = random.sample(patch_folders_2, num_patch_folders)

                patch_folder_pairs = np.array(np.meshgrid(patch_folders_1, patch_folders_2)).T.reshape(-1,2)
                patch_folder_pairs = [tuple(x) for x in patch_folder_pairs.tolist()]

                if max_neg_folder_pairs:
                    if len(patch_folder_pairs) > max_neg_folder_pairs:
                        patch_folder_pairs = random.sample(patch_folder_pairs, max_neg_folder_pairs)

                patch_folder_pairs = [x for x in patch_folder_pairs if x[0] != x[1]]

                for p_idx, patch_folder_pair in enumerate(patch_folder_pairs):
                    id_1 = patch_folder_pair[0].split('/')
                    id_1 = id_1[-3] + '_' + id_1[-1]
                    id_2 = patch_folder_pair[1].split('/')
                    id_2 = id_2[-3] + '_' + id_2[-1]
                    logging.debug("P idx: " + str(p_idx))

                    if first_last:
                        patch_imgs_1 = [sorted(glob.glob(os.path.join(patch_folder_pair[0], '*.png')))[0], sorted(glob.glob(os.path.join(patch_folder_pair[0], '*.png')))[-1]]
                        patch_imgs_2 = [sorted(glob.glob(os.path.join(patch_folder_pair[1], '*.png')))[0]]
                    else:
                        patch_imgs_1 = sorted(glob.glob(os.path.join(patch_folder_pair[0], '*.png')))
                        patch_imgs_2 = sorted(glob.glob(os.path.join(patch_folder_pair[1], '*.png')))

                    trios, num_examples = build_trios(patch_imgs_1, patch_imgs_2, id_1, id_2)
                    trios_all.extend(trios)

    logging.info("Number of trios: " + str(len(trios_all)))
    return trios_all



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Load patch sequences and generate trios (anchor/similar/different).')
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

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
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
            print('        ' + str(idx) + '. ' + str(input_path))
    print("\n")


    trios = build_lbld_trios(INPUT_PATHS, cluster_params, max_neg_folder_pairs=3, num_patch_folders=20)
    num_trio_examples = 50000
    random.shuffle(trios)

    # Only take a subset for now:
    if len(trios) > num_trio_examples:
        trios = trios[:num_trio_examples]

    random.shuffle(trios)
    visualize_trios(trios, normalized=False)

    image_trio_gen = image_trio_generator(trios, with_aug=True, inference=False, single=False, imagenet=False)

    for idx, trio in enumerate(trios):
        print(f"Processed {idx}/{len(trios)}")
        X = image_trio_gen.__next__()
        visualize_trios([(X[0][0],X[0][1], X[0][2])], normalized=True)

        if idx == 10:
            exit()
