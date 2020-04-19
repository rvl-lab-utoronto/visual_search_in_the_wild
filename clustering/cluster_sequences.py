"""
Python script to cluster patch sequences

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
from utils.vis_utils import *

import tensorflow as tf
from tensorflow.keras import Model
from network.layers.losses import *
from network.layers.metrics import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from network.siamese_nets import SiameseNetTriplet
from generators.similar_trios_generator import image_trio_generator, build_lbld_trios

from sklearn.cluster import *


def get_embeddings(input_dir, weights=None, network='SiameseNetTriplet', img_by_img=True, scale_x=1, scale_y=1, patch_dims=(128,128), imagenet=True, freeze_until=None, visualize=False):

    embeddings = []
    sequences = []
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

    patches_path = os.path.join(input_dir, 'patches')
    main_dir = os.path.basename(input_dir)

    if not os.path.isdir(patches_path):
        logging.error('Patches folder is missing from this directory.')
        return

    patch_folders = [f.path for f in os.scandir(patches_path) if f.is_dir()]

    bucket_style = False
    if len(patch_folders) == 0:
        patch_folders = [patches_path]
        bucket_style = True

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    color = (0,0,255)

    for patch_idx, patch_folder in enumerate(patch_folders):
        if patch_idx % 500 == 0:
            print("\t\tPatch folder #" + str(patch_idx) + "/" + str(len(patch_folders)))

        patch_imgs = sorted(glob.glob(os.path.join(patch_folder, '*.png')))
        base_name = os.path.basename(patch_folder)

        images = []
        img_paths = []

        for patch in patch_imgs:
            img_paths.append(patch)
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
        if visualize:
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
            cv2.waitKey(0)

        if not imagenet:
            for idx, img in enumerate(images):
                images[idx] = img/255.

        if bucket_style:
            print("Embedding a general folder: ")
            for idx, img in enumerate(images):
                if idx % 250 == 0:
                    print("\tAt image #" + str(idx) + "/" + str(len(images)))
                embedding_vec, _, _ = siamese_net.vector_model.predict([np.array([img]), np.array([img]), np.array([img])])
                embedding_vec = embedding_vec[0]
                sequence = {'first_img': img_paths[idx], 'last_img': img_paths[idx], 'embedding': embedding_vec}
                sequences.append(sequence)
                embeddings.append(embedding_vec)
        else:
            # print("Embedding a patch sequence: ")
            first_patch = images[0]
            last_patch = images[-1]
            embedding_vec_first, embedding_vec_first, embedding_vec_last = siamese_net.vector_model.predict([np.array([first_patch]), np.array([first_patch]), np.array([last_patch])])

            embedding_vec = embedding_vec_first[0]

            sequence = {'first_img': img_paths[0], 'last_img': img_paths[-1], 'embedding': embedding_vec}
            sequences.append(sequence)
            embeddings.append(embedding_vec)

    if visualize:
        cv2.destroyAllWindows()

    return embeddings, sequences

def fit_cluster(embeddings, method='AgglomerativeClustering', n_clusters=None, n_components=None):

    print("Clustering...")

    if method == 'kmeans':
        trained_cluster_obj = KMeans(n_clusters=n_clusters,
                                     n_init=10,
                                     n_jobs=-1).fit(embeddings)
    elif method == 'SpectralClustering':
        trained_cluster_obj = SpectralClustering(n_clusters=n_clusters,
                                                 n_init=10,
                                                 gamma=5,
                                                 affinity='rbf',
                                                 eigen_tol=0.0,
                                                 assign_labels='kmeans',
                                                 n_jobs=-1).fit(embeddings)
    elif method == 'DBSCAN':
        trained_cluster_obj = DBSCAN(eps=0.25,
                                     min_samples=150,
                                     metric='l2',
                                     n_jobs=-1).fit(embeddings)
        labels = trained_cluster_obj.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    elif method == 'OPTICS':
        trained_cluster_obj = OPTICS(min_samples=25, cluster_method='dbscan', metric='l2', n_jobs=-1).fit(embeddings)
        labels = trained_cluster_obj.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    elif method == 'AgglomerativeClustering':
        trained_cluster_obj = AgglomerativeClustering(n_clusters=None,
                                                      linkage='average',
                                                      distance_threshold=0.95,
                                                      affinity='l2').fit(embeddings)
        labels = trained_cluster_obj.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


    print("Fitted " + str(n_clusters) + " clusters with " + str(method))

    return trained_cluster_obj

def cluster_embeddings(sequences, clustering_obj, cluster_dir_path, n_clusters=None):

    print("Clustering all sequences")
    clusters_count = {}
    clusters_count[-1] = 0
    for cluster_id in range(n_clusters):
        clusters_count[cluster_id] = 0

    for idx, seq in enumerate(sequences):
        if idx % 50 == 0:
            print("Sequence #" + str(idx) + "/" + str(len(sequences)))
        first_patch_path = seq['first_img']
        last_patch_path = seq['last_img']
        first_patch_fname = os.path.basename(first_patch_path)
        last_patch_fname = os.path.basename(last_patch_path)

        first_patch = cv2.imread(first_patch_path)
        last_patch = cv2.imread(last_patch_path)

        embedding_vec = seq['embedding']
        pred_id = clustering_obj.labels_[idx]

        clusters_count[pred_id] += 1
        print("\t" + str(first_patch_path) + " labeled as cluster: " + str(pred_id))
        patch_dir = os.path.join(cluster_dir_path, "{:04d}".format(pred_id), 'patches')
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        cv2.imwrite(os.path.join(patch_dir, first_patch_fname), first_patch)
        cv2.imwrite(os.path.join(patch_dir, last_patch_fname), last_patch)

    print("Done clustering all sequences")

    return clusters_count

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Script to cluster patch sequences')
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
    parser.add_argument('-pd', '--pickle-dir',
                        help='Path to dir containing embeddings and sequences',
                        dest='pickle_dir',
                        default=None)
    parser.add_argument('-m', '--method',
                        help='Clustering method',
                        dest='method',
                        default='AgglomerativeClustering')
    parser.add_argument('-nc', '--num-clusters',
                        dest='n_clusters',
                        type=int,
                        default=None)
    parser.add_argument('-v --visualize',
                        dest='visualize',
                        action='store_true')
    parser.add_argument('-s --subcluster',
                        dest='subcluster',
                        action='store_true')


    args = parser.parse_args()

    return args

if __name__ == "__main__":
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
    for bucket_id, paths in INPUT_PATHS.items():
        print('    Bucket ' + str(bucket_id) + ':')
        for idx, input_path in enumerate(paths):
            print('        ' + str(idx) + '. ' + str(input_path))
    print("\n")

    output_dir = args.output
    if args.subcluster:
        args.output = os.path.join(args.output, args.method+'_subclusters')
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    if not os.path.isdir(args.output):
        logging.error('Output must be a directory with classes folders already created.')
        exit()


    # print("\n")
    clusters_count = {}
    # n_clusters = len(list(final_cluster_folders.keys()))
    n_clusters = args.n_clusters
    embeddings = []
    sequences = []

    # Get all embeddings:
    if args.pickle_dir:
        embedding_pkl_file = os.path.join(args.pickle_dir, 'train_embedding_file.pkl')
        sequences_pkl_file = os.path.join(args.pickle_dir, 'train_sequences_file.pkl')
    else:
        embedding_pkl_file = os.path.join(output_dir, 'train_embedding_file.pkl')
        sequences_pkl_file = os.path.join(output_dir, 'train_sequences_file.pkl')

    if os.path.isfile(embedding_pkl_file) and os.path.isfile(sequences_pkl_file):
        with open(embedding_pkl_file, 'rb') as handle:
            embeddings = pkl.load(handle)
            print("Embeddings: " + str(len(embeddings)))
            embeddings = embeddings[:20000]
        with open(sequences_pkl_file, 'rb') as handle:
            sequences = pkl.load(handle)
            print("Sequences: " + str(len(sequences)))
            sequences = sequences[:20000]

    else:
        for bucket_id, paths in INPUT_PATHS.items():
            print('Bucket ' + str(bucket_id) + ':')
            for idx, input_path in enumerate(paths):
                print('\t' + str(idx) + '. ' + str(input_path))
                new_embeddings, new_seqs = get_embeddings(input_path, img_by_img=False, visualize=args.visualize)
                embeddings.extend(new_embeddings)
                sequences.extend(new_seqs)
                print("\t# of embedded patch sequences: " + str(len(embeddings)))

        with open(embedding_pkl_file, 'wb') as handle:
            pkl.dump(embeddings, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(sequences_pkl_file, 'wb') as handle:
            pkl.dump(sequences, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Cluster embeddings:
    if args.method in ['kmeans', 'SpectralClustering']:
        trained_clustering_obj = fit_cluster(embeddings, method=args.method, n_clusters=n_clusters)
        counts = cluster_embeddings(sequences, trained_clustering_obj, args.output, n_clusters=n_clusters)

    elif args.method in ['DBSCAN', 'OPTICS', 'AgglomerativeClustering']:
        trained_clustering_obj = fit_cluster(embeddings, method=args.method)
        labels = trained_clustering_obj.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        counts = cluster_embeddings(sequences, trained_clustering_obj, args.output, n_clusters=n_clusters)

    print("Overall counts: " + str(clusters_count))
